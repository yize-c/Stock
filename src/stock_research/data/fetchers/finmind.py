"""FinMind API v4 client: https://api.finmindtrade.com/api/v4/data

Responses use the envelope ``{"msg": ..., "status": ..., "data": [...]}``. Status 402 means
the request quota is used up; status 400 with "Your level is register" means the dataset needs
a paid tier (``TaiwanStockPriceAdj``, ``TaiwanStockSuspended`` and whole-market capital
reductions do). The token is sent as a Bearer header and is never part of the cache key or the
stored payload.

Field names and quirks below come from real responses (``tests/fixtures/sources/finmind_*``):

* ``TaiwanStockPrice`` encodes an absent price as ``0``. On days without trades (volume 0)
  the price fields are leftovers (e.g. ``open`` = previous close, the rest 0), so every
  price on such a day becomes NaN and ``no_trade`` is set. A ``0`` on a traded day becomes NaN
  too, and validation then reports it; nothing is filled in.
* ``TaiwanStockFinancialStatements`` (income) values are single-quarter figures; Q4 is
  derived by FinMind as the year minus the first three quarters.
* ``TaiwanStockCashFlowsStatement`` values are cumulative from 1 January (year to date).
* ``TaiwanStockBalanceSheet`` rows ending in ``_per`` are percentages of total assets; they
  are derived numbers and dropped (code computes its own ratios).
* No statement dataset carries a publication date, and FinMind serves only the latest
  (possibly restated) version. Rows are therefore marked ``latest_only`` and their
  ``available_date`` is the statutory deadline (``available_date_estimated``).
* ``TaiwanStockMonthRevenue.date`` is the first day of the month *after* the revenue month;
  ``create_time`` is when FinMind stored the row (empty for older rows).
* ``TaiwanStockDividendResult.before_price`` equals the previous trading day's raw close.
  Its ``date`` is the scheduled ex-date; if the market was closed that day (e.g. a typhoon),
  the event takes effect on the next trading day.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

import pandas as pd

from stock_research.config import FilingDeadlineConfig
from stock_research.data.cache import Cache
from stock_research.data.errors import AnomalyError, MissingDataError
from stock_research.data.http import JsonClient, SourceRateLimitedError, SourceUnavailableError
from stock_research.data.point_in_time import PeriodType, VersionKind, statutory_available_date
from stock_research.markets.tickers import Market

SOURCE = "finmind"
URL = "https://api.finmindtrade.com/api/v4/data"
TAIEX = "TAIEX"
PRICE_COLUMNS = ("open", "high", "low", "close")


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def check_envelope(body: Any) -> None:
    if not isinstance(body, dict) or "status" not in body:
        raise SourceUnavailableError(SOURCE, URL, str(body)[:200], "a FinMind response envelope")
    status = body.get("status")
    if status == 402:
        raise SourceRateLimitedError(
            SOURCE, URL, body.get("msg"), "request quota available (status 402 = limit reached)"
        )
    if status != 200:
        raise SourceUnavailableError(SOURCE, URL, f"{status}: {body.get('msg')}", "status 200")
    if not isinstance(body.get("data"), list):
        raise SourceUnavailableError(SOURCE, URL, type(body.get("data")).__name__, "a data list")


class FinMindClient:
    def __init__(self, client: JsonClient, max_age: timedelta) -> None:
        self._client = client
        self._max_age = max_age

    def dataset(
        self,
        dataset: str,
        *,
        data_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        allow_empty: bool = False,
    ) -> pd.DataFrame:
        """Raw rows of one dataset as a DataFrame.

        An empty result raises unless ``allow_empty`` (for event datasets, where "no events"
        is a valid answer).
        """
        params: dict[str, str] = {"dataset": dataset}
        if data_id is not None:
            params["data_id"] = data_id
        if start_date is not None:
            params["start_date"] = start_date.isoformat()
        if end_date is not None:
            params["end_date"] = end_date.isoformat()
        result = self._client.get_json(URL, params, max_age=self._max_age, check=check_envelope)
        rows = result.data["data"]
        if not rows and not allow_empty:
            raise MissingDataError(data_id, f"{SOURCE}:{dataset}", "no rows", "at least one row")
        frame = pd.DataFrame(rows)
        frame["source"] = SOURCE
        return frame


# --- normalizers (pure functions over raw dataset frames) -----------------------------------


class Statement(StrEnum):
    INCOME = "income"
    BALANCE = "balance"
    CASH_FLOW = "cash_flow"


STATEMENT_DATASETS = {
    Statement.INCOME: "TaiwanStockFinancialStatements",
    Statement.BALANCE: "TaiwanStockBalanceSheet",
    Statement.CASH_FLOW: "TaiwanStockCashFlowsStatement",
}

_MARKETS = {"twse": Market.TWSE, "tpex": Market.TPEX}
_INDEX_CATEGORIES = ("Index", "大盤")


class EventKind(StrEnum):
    """Corporate events that change the raw price basis on their effective day."""

    EX_RIGHTS_DIVIDEND = "ex_rights_dividend"
    CAPITAL_REDUCTION = "capital_reduction"
    SPLIT = "split"  # includes reverse splits and par-value changes


def _dates(values: pd.Series, field: str, ticker: str | None) -> list[date]:
    out: list[date] = []
    for value in values:
        try:
            out.append(date.fromisoformat(str(value)))
        except ValueError as exc:
            raise AnomalyError(ticker, field, value, "an ISO date (YYYY-MM-DD)") from exc
    return out


def _require(frame: pd.DataFrame, columns: tuple[str, ...], ticker: str | None, what: str) -> None:
    missing = [c for c in columns if c not in frame]
    if missing:
        raise MissingDataError(ticker, f"{SOURCE}:{what}.columns", missing, f"columns {columns}")


def normalize_stock_info(raw: pd.DataFrame) -> pd.DataFrame:
    """Listed (TWSE) and OTC (TPEx) securities, one row per code.

    FinMind lists a security once per industry it belongs to and keeps stale rows from
    earlier listings (e.g. a TPEx row for a stock that later moved to TWSE). Only the rows
    with the newest date count: they give the current market and ``industries``. Codes whose
    newest rows are emerging-market (興櫃) are out of scope and dropped; so are the index
    rows (TAIEX and sector indices), which carry the date ``"None"``. If the newest rows name
    two markets, ``market`` is None and resolving that code fails.
    """
    _require(raw, ("stock_id", "stock_name", "type", "industry_category", "date"), None, "info")
    frame = raw.loc[~raw["industry_category"].isin(_INDEX_CATEGORIES)].copy()
    frame["date"] = _dates(frame["date"], "info.date", None)
    rows = []
    for code, group in frame.groupby("stock_id", sort=True):
        newest = group.loc[group["date"] == group["date"].max()]
        markets = set(newest["type"])
        if not markets & set(_MARKETS):
            continue
        rows.append(
            {
                "code": str(code),
                "name": str(newest["stock_name"].iloc[0]),
                "market": _MARKETS[markets.pop()].value if len(markets) == 1 else None,
                "listed_types": tuple(sorted(set(newest["type"]))),
                "industries": tuple(sorted(set(newest["industry_category"]))),
                "data_date": newest["date"].iloc[0],
                "source": SOURCE,
            }
        )
    return pd.DataFrame(rows)


def normalize_prices(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Raw (unadjusted) daily OHLCV from ``TaiwanStockPrice``. See module notes on zeros."""
    _require(
        raw,
        ("date", "open", "max", "min", "close", "Trading_Volume", "Trading_money"),
        ticker,
        "price",
    )
    frame = pd.DataFrame(
        {
            "date": _dates(raw["date"], "price.date", ticker),
            "open": raw["open"].astype("float64"),
            "high": raw["max"].astype("float64"),
            "low": raw["min"].astype("float64"),
            "close": raw["close"].astype("float64"),
            "volume": raw["Trading_Volume"].astype("float64"),
            "turnover": raw["Trading_money"].astype("float64"),
        }
    )
    no_trade = frame["volume"] == 0
    for col in PRICE_COLUMNS:
        frame[col] = frame[col].mask(no_trade | (frame[col] == 0))
    frame["no_trade"] = no_trade
    frame["source"] = SOURCE
    return frame.sort_values("date").reset_index(drop=True)


def normalize_index_closes(raw: pd.DataFrame, data_id: str) -> pd.DataFrame:
    """Index closes from ``TaiwanStockPrice`` (index volume can be 0; closes are kept)."""
    _require(raw, ("date", "close"), data_id, "index")
    frame = pd.DataFrame(
        {"date": _dates(raw["date"], "index.date", data_id), "close": raw["close"].astype(float)}
    )
    bad = frame["close"].isna() | (frame["close"] <= 0)
    if bad.any():
        days = [d.isoformat() for d in frame.loc[bad, "date"]][:5]
        raise AnomalyError(data_id, "index.close", days, "a positive close on every day")
    frame["source"] = SOURCE
    return frame.sort_values("date").reset_index(drop=True)


def normalize_total_return_index(raw: pd.DataFrame) -> pd.DataFrame:
    """``TaiwanStockTotalReturnIndex``: the dividend-inclusive benchmark level."""
    _require(raw, ("date", "price"), TAIEX, "total_return_index")
    frame = pd.DataFrame(
        {
            "date": _dates(raw["date"], "total_return_index.date", TAIEX),
            "total_return_index": raw["price"].astype(float),
        }
    )
    bad = frame["total_return_index"].isna() | (frame["total_return_index"] <= 0)
    if bad.any():
        days = [d.isoformat() for d in frame.loc[bad, "date"]][:5]
        raise AnomalyError(TAIEX, "total_return_index", days, "a positive level on every day")
    frame["source"] = SOURCE
    return frame.sort_values("date").reset_index(drop=True)


def normalize_trading_dates(raw: pd.DataFrame) -> list[date]:
    _require(raw, ("date",), None, "trading_date")
    return sorted(set(_dates(raw["date"], "trading_date.date", None)))


def _events(
    ticker: str,
    kind: EventKind,
    days: list[date],
    before: pd.Series,
    reference: pd.Series,
    detail: pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": days,
            "kind": kind.value,
            "close_before": before.astype(float).to_numpy(),
            "reference_price": reference.astype(float).to_numpy(),
            "detail": detail.astype(str).to_numpy(),
            "source": SOURCE,
        }
    )
    bad = ~((frame["close_before"] > 0) & (frame["reference_price"] > 0))
    if bad.any():
        raise AnomalyError(
            ticker,
            f"{kind}.price",
            [d.isoformat() for d in frame.loc[bad, "date"]][:5],
            "positive close_before and reference_price",
        )
    return frame


EVENT_COLUMNS = ["date", "kind", "close_before", "reference_price", "detail", "source"]


def normalize_dividend_results(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Ex-rights / ex-dividend reference prices (``TaiwanStockDividendResult``)."""
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    _require(
        raw, ("date", "before_price", "reference_price", "stock_or_cache_dividend"), ticker, "ex"
    )
    return _events(
        ticker,
        EventKind.EX_RIGHTS_DIVIDEND,
        _dates(raw["date"], "ex.date", ticker),
        raw["before_price"],
        raw["reference_price"],
        raw["stock_or_cache_dividend"],
    )


def normalize_capital_reductions(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """``TaiwanStockCapitalReductionReferencePrice``; ``date`` is the day trading resumes."""
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    _require(
        raw,
        (
            "date",
            "ClosingPriceonTheLastTradingDay",
            "PostReductionReferencePrice",
            "ReasonforCapitalReduction",
        ),
        ticker,
        "capital_reduction",
    )
    return _events(
        ticker,
        EventKind.CAPITAL_REDUCTION,
        _dates(raw["date"], "capital_reduction.date", ticker),
        raw["ClosingPriceonTheLastTradingDay"],
        raw["PostReductionReferencePrice"],
        raw["ReasonforCapitalReduction"],
    )


def normalize_splits(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """``TaiwanStockSplitPrice``: splits, reverse splits and par-value changes."""
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    _require(raw, ("date", "type", "before_price", "after_price"), ticker, "split")
    return _events(
        ticker,
        EventKind.SPLIT,
        _dates(raw["date"], "split.date", ticker),
        raw["before_price"],
        raw["after_price"],
        raw["type"],
    )


def _quarter_start(period_end: date) -> date:
    if period_end.month not in (3, 6, 9, 12) or (period_end + timedelta(days=1)).day != 1:
        raise AnomalyError(None, "period_end", period_end.isoformat(), "a calendar quarter end")
    return date(period_end.year, period_end.month - 2, 1)


def normalize_statement(
    raw: pd.DataFrame, ticker: str, statement: Statement, deadlines: FilingDeadlineConfig
) -> pd.DataFrame:
    """Long-format statement rows with point-in-time columns (see module notes)."""
    _require(raw, ("date", "type", "value", "origin_name"), ticker, statement.value)
    frame = raw.loc[~raw["type"].astype(str).str.endswith("_per")]
    frame = frame.loc[frame["type"].astype(str) != "-"]  # unnamed pre-IFRS lines
    ends = _dates(frame["date"], f"{statement}.date", ticker)
    if statement is Statement.INCOME:
        starts = [_quarter_start(e) for e in ends]
        types = [PeriodType.QUARTER.value] * len(ends)
    elif statement is Statement.CASH_FLOW:
        for e in ends:
            _quarter_start(e)
        starts = [date(e.year, 1, 1) for e in ends]
        types = [
            (PeriodType.ANNUAL if e.month == 12 else PeriodType.YEAR_TO_DATE).value for e in ends
        ]
    else:
        for e in ends:
            _quarter_start(e)
        starts = list(ends)
        types = [PeriodType.INSTANT.value] * len(ends)
    out = pd.DataFrame(
        {
            "ticker": ticker,
            "statement": statement.value,
            "item": frame["type"].astype(str).to_numpy(),
            "item_name": frame["origin_name"].astype(str).to_numpy(),
            "period_type": types,
            "period_start": starts,
            "period_end": ends,
            "value": frame["value"].astype(float).to_numpy(),
            "available_date": [statutory_available_date(e, "quarter", deadlines) for e in ends],
            "available_date_estimated": True,
            "version_kind": VersionKind.LATEST_ONLY.value,
            "source": SOURCE,
        }
    )
    dupes = out.duplicated(["item", "period_end"], keep=False)
    if dupes.any():
        sample = out.loc[dupes, ["item", "period_end"]].astype(str).agg("@".join, axis=1)
        raise AnomalyError(ticker, f"{statement}.item", sample.tolist()[:5], "one value per item")
    return out.sort_values(["period_end", "item"]).reset_index(drop=True)


def normalize_monthly_revenue(
    raw: pd.DataFrame, ticker: str, deadlines: FilingDeadlineConfig
) -> pd.DataFrame:
    """One row per revenue month.

    ``available_date`` is the earlier of FinMind's ``create_time`` and the statutory day
    (the 10th of the next month). Both are upper bounds on the real announcement date, so the
    earlier one is still conservative. It is always flagged as estimated, and
    ``available_date_basis`` records which bound was used.
    """
    _require(raw, ("date", "revenue", "revenue_year", "revenue_month", "create_time"), ticker, "rv")
    rows = []
    for rec in raw.itertuples(index=False):
        year, month = int(rec.revenue_year), int(rec.revenue_month)
        period_start = date(year, month, 1)
        period_end = (period_start + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        listed = date.fromisoformat(str(rec.date))
        if listed != period_end + timedelta(days=1):
            raise AnomalyError(
                ticker, "revenue.date", str(rec.date), f"first day after {year}-{month:02d}"
            )
        deadline = statutory_available_date(period_end, PeriodType.MONTH, deadlines)
        created = date.fromisoformat(str(rec.create_time)) if rec.create_time else None
        if created is not None and created < period_end:
            raise AnomalyError(
                ticker, "revenue.create_time", created.isoformat(), f"after {period_end}"
            )
        use_created = created is not None and created < deadline
        rows.append(
            {
                "ticker": ticker,
                "item": "revenue",
                "period_type": PeriodType.MONTH.value,
                "period_start": period_start,
                "period_end": period_end,
                "value": float(rec.revenue),
                "available_date": created if use_created else deadline,
                "available_date_estimated": True,
                "available_date_basis": "source_create_time" if use_created else "statutory",
                "version_kind": VersionKind.LATEST_ONLY.value,
                "source": SOURCE,
            }
        )
    out = pd.DataFrame(rows)
    if out["period_end"].duplicated().any():
        dupes = [d.isoformat() for d in out.loc[out["period_end"].duplicated(), "period_end"]]
        raise AnomalyError(ticker, "revenue.period_end", dupes[:5], "one row per month")
    return out.sort_values("period_end").reset_index(drop=True)


_FISCAL_RE = re.compile(r"^(\d{2,3})年(?:(第[1-4]季)|(前半年度|後半年度))?$")


def parse_fiscal_period(label: str, ticker: str | None = None) -> tuple[date, date, PeriodType]:
    """ROC fiscal-period labels: ``107年``, ``108年第1季``, ``112年前半年度``/``後半年度``."""
    match = _FISCAL_RE.match(str(label).strip())
    if match is None:
        raise AnomalyError(ticker, "dividend.year", label, "'NNN年', 'NNN年第Q季' or half-year")
    year = int(match.group(1)) + 1911
    if match.group(2):
        quarter = int(match.group(2)[1])
        end = _month_end(year, quarter * 3)
        return date(year, quarter * 3 - 2, 1), end, PeriodType.QUARTER
    if match.group(3) == "前半年度":
        return date(year, 1, 1), date(year, 6, 30), PeriodType.YEAR_TO_DATE
    if match.group(3) == "後半年度":
        return date(year, 7, 1), date(year, 12, 31), PeriodType.YEAR_TO_DATE
    return date(year, 1, 1), date(year, 12, 31), PeriodType.ANNUAL


def _month_end(year: int, month: int) -> date:
    return (date(year, month, 1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)


def _optional_date(value: Any) -> date | None:
    return date.fromisoformat(str(value)) if value not in (None, "") else None


def normalize_dividends(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Declared dividends per share in NT$ (``TaiwanStockDividend``).

    ``stock_dividend`` is the NT$ amount distributed as shares (NT$1 = 0.1 share per share at
    a NT$10 par). ``available_date`` is the board/shareholder announcement date; a row without
    one raises rather than guessing.
    """
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "fiscal_label",
                "period_type",
                "period_start",
                "period_end",
                "cash_dividend",
                "stock_dividend",
                "cash_ex_date",
                "stock_ex_date",
                "payment_date",
                "available_date",
                "available_date_estimated",
                "source",
            ]
        )
    needed = (
        "year",
        "CashEarningsDistribution",
        "CashStatutorySurplus",
        "StockEarningsDistribution",
        "StockStatutorySurplus",
        "CashExDividendTradingDate",
        "StockExDividendTradingDate",
        "CashDividendPaymentDate",
        "AnnouncementDate",
    )
    _require(raw, needed, ticker, "dividend")
    rows = []
    for rec in raw.to_dict("records"):
        start, end, ptype = parse_fiscal_period(rec["year"], ticker)
        announced = _optional_date(rec["AnnouncementDate"])
        if announced is None:
            raise MissingDataError(ticker, "dividend.AnnouncementDate", rec["year"], "a date")
        rows.append(
            {
                "ticker": ticker,
                "fiscal_label": rec["year"],
                "period_type": ptype.value,
                "period_start": start,
                "period_end": end,
                "cash_dividend": float(rec["CashEarningsDistribution"])
                + float(rec["CashStatutorySurplus"]),
                "stock_dividend": float(rec["StockEarningsDistribution"])
                + float(rec["StockStatutorySurplus"]),
                "cash_ex_date": _optional_date(rec["CashExDividendTradingDate"]),
                "stock_ex_date": _optional_date(rec["StockExDividendTradingDate"]),
                "payment_date": _optional_date(rec["CashDividendPaymentDate"]),
                "available_date": announced,
                "available_date_estimated": False,
                "source": SOURCE,
            }
        )
    return pd.DataFrame(rows).sort_values(["period_end", "available_date"]).reset_index(drop=True)


# --- fetcher --------------------------------------------------------------------------------


class FinMindFetcher:
    """Typed access to the FinMind datasets this tool uses.

    Statement and revenue fetches are also written to the cache's immutable snapshot table
    (one snapshot per ticker, dataset and period), so original versions accumulate over time
    even though FinMind itself serves only the latest one.
    """

    def __init__(
        self,
        client: FinMindClient,
        deadlines: FilingDeadlineConfig,
        snapshots: Cache | None = None,
    ) -> None:
        self._client = client
        self._deadlines = deadlines
        self._snapshots = snapshots

    def stock_info(self) -> pd.DataFrame:
        return normalize_stock_info(self._client.dataset("TaiwanStockInfo"))

    def trading_dates(self, start: date, end: date) -> list[date]:
        """Past index trading days plus FinMind's copy of the published future schedule."""
        raw = self._client.dataset("TaiwanStockTradingDate", start_date=start, end_date=end)
        return normalize_trading_dates(raw)

    def taiex_closes(self, start: date, end: date) -> pd.DataFrame:
        raw = self._client.dataset(
            "TaiwanStockPrice", data_id=TAIEX, start_date=start, end_date=end
        )
        return normalize_index_closes(raw, TAIEX)

    def total_return_index(self, start: date, end: date) -> pd.DataFrame:
        raw = self._client.dataset(
            "TaiwanStockTotalReturnIndex", data_id=TAIEX, start_date=start, end_date=end
        )
        return normalize_total_return_index(raw)

    def prices(self, code: str, start: date, end: date) -> pd.DataFrame:
        raw = self._client.dataset("TaiwanStockPrice", data_id=code, start_date=start, end_date=end)
        return normalize_prices(raw, code)

    def price_events(self, code: str, start: date, end: date) -> pd.DataFrame:
        """Ex-rights/dividend, capital-reduction and split events, oldest first."""
        kwargs: dict[str, Any] = {
            "data_id": code,
            "start_date": start,
            "end_date": end,
            "allow_empty": True,
        }
        frames = [
            normalize_dividend_results(
                self._client.dataset("TaiwanStockDividendResult", **kwargs), code
            ),
            normalize_capital_reductions(
                self._client.dataset("TaiwanStockCapitalReductionReferencePrice", **kwargs), code
            ),
            normalize_splits(self._client.dataset("TaiwanStockSplitPrice", **kwargs), code),
        ]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame(columns=EVENT_COLUMNS)
        return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    def statement(self, code: str, statement: Statement, start: date, end: date) -> pd.DataFrame:
        dataset = STATEMENT_DATASETS[statement]
        raw = self._client.dataset(dataset, data_id=code, start_date=start, end_date=end)
        self._snapshot(code, dataset, raw, "date")
        return normalize_statement(raw, code, statement, self._deadlines)

    def monthly_revenue(self, code: str, start: date, end: date) -> pd.DataFrame:
        dataset = "TaiwanStockMonthRevenue"
        raw = self._client.dataset(dataset, data_id=code, start_date=start, end_date=end)
        self._snapshot(code, dataset, raw, "date")
        return normalize_monthly_revenue(raw, code, self._deadlines)

    def valuation_dates(self, code: str, start: date, end: date) -> list[date]:
        """Days in TaiwanStockPER (exchange valuation files, which cover listed stocks only).

        A code can be reused after a delisting, so this may include an earlier company's days.
        """
        raw = self._client.dataset("TaiwanStockPER", data_id=code, start_date=start, end_date=end)
        _require(raw, ("date",), code, "per")
        return sorted(_dates(raw["date"], "per.date", code))

    def dividends(self, code: str, start: date, end: date) -> pd.DataFrame:
        raw = self._client.dataset(
            "TaiwanStockDividend", data_id=code, start_date=start, end_date=end, allow_empty=True
        )
        return normalize_dividends(raw, code)

    def _snapshot(self, code: str, dataset: str, raw: pd.DataFrame, period_col: str) -> None:
        if self._snapshots is None:
            return
        body = raw.drop(columns=["source"])
        for period, group in body.groupby(period_col, sort=True):
            records = group.sort_values(list(group.columns)).to_dict("records")
            payload = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str)
            self._snapshots.put_snapshot(SOURCE, dataset, f"{code}/{period}", payload)
