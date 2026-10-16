"""One entry point to the data layer: fetch, adjust, apply point-in-time rules, validate.

``MarketData`` hands analysis code only what was knowable at ``as_of``:

* prices on or before ``as_of`` (raw and adjusted kept apart, see ``adjustment``);
* financial statements, monthly revenue and dividends whose ``available_date`` is on or
  before ``as_of``, newest published version per figure, IFRS periods only;
* the dividend-inclusive TAIEX total-return index as the benchmark.

``health_report`` runs every spec §7 check and records each result, so one failing check does
not hide the others. Analysis must not proceed when ``report.passed`` is false.

The trading calendar's history is the set of days with a TAIEX close; dates after the last
close come from FinMind's copy of the exchange's published schedule (TWSE itself is not
reachable from the development environment). Where both cover the same days they must agree.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import TypeVar
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from stock_research.config import AppConfig, KnownSuspension, ManualClosure, Secrets
from stock_research.data.adjustment import AdjustedPrices, adjust_prices
from stock_research.data.cache import Cache
from stock_research.data.errors import (
    AnomalyError,
    DataValidationError,
    MissingDataError,
    SourceMismatchError,
)
from stock_research.data.fetchers.finmind import (
    EventKind,
    FinMindClient,
    FinMindFetcher,
    Statement,
    auth_headers,
)
from stock_research.data.fetchers.tpex import TpexFetcher
from stock_research.data.fetchers.yahoo import YahooFetcher
from stock_research.data.http import JsonClient, RateLimiter
from stock_research.data.point_in_time import as_of_view
from stock_research.data.shares import ShareBasis, check_share_count, share_basis, share_counts
from stock_research.data.validation import (
    HealthReport,
    SourceStatus,
    check_balance_sheet,
    check_cross_source,
    check_freshness,
    check_monthly_revenue,
    check_price_completeness,
    check_price_limits,
)
from stock_research.markets.calendar import (
    CalendarConflictError,
    CalendarError,
    HolidaySchedule,
    TradingCalendar,
)
from stock_research.markets.tickers import Market, Ticker

SCHEDULE_SOURCE = "FinMind TaiwanStockTradingDate"
# A year's published schedule is treated as complete only if it reaches this day of December
# (Dec 28-31 always contains at least two weekdays).
_SCHEDULE_YEAR_END = (12, 28)

LIMITATIONS = (
    "證交所（TWSE）與公開資訊觀測站（MOPS）在開發環境無法連線：上市股票的收盤價只能比對 "
    "FinMind 與 Yahoo，沒有官方基準。",
    "歷史交易日取自 FinMind 的加權指數收盤日期；今天以後的休市日取自 FinMind 轉載的證交所"
    "行事曆，再加上 config 的手動休市日。",
    "財報只有最新版本（可能已重編），無法取得原始公告版本；公告日以法定期限估計。",
    "月營收公告日以 FinMind 建檔日與法定期限（次月 10 日）較早者估計。",
    "還原權息價格由程式依除權息、減資、分割的參考價自行計算（FinMind 還原股價需付費）。",
    "股價只使用 2003 年起的資料：FinMind 的除權息參考價與加權報酬指數都從 2003 年開始，"
    "更早的價格無法正確還原。上市前的興櫃交易資料不納入。",
    "Yahoo 的收盤價已做分割調整（配股也被當成分割），配股、減資、分割之前的日期不做跨來源比對。",
)


@dataclass(frozen=True, slots=True)
class StockInfo:
    ticker: Ticker
    name: str
    industries: tuple[str, ...]


def calendar_from_sources(
    index_close_dates: list[date],
    scheduled_dates: list[date],
    manual_closures: list[ManualClosure],
    *,
    timezone: str,
    daily_data_ready_time: time,
) -> TradingCalendar:
    """History = index close days; after them, the published schedule (whole years only)."""
    if not index_close_dates:
        raise CalendarConflictError("no index closes to build the trading calendar from")
    history = sorted(set(index_close_dates))
    last = history[-1]
    overlap = [d for d in scheduled_dates if history[0] <= d <= last]
    differ = sorted(set(overlap) ^ set(history))
    if differ:
        shown = ", ".join(d.isoformat() for d in differ[:5])
        raise CalendarConflictError(
            f"{SCHEDULE_SOURCE} disagrees with index close days on: {shown}"
        )
    future = sorted(d for d in scheduled_dates if d > last)
    years: set[int] = set()
    if future:
        final = future[-1]
        complete = final >= date(final.year, *_SCHEDULE_YEAR_END)
        complete_to = final.year if complete else final.year - 1
        years = set(range(last.year, complete_to + 1))
    future = [d for d in future if d.year in years]
    closures: dict[date, str] = {}
    cursor = last + timedelta(days=1)
    while years and cursor.year <= max(years):
        if cursor.weekday() < 5 and cursor not in future:
            closures[cursor] = f"not a trading day in {SCHEDULE_SOURCE}"
        cursor += timedelta(days=1)
    schedule = HolidaySchedule(
        years=frozenset(years),
        closures=closures,
        extra_trading_days=frozenset(d for d in future if d.weekday() >= 5),
    )
    return TradingCalendar(
        history,
        schedule,
        manual_closures,
        timezone=timezone,
        daily_data_ready_time=daily_data_ready_time,
    )


@dataclass
class _Memo:
    calendar: TradingCalendar | None = None
    info: pd.DataFrame | None = None
    raw: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)


class MarketData:
    def __init__(
        self,
        config: AppConfig,
        finmind: FinMindFetcher,
        yahoo: YahooFetcher,
        tpex: TpexFetcher | None = None,
        *,
        peer_finmind: FinMindFetcher | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._tw = config.markets.tw
        self._finmind = finmind
        # Peer statements use a longer cache (see markets.tw.industries.peers).
        self._peer_finmind = peer_finmind or finmind
        self._yahoo = yahoo
        self._tpex = tpex
        self._clock = clock
        self._memo = _Memo()

    @property
    def now(self) -> datetime:
        return self._clock()

    def today(self) -> date:
        """Today in the market's timezone."""
        return self.now.astimezone(ZoneInfo(self._config.general.timezone)).date()

    # --- reference data ---------------------------------------------------------------------

    def calendar(self) -> TradingCalendar:
        if self._memo.calendar is None:
            start = self._config.data_sources.finmind.history_start
            today = self.today()
            closes = self._finmind.taiex_closes(start, today)
            scheduled = self._finmind.trading_dates(start, date(today.year + 1, 12, 31))
            cal = self._tw.trading_calendar
            self._memo.calendar = calendar_from_sources(
                list(closes["date"]),
                scheduled,
                cal.manual_closures,
                timezone=self._config.general.timezone,
                daily_data_ready_time=cal.daily_data_ready_time,
            )
        return self._memo.calendar

    def stock_info(self, code: str) -> StockInfo:
        if self._memo.info is None:
            self._memo.info = self._finmind.stock_info()
        rows = self._memo.info.loc[self._memo.info["code"] == code]
        if rows.empty:
            raise MissingDataError(
                code, "finmind:TaiwanStockInfo", "not listed", "a TWSE or TPEx code"
            )
        row = rows.iloc[0]
        if row["market"] is None:
            raise AnomalyError(
                code, "finmind:TaiwanStockInfo.type", row["listed_types"], "exactly one market"
            )
        return StockInfo(Ticker(code, Market(row["market"])), row["name"], row["industries"])

    def resolve(self, code: str) -> Ticker:
        return self.stock_info(code).ticker

    # --- industries ---------------------------------------------------------------------------

    def _candidate_industries(self, categories: tuple[str, ...]) -> set[str]:
        """Exchange categories minus boards/broad parents, with TWSE/TPEx spellings merged."""
        industries = self._tw.industries
        named = {industries.aliases.get(c, c) for c in categories}
        specific = named - set(industries.ignored_categories)
        return specific or named

    def industry(self, code: str) -> str:
        """The stock's industry; a config override wins, an ambiguous one raises."""
        overrides = {o.ticker: o.industry for o in self._tw.industries.overrides}
        if code in overrides:
            return overrides[code]
        candidates = self._candidate_industries(self.stock_info(code).industries)
        if len(candidates) != 1:
            raise AnomalyError(
                code,
                "industry",
                sorted(candidates),
                "one industry (add markets.tw.industries.overrides)",
            )
        return candidates.pop()

    def industry_groups(self, industry: str) -> frozenset[str]:
        groups = self._tw.industries.groups
        return frozenset(name for name, members in groups.items() if industry in members)

    def peers(self, industry: str, exclude: str) -> list[Ticker]:
        """Every listed common share in ``industry`` (both markets), except ``exclude``."""
        self.stock_info(exclude)  # loads the listing
        assert self._memo.info is not None
        overrides = {o.ticker: o.industry for o in self._tw.industries.overrides}
        found = []
        for row in self._memo.info.itertuples(index=False):
            if row.code == exclude or row.market is None or not _COMMON_SHARE.match(row.code):
                continue
            industries = (
                {overrides[row.code]}
                if row.code in overrides
                else self._candidate_industries(row.industries)
            )
            if industry in industries:
                found.append(Ticker(row.code, Market(row.market)))
        return found

    def peer_financials(self, ticker: Ticker, as_of: date) -> pd.DataFrame:
        """Income and balance-sheet rows for a peer (long cache), as published by ``as_of``."""
        start = date(self._config.general.ifrs_start_year, 1, 1)
        frames = [
            self._peer_finmind.statement(ticker.code, s, start, self.today())
            for s in (Statement.INCOME, Statement.BALANCE)
        ]
        rows = pd.concat(frames, ignore_index=True)
        rows = rows.loc[pd.to_datetime(rows["period_start"]) >= pd.Timestamp(start)]
        return as_of_view(rows, as_of).reset_index(drop=True)

    # --- point-in-time data -----------------------------------------------------------------

    def listing_date(self, ticker: Ticker) -> date | None:
        """The day the stock started trading as a listed share, or None if it first traded
        before the emerging board existed (so the data holds no pre-listing prices).

        Order: a config override; else, for a stock that first traded on or after the
        emerging board's start, the first TaiwanStockPER day on or after its first trade
        (codes are reused, so earlier valuation days may belong to another company). That day
        is only meaningful after the valuation dataset's coverage starts; otherwise it raises.
        """
        overrides = {o.ticker: o.listing_date for o in self._tw.listing.overrides}
        if ticker.code in overrides:
            return overrides[ticker.code]
        first_traded = min(self._raw_prices(ticker)[0]["date"])
        if first_traded < self._tw.listing.emerging_board_start:
            return None
        sources = self._config.data_sources.finmind
        valued = self._finmind.valuation_dates(ticker.code, sources.history_start, self.today())
        first_valued = next((d for d in valued if d >= first_traded), None)
        if first_valued is None or first_valued <= sources.valuation_coverage_start:
            raise MissingDataError(
                ticker.code,
                "listing_date",
                f"first trade {first_traded}, first valuation day {first_valued}",
                f"a valuation day after {sources.valuation_coverage_start} "
                "or a listing date in markets.tw.listing.overrides",
            )
        return first_valued

    def prices(self, ticker: Ticker, as_of: date) -> AdjustedPrices:
        """Listed-period prices from ``price_history_start`` on, adjusted as of ``as_of``."""
        raw, events = self._raw_prices(ticker)
        listing = self.listing_date(ticker)
        start = self._config.data_sources.finmind.price_history_start
        if listing is not None:
            start = max(start, listing)
        raw = raw.loc[pd.to_datetime(raw["date"]) >= pd.Timestamp(start)]
        return adjust_prices(ticker.code, raw.reset_index(drop=True), events, as_of)

    def _raw_prices(self, ticker: Ticker) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Full FinMind history (including any emerging-board period) and price events."""
        if ticker.code not in self._memo.raw:
            start = self._config.data_sources.finmind.history_start
            today = self.today()
            raw = self._finmind.prices(ticker.code, start, today)
            events = self._finmind.price_events(ticker.code, start, today)
            self._memo.raw[ticker.code] = (raw, events)
        return self._memo.raw[ticker.code]

    def benchmark(self, as_of: date) -> pd.DataFrame:
        """TAIEX total-return index (dividends included) on or before ``as_of``."""
        tri = self._finmind.total_return_index(
            self._config.data_sources.finmind.history_start, self.today()
        )
        return tri.loc[pd.to_datetime(tri["date"]) <= pd.Timestamp(as_of)].reset_index(drop=True)

    def financials(self, ticker: Ticker, as_of: date) -> pd.DataFrame:
        """Income (quarterly), balance (instant) and cash-flow (year-to-date) rows, IFRS only."""
        start = date(self._config.general.ifrs_start_year, 1, 1)
        frames = [self._finmind.statement(ticker.code, s, start, self.today()) for s in Statement]
        rows = pd.concat(frames, ignore_index=True)
        rows = rows.loc[pd.to_datetime(rows["period_start"]) >= pd.Timestamp(start)]
        return as_of_view(rows, as_of).reset_index(drop=True)

    def monthly_revenue(self, ticker: Ticker, as_of: date) -> pd.DataFrame:
        # One year before the IFRS start so the first IFRS month has a year-ago value.
        start = date(self._config.general.ifrs_start_year - 1, 1, 1)
        rows = self._finmind.monthly_revenue(ticker.code, start, self.today())
        return as_of_view(rows, as_of).reset_index(drop=True)

    def dividends(self, ticker: Ticker, as_of: date) -> pd.DataFrame:
        """Declared dividends and rights offerings announced by ``as_of``.

        Fetched from the price-history start: share multipliers need every stock dividend
        behind the price events, not only the IFRS years.
        """
        start = self._config.data_sources.finmind.price_history_start
        rows = self._finmind.dividends(ticker.code, start, self.today())
        visible = pd.to_datetime(rows["available_date"]) <= pd.Timestamp(as_of)
        return rows.loc[visible].reset_index(drop=True)

    def share_basis(self, ticker: Ticker, as_of: date) -> ShareBasis:
        """Share multipliers from the price events applied as of ``as_of``.

        The basis starts at the IFRS start. Par at that date is TPEx's official par (TPEx
        stocks, backed out through every later split) or the configured default (TWSE).
        """
        per_share = self._tw.per_share
        start = date(self._config.general.ifrs_start_year, 1, 1)
        events = self.prices(ticker, as_of).events
        initial_par = per_share.default_par_value
        if ticker.market is Market.TPEX and self._tpex is not None:
            companies = self._tpex.listed_companies()
            row = companies.loc[companies["code"] == ticker.code]
            if row.empty:
                raise MissingDataError(ticker.code, "tpex:par_value", "absent", "a TPEx listing")
            all_events = self.prices(ticker, self.today()).events
            splits = all_events.loc[
                (all_events["kind"] == EventKind.SPLIT.value)
                & np.array([d >= start for d in all_events["effective_date"]], dtype=bool)
            ]
            ratio = float((splits["close_before"] / splits["reference_price"]).prod())
            initial_par = float(row["par_value"].iloc[0]) * ratio
        return share_basis(
            ticker.code,
            events,
            self.dividends(ticker, as_of),
            start=start,
            initial_par=initial_par,
            tolerance=per_share.share_multiplier_tolerance,
        )

    # --- validation -------------------------------------------------------------------------

    def health_report(self, code: str, as_of: date) -> HealthReport:
        report = HealthReport(ticker=code, as_of=as_of, limitations=list(LIMITATIONS))
        now = self.now

        calendar = _attempt(report, "calendar", self.calendar)
        ticker = _attempt(report, "listing", lambda: self.resolve(code))
        if calendar is None or ticker is None:
            return report
        if ticker.market is Market.TPEX and self._tpex is not None:
            _attempt(report, "listing.official_tpex", lambda: self._check_tpex_listing(ticker))
        prices = _attempt(report, "prices.fetch_and_adjust", lambda: self.prices(ticker, as_of))
        if prices is not None:
            self._price_checks(report, ticker, prices, calendar, as_of, now)
        bench = _attempt(report, "benchmark.fetch", lambda: self.benchmark(as_of))
        if bench is not None:
            report.add_source(
                SourceStatus("finmind:TaiwanStockTotalReturnIndex", now, max(bench["date"]))
            )
            _attempt(
                report,
                "benchmark.completeness",
                lambda: _check_index_days(bench, calendar, as_of),
            )

        tol = self._tw.data_validation
        fin = _attempt(report, "financials.fetch", lambda: self.financials(ticker, as_of))
        if fin is not None:
            _attempt(
                report,
                "financials.balance_sheet",
                lambda: check_balance_sheet(
                    code, _balance_totals(fin), tol.balance_sheet_tolerance
                ),
            )
            basis = _attempt(
                report, "per_share.share_basis", lambda: self.share_basis(ticker, as_of)
            )
            if basis is not None:
                _attempt(
                    report,
                    "per_share.share_count",
                    lambda: check_share_count(
                        code,
                        *share_counts(fin, basis),
                        self._tw.per_share.share_count_tolerance,
                    ),
                )
        revenue = _attempt(report, "revenue.fetch", lambda: self.monthly_revenue(ticker, as_of))
        if revenue is not None:
            report.add_source(
                SourceStatus("finmind:TaiwanStockMonthRevenue", now, max(revenue["period_end"]))
            )
            _attempt(
                report,
                "revenue.sanity",
                lambda: check_monthly_revenue(
                    code, revenue.rename(columns={"value": "revenue"}), tol.revenue_yoy_abs_max
                ),
            )
        return report

    def _check_tpex_listing(self, ticker: Ticker) -> None:
        """The derived listing date must match TPEx's official one."""
        assert self._tpex is not None
        companies = self._tpex.listed_companies()
        row = companies.loc[companies["code"] == ticker.code]
        if row.empty:
            raise MissingDataError(ticker.code, "tpex:listed_companies", "absent", "a TPEx listing")
        official = row["listing_date"].iloc[0]
        derived = self.listing_date(ticker)
        agrees = (
            official < self._tw.listing.emerging_board_start
            if derived is None
            else derived == official
        )
        if not agrees:
            raise SourceMismatchError(
                ticker.code,
                "listing_date",
                f"derived {derived or 'before the emerging board'}, TPEx {official}",
                "the official TPEx listing date (add a listing override if TPEx is right)",
            )

    def _price_checks(
        self,
        report: HealthReport,
        ticker: Ticker,
        prices: AdjustedPrices,
        calendar: TradingCalendar,
        as_of: date,
        now: datetime,
    ) -> None:
        code = ticker.code
        raw = prices.raw
        latest = max(raw["date"])
        report.add_source(SourceStatus("finmind:TaiwanStockPrice", now, latest))
        halts = [
            KnownSuspension(ticker=code, start=start, end=end, reason=f"{kind} share re-issue")
            for start, end, kind in prices.capital_change_halts()
        ]
        _attempt(
            report,
            "prices.completeness",
            lambda: check_price_completeness(
                code, raw, calendar, [*self._tw.known_suspensions, *halts]
            ),
        )
        traded = raw.loc[raw["close"].notna()]
        listing = self.listing_date(ticker)
        if listing is not None and listing < min(raw["date"]):
            listing = None  # listed before the price history starts: no exempt days in range
        _attempt(
            report,
            "prices.price_limits",
            lambda: check_price_limits(
                code,
                traded,
                self._tw.price_limits,
                listing_date=listing,
                ex_rights_reference=prices.ex_rights_reference(),
                capital_change_resumptions=prices.capital_change_resumptions(),
            ),
        )
        _attempt(
            report,
            "prices.freshness",
            lambda: check_freshness(code, latest, calendar, as_of, now),
        )
        _attempt(
            report,
            "prices.cross_source",
            lambda: self._cross_source(report, ticker, prices, as_of, now),
        )

    def _cross_source(
        self,
        report: HealthReport,
        ticker: Ticker,
        prices: AdjustedPrices,
        as_of: date,
        now: datetime,
    ) -> None:
        tol = self._tw.data_validation
        raw = prices.raw.loc[prices.raw["close"].notna()]
        recent = list(raw["date"])[-tol.cross_source_lookback_trading_days :]
        yahoo = self._yahoo.history(ticker, recent[0], recent[-1])
        report.add_source(SourceStatus("yahoo", now, max(yahoo["date"])))
        changes = prices.basis_change_dates() + [
            d for d, s in zip(yahoo["date"], yahoo["splits"], strict=True) if s not in (0, 1)
        ]
        cutoff = max([d for d in changes if d <= recent[-1]], default=None)
        excluded = [d for d in recent if cutoff is not None and d < cutoff]
        finmind = raw.set_index("date")["close"]
        check_cross_source(
            ticker.code,
            {"finmind": finmind, "yahoo": yahoo.set_index("date")["close"]},
            primary="finmind",
            tolerance=tol.cross_source_tolerance,
            lookback_trading_days=tol.cross_source_lookback_trading_days,
            excluded_dates=excluded,
        )
        if ticker.market is Market.TPEX and self._tpex is not None and as_of == self.today():
            quotes = self._tpex.daily_quotes()
            row = quotes.loc[(quotes["code"] == ticker.code) & quotes["close"].notna()]
            if not row.empty:
                report.add_source(SourceStatus("tpex", now, row["date"].iloc[0]))
                check_cross_source(
                    ticker.code,
                    {"tpex": row.set_index("date")["close"], "finmind": finmind},
                    primary="tpex",
                    tolerance=tol.cross_source_tolerance,
                    lookback_trading_days=1,
                )

    # --- construction -----------------------------------------------------------------------

    @classmethod
    def open(cls, config: AppConfig, secrets: Secrets, cache: Cache) -> MarketData:
        sources = config.data_sources
        session = requests.Session()
        max_age = timedelta(hours=sources.cache_max_age_hours)
        finmind_http = JsonClient(
            "finmind",
            session,
            RateLimiter(max_per_hour=sources.finmind.max_requests_per_hour),
            cache,
            timeout_seconds=sources.request_timeout_seconds,
            headers=auth_headers(secrets.get(Secrets.FINMIND_TOKEN)),
        )
        tpex_http = JsonClient(
            "tpex",
            session,
            RateLimiter(sources.official.min_request_interval_seconds),
            cache,
            timeout_seconds=sources.request_timeout_seconds,
        )
        peer_age = timedelta(days=config.markets.tw.industries.peers.cache_max_age_days)
        return cls(
            config,
            FinMindFetcher(
                FinMindClient(finmind_http, max_age),
                config.markets.tw.filing_deadlines,
                snapshots=cache,
            ),
            YahooFetcher(
                config.markets.tw,
                RateLimiter(sources.yahoo.min_request_interval_seconds),
                cache,
                max_age=max_age,
            ),
            TpexFetcher(tpex_http, max_age),
            peer_finmind=FinMindFetcher(
                FinMindClient(finmind_http, peer_age),
                config.markets.tw.filing_deadlines,
                snapshots=cache,
            ),
        )


T = TypeVar("T")
_COMMON_SHARE = re.compile(r"^[1-9]\d{3}$")


def _attempt(report: HealthReport, name: str, check: Callable[[], T]) -> T | None:
    """Run one check and record it; a failure is recorded and returns None."""
    try:
        return report.run(name, lambda: _wrap_calendar_errors(check))
    except DataValidationError:
        return None


def _wrap_calendar_errors(check: Callable[[], T]) -> T:
    try:
        return check()
    except CalendarError as exc:
        raise AnomalyError(None, "trading_calendar", str(exc), "a consistent calendar") from exc


def _check_index_days(index: pd.DataFrame, calendar: TradingCalendar, as_of: date) -> None:
    days = list(index["date"])
    start = max(min(days), calendar.first_historical_day)
    end = min(as_of, calendar.last_historical_day)
    missing = calendar.missing_trading_days(days, start, end)
    if missing:
        raise MissingDataError(
            "TAIEX", "total_return_index.date", [d.isoformat() for d in missing[:5]], "every day"
        )


_BALANCE_ITEMS = {
    "TotalAssets": "total_assets",
    "Liabilities": "total_liabilities",
    "Equity": "total_equity",
}


def _balance_totals(financials: pd.DataFrame) -> pd.DataFrame:
    rows = financials.loc[
        (financials["statement"] == Statement.BALANCE.value)
        & financials["item"].isin(list(_BALANCE_ITEMS))
    ]
    wide = rows.pivot(index="period_end", columns="item", values="value")
    missing = [k for k in _BALANCE_ITEMS if k not in wide]
    if missing or wide.isna().any().any():
        raise MissingDataError(
            None, "balance_sheet", missing or "gaps", f"{list(_BALANCE_ITEMS)} each period"
        )
    return wide.rename(columns=_BALANCE_ITEMS).reset_index()
