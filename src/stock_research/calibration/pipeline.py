"""Calibration stages, run on GitHub Actions with the second FinMind token.

1. ``fetch`` (resumable, within a time budget): for every common stock (listed now, or
   delisted after the first base date), the raw daily prices and the exchange valuation
   series (2 requests each); then the liquidity screen; then, for the stocks kept in any
   month, everything the score needs. All of it lands in the calibration cache, read with
   a fixed clock (``data_as_of``) so runs on different days hit the same entries.
2. ``liquidity``: each month-end base date, the 60-trading-day average traded value of every
   stock listed then (after its listing; FinMind's emerging-board days are not counted); the
   most traded ``liquidity_keep_share`` are kept.
3. ``score`` (one shard of the kept stocks): the data checks once per stock (no Yahoo
   comparison), then for each kept base date the point-in-time base score and the next
   ``horizon_months`` gross return on adjusted prices. A stock delisted inside the horizon
   is valued at its last close (what a holder last could sell at).
4. ``summary`` (see ``summary.py``).

Every row the stages write is a derived value (scores, returns, liquidity), never a raw
FinMind record.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stock_research.analysis.costs import net_return, position_shares
from stock_research.analysis.inputs import load_stock_data
from stock_research.analysis.research_score import score_stock
from stock_research.analysis.stability_inputs import stability_report
from stock_research.config import AppConfig, CalibrationConfig, Secrets, add_months
from stock_research.data.cache import Cache
from stock_research.data.errors import DataValidationError
from stock_research.data.market_data import MarketData
from stock_research.validation.metric_walk_forward import positive_metrics

LIQUIDITY_FILE = "liquidity.parquet"
FETCH_STATUS_FILE = "fetch_status.json"
PART_KEYS = ("principles", "consistency", "valuation", "risk", "chips")
# A listed stock whose newest TaiwanStockInfo row is older than this is no longer listed.
_STALE_INFO_DAYS = 30
# A close used for a base date or a horizon end must be this recent.
_MAX_STALE_DAYS = 10


def base_dates(settings: CalibrationConfig) -> list[date]:
    return [d.date() for d in pd.date_range(settings.start, settings.last_base_date, freq="ME")]


def calibration_market(config: AppConfig, *, wait_on_quota: bool = True) -> MarketData:
    """Market data on the calibration cache, with a long cache life and a fixed clock."""
    settings = config.calibration
    sources = config.data_sources.model_copy(
        update={
            "cache_dir": settings.cache_dir,
            "cache_max_age_hours": settings.cache_max_age_days * 24,
        }
    )
    fixed = config.model_copy(update={"data_sources": sources})
    tz = ZoneInfo(config.general.timezone)
    moment = datetime.combine(settings.data_as_of, time(22, 0), tz).astimezone(UTC)
    cache = Cache(settings.cache_dir / "cache.sqlite")
    return MarketData.open(
        fixed, Secrets(), cache, wait_on_quota=wait_on_quota, clock=lambda: moment
    )


@dataclass(frozen=True)
class Universe:
    codes: list[str]
    delisted: dict[str, date]  # first day no longer listed


def universe(market_data: MarketData, config: AppConfig) -> Universe:
    """Common stocks listed at some base date: listed now, or delisted after the first one.

    A code whose newest TaiwanStockInfo row is current is listed now, even if an earlier
    company with that code was delisted (codes are reused).
    """
    settings = config.calibration
    info = market_data.common_stocks(settings.excluded_code_prefixes)
    delistings = market_data.delisting_dates()
    codes, delisted = [], {}
    for code, data_date in zip(info["code"], info["data_date"], strict=True):
        current = (settings.data_as_of - data_date).days <= _STALE_INFO_DAYS
        if current:
            codes.append(code)
        elif code in delistings and delistings[code] > settings.start:
            codes.append(code)
            delisted[code] = delistings[code]
    return Universe(codes, delisted)


# --- stage 1: fetch ----------------------------------------------------------------------------


def fetch(
    market_data: MarketData,
    config: AppConfig,
    deadline: datetime,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Warm the cache until ``deadline``; the returned status says whether all is fetched.

    A stock whose data cannot be read is recorded (it will be excluded with that reason),
    not retried forever.
    """
    found = universe(market_data, config)
    failed: dict[str, str] = {}

    def out_of_time() -> bool:
        return datetime.now(UTC) >= deadline

    for i, code in enumerate(found.codes):
        if out_of_time():
            return {"complete": False, "stage": "prices", "done": i, "of": len(found.codes)}
        try:
            ticker = market_data.resolve(code)
            market_data.raw_daily(ticker)
            market_data.listing_date(ticker)
        except DataValidationError as exc:
            failed[code] = f"{type(exc).__name__}: {exc}"
        if i % 50 == 0:
            progress(f"prices {i}/{len(found.codes)}")
    kept = liquidity(market_data, config, found)
    kept_codes = sorted(set(kept.loc[kept["kept"], "code"]))
    for i, code in enumerate(kept_codes):
        if out_of_time():
            return {"complete": False, "stage": "full", "done": i, "of": len(kept_codes)}
        try:
            ticker = market_data.resolve(code)
            load_stock_data(market_data, code, config.calibration.data_as_of, with_peers=False)
            market_data.exchange_valuation(ticker, config.calibration.data_as_of)
            market_data.institutional(ticker, config.calibration.data_as_of)
            market_data.margin(ticker, config.calibration.data_as_of)
        except DataValidationError as exc:
            failed[code] = f"{type(exc).__name__}: {exc}"
        if i % 25 == 0:
            progress(f"full {i}/{len(kept_codes)}")
    return {"complete": True, "codes": len(found.codes), "kept": len(kept_codes), "failed": failed}


# --- stage 2: liquidity ------------------------------------------------------------------------


def liquidity(market_data: MarketData, config: AppConfig, found: Universe) -> pd.DataFrame:
    """code, base_date, liquidity (NT$ per day), kept: the screen of every base date."""
    settings = config.calibration
    dates = base_dates(settings)
    start = config.data_sources.finmind.price_history_start
    rows = []
    for code in found.codes:
        try:
            ticker = market_data.resolve(code)
            raw = market_data.raw_daily(ticker)
            listing = market_data.listing_date(ticker)
        except DataValidationError:
            continue
        first = max(start, listing) if listing is not None else start
        listed = np.array([d >= first for d in raw["date"]], dtype=bool)
        traded = raw.loc[raw["close"].notna().to_numpy() & listed]
        days = list(traded["date"])
        turnover = traded["turnover"].to_numpy(dtype=float)
        delisted = found.delisted.get(code)
        for base in dates:
            if delisted is not None and delisted <= base:
                continue
            upto = sum(1 for d in days if d <= base)
            window = turnover[max(0, upto - settings.liquidity_lookback_trading_days) : upto]
            if (
                len(window) < settings.liquidity_lookback_trading_days
                or (base - days[upto - 1]).days > _MAX_STALE_DAYS
            ):
                continue
            rows.append({"code": code, "base_date": base, "liquidity": float(window.mean())})
    frame = pd.DataFrame(rows, columns=["code", "base_date", "liquidity"])
    rank = frame.groupby("base_date")["liquidity"].rank(ascending=False, pct=True)
    frame["kept"] = rank <= settings.liquidity_keep_share
    return frame


# --- stage 3: score ----------------------------------------------------------------------------


def _close_on_or_before(series: pd.Series, day: date) -> tuple[date, float] | None:
    eligible = series.loc[[d <= day for d in series.index]]
    if eligible.empty:
        return None
    return eligible.index[-1], float(eligible.iloc[-1])


def forward_gross(
    adjusted: pd.Series, base: date, months: int, delisted: date | None
) -> tuple[float | None, bool]:
    """Gross return on adjusted closes from ``base`` to ``base + months`` and whether the
    stock was delisted inside the horizon (valued at its last close then)."""
    end = add_months(base, months)
    start = _close_on_or_before(adjusted, base)
    if start is None or (base - start[0]).days > _MAX_STALE_DAYS:
        return None, False
    inside_delisting = delisted is not None and delisted <= end
    stop = _close_on_or_before(adjusted, end)
    if stop is None or stop[0] <= start[0]:
        return None, inside_delisting
    if not inside_delisting and (end - stop[0]).days > _MAX_STALE_DAYS:
        return None, False  # no close near the horizon end (a long halt): no outcome
    return stop[1] / start[1] - 1.0, inside_delisting


def tier_net(
    gross: float | None, price: float | None, capital: int, holdings: int, etf: bool, config
) -> float:
    """``gross`` net of one round trip for a position of ``capital / holdings`` at ``price``
    (board lots, or odd lots when one lot does not fit); NaN when not computable."""
    if gross is None or price is None or np.isnan(gross) or np.isnan(price):
        return np.nan
    position = position_shares(price, capital / holdings, config.costs)
    if position.shares <= 0:
        return np.nan
    return net_return(gross, position.shares * price, etf, config.costs)


def score_shard(
    market_data: MarketData,
    config: AppConfig,
    kept: pd.DataFrame,
    delisted: dict[str, date],
    shard: int,
    shards: int,
    progress: Callable[[str], None] = lambda _: None,
) -> pd.DataFrame:
    """Rows for this shard's stocks: status, base score and parts, forward gross return."""
    settings = config.calibration
    tiers = config.research_score.capital_tiers
    positive = positive_metrics(config.research_score.consistency.walk_forward_results)
    codes = sorted(set(kept["code"]))[shard::shards]
    rows: list[dict[str, Any]] = []
    for n, code in enumerate(codes):
        dates = sorted(kept.loc[kept["code"] == code, "base_date"])
        last_day = settings.data_as_of
        if code in delisted:
            last_day = min(last_day, delisted[code] - timedelta(days=1))
        problem = _health_problem(market_data, code, last_day)
        if problem is not None:
            rows += [
                {"code": code, "base_date": d, "status": "DATA_CHECK_FAILED", "reason": problem}
                for d in dates
            ]
            continue
        ticker = market_data.resolve(code)
        later = market_data.prices(ticker, last_day)
        adjusted = later.adjusted.set_index("date")["adj_close"].dropna().astype(float)
        raw = later.raw.set_index("date")["close"].dropna().astype(float)
        progress(f"{code} ({n + 1}/{len(codes)}): {len(dates)} base dates")
        for base in dates:
            row: dict[str, Any] = {"code": code, "base_date": base}
            try:
                data = load_stock_data(market_data, code, base, with_peers=False)
                report = stability_report(data, config, with_breakpoints=False)
                score = score_stock(market_data, data, report, config, positive)
            except (DataValidationError, ValueError, KeyError) as exc:
                rows.append(row | {"status": "ERROR", "reason": f"{type(exc).__name__}: {exc}"})
                continue
            gross, delisted_inside = forward_gross(
                adjusted, base, settings.horizon_months, delisted.get(code)
            )
            close = _close_on_or_before(raw, base)
            price = close[1] if close else None  # used for position sizes, never written
            row |= {
                "status": "OK" if score.score is not None else "NO_SCORE",
                "reason": score.problem,
                "score": score.score,
                **{f"part_{p.key}": p.points for p in score.parts},
                "gross": gross,
                "delisted_in_horizon": delisted_inside,
                **{
                    f"net_{t.capital}": tier_net(gross, price, t.capital, t.holdings, False, config)
                    for t in tiers
                },
            }
            rows.append(row)
    columns = [
        "code", "base_date", "status", "reason", "score",
        *[f"part_{k}" for k in PART_KEYS], "gross", "delisted_in_horizon",
        *[f"net_{t.capital}" for t in tiers],
    ]  # fmt: skip
    return pd.DataFrame(rows).reindex(columns=columns)


def _health_problem(market_data: MarketData, code: str, as_of: date) -> str | None:
    report = market_data.health_report(code, as_of, cross_source=False)
    if report.passed:
        return None
    return "data checks failed: " + ", ".join(c.name for c in report.checks if not c.passed)
