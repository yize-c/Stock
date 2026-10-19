"""Dated causes of the calibration exclusions (CI only; reads the calibration cache).

``probe_calibration_rows.py`` counted the failed checks of run 39162884540 but its normalized
details hid the dates. This probe re-derives, from the cache, when and why each check fails:
* prices.completeness: traded days without an open/high/low/close, and unexplained gaps, by
  year;
* prices.price_limits: every move beyond the limit, by year and by what is near it (a capital
  change resumption, a listing, a halt, an unapplied event);
* listing.official_tpex: derived vs official listing dates and whether both precede the price
  history start (where the listing date changes nothing);
* per_share.share_basis / share_count, financials.balance_sheet, prices.freshness: the labels,
  dates and counts behind them;
* ERROR rows: the missing revenue months; NO_SCORE rows: the listing years;
* how many rows a point-in-time scope (a failure only excludes base dates whose 12-month
  forward window reaches it) would keep.
Only dates, codes, labels, counts and derived ratios are printed; no raw FinMind values.
"""

from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.calibration import pipeline
from stock_research.config import KnownSuspension, load_config
from stock_research.data.errors import DataValidationError
from stock_research.data.fetchers.finmind import EventKind, parse_fiscal_period
from stock_research.data.shares import share_counts
from stock_research.markets.price_limits import (
    PriceLimitStatus,
    PriceMove,
    ReferencePriceKind,
    check_price_move,
)

WORK = Path("calibration/work")
PRICE_COLUMNS = ("open", "high", "low", "close")


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def checks_of(reason: str) -> list[str]:
    return [c.strip() for c in str(reason).removeprefix("data checks failed: ").split(",")]


def years(days: list[date]) -> str:
    counts = Counter(d.year for d in days)
    return " ".join(f"{y}:{n}" for y, n in sorted(counts.items()))


def bucket(change: float) -> str:
    size = abs(change)
    for edge in (0.15, 0.3, 0.6):
        if size <= edge:
            return f"<= {edge:.0%}"
    return "> 60%"


def main() -> None:
    config = load_config(Path("config.yaml"))
    tw = config.markets.tw
    finmind = config.data_sources.finmind
    rows = pd.concat(
        [pd.read_parquet(p) for p in sorted(WORK.glob("rows_*.parquet"))], ignore_index=True
    )
    rows["base_date"] = pd.to_datetime(rows["base_date"]).dt.date
    md = pipeline.calibration_market(config, wait_on_quota=False)
    found = pipeline.universe(md, config)
    calendar = md.calendar()
    failed = rows.loc[rows["status"] == "DATA_CHECK_FAILED"]
    checks = failed.drop_duplicates("code").set_index("code")["reason"].map(checks_of)

    def last_day(code: str) -> date:
        day = config.calibration.data_as_of
        if code in found.delisted:
            day = min(day, found.delisted[code] - timedelta(days=1))
        return day

    first_failure: dict[str, date] = {}  # earliest dated cause per stock

    def mark(code: str, day: date | None) -> None:
        if day is not None:
            first_failure[code] = min(first_failure.get(code, day), day)

    completeness: dict[str, Counter] = defaultdict(Counter)
    nan_days: dict[str, list[date]] = defaultdict(list)
    gap_days: list[date] = []
    limit_rows: list[dict] = []
    listing_rows: list[dict] = []
    reasons: Counter = Counter()
    labels: Counter = Counter()
    undeclared: list[str] = []
    share_count_rows: list[str] = []
    freshness_rows: list[str] = []
    balance_rows: list[str] = []
    other: Counter = Counter()
    started = time.monotonic()

    for n, (code, names) in enumerate(checks.items()):
        if n % 100 == 0:
            print(f"  {n}/{len(checks)} ({time.monotonic() - started:.0f}s)", flush=True)
        day = last_day(code)
        try:
            ticker = md.resolve(code)
        except DataValidationError as exc:
            other[f"resolve: {type(exc).__name__}"] += 1
            continue
        try:
            prices = md.prices(ticker, day)
        except DataValidationError as exc:
            other[f"prices: {type(exc).__name__} {getattr(exc, 'field', '')}"] += 1
            prices = None

        if prices is not None and "prices.completeness" in names:
            raw = prices.raw
            traded = pd.to_numeric(raw["volume"], errors="coerce") > 0
            missing = {c: raw.loc[traded & raw[c].isna(), "date"].tolist() for c in PRICE_COLUMNS}
            kind = "+".join(c for c in PRICE_COLUMNS if missing[c]) or "none"
            completeness["traded days missing"][kind] += 1
            for column, days in missing.items():
                nan_days[column].extend(days)
            if missing["open"]:
                mark(code, min(missing["open"]))
            halts = [
                KnownSuspension(ticker=code, start=s, end=e, reason=k)
                for s, e, k in prices.capital_change_halts()
            ]
            suspensions = [s for s in [*tw.known_suspensions, *halts] if s.ticker == code]
            dates = list(raw["date"])
            gaps = [
                d
                for d in calendar.missing_trading_days(dates, min(dates), max(dates))
                if not any(s.covers(d) for s in suspensions)
            ]
            if gaps:
                gap_days.extend(gaps)
                completeness["unexplained gaps"][f"{len(gaps)} day(s) {years(gaps)}"] += 1
                mark(code, min(gaps))

        if prices is not None and "prices.price_limits" in names:
            raw = prices.raw.loc[prices.raw["close"].notna()]
            days = list(raw["date"])
            closes = [float(c) for c in raw["close"]]
            refs = prices.ex_rights_reference()
            resumptions = sorted(prices.capital_change_resumptions())
            listing = md.listing_date(ticker)
            in_range = bool(days) and listing is not None and days[0] == listing
            _, all_events = md._raw_prices(ticker)  # noqa: SLF001 - a probe reads internals
            event_days = [pd.Timestamp(d).date() for d in all_events["date"]]
            applied = set(prices.events["date"]) if not prices.events.empty else set()
            unapplied = [d for d in event_days if d not in applied]
            position = {d: i for i, d in enumerate(days)}
            for i in range(1, len(days)):
                d = days[i]
                result = check_price_move(
                    PriceMove(
                        day=d,
                        reference_price=refs[d] if d in refs else closes[i - 1],
                        close=closes[i],
                        reference_kind=(
                            ReferencePriceKind.EX_RIGHTS_REFERENCE
                            if d in refs
                            else ReferencePriceKind.PREVIOUS_CLOSE
                        ),
                        trading_day_since_listing=i + 1 if in_range else None,
                        resumed_after_capital_change=d in set(resumptions),
                    ),
                    tw.price_limits,
                )
                if result.status is not PriceLimitStatus.EXCEEDED:
                    continue
                since_change = [i - position[r] + 1 for r in resumptions if r in position]
                since_change = [k for k in since_change if 1 <= k <= 10]
                halt_days = len(calendar.trading_days(days[i - 1], d)) - 2
                if since_change:
                    near = f"capital change resumption day {min(since_change)}"
                elif in_range and i + 1 <= 10:
                    near = f"listing trading day {i + 1}"
                elif any(abs((d - e).days) <= 7 for e in unapplied):
                    near = "unapplied event within 7 days"
                elif d in refs:
                    near = "ex-rights day (vs reference)"
                elif halt_days >= 5:
                    near = "after >= 5 trading days without a close"
                elif halt_days >= 1:
                    near = "after 1-4 trading days without a close"
                else:
                    near = "isolated"
                limit_rows.append(
                    {
                        "code": code,
                        "date": d,
                        "year": d.year,
                        "size": bucket(result.change),
                        "sign": "+" if result.change > 0 else "-",
                        "near": near,
                        "change": result.change,
                    }
                )
            mine = [r for r in limit_rows if r["code"] == code]
            if mine:
                mark(code, min(r["date"] for r in mine))

        if "listing.official_tpex" in names and md._tpex is None:  # noqa: SLF001
            other["listing: no TPEx fetcher"] += 1
        elif "listing.official_tpex" in names:
            try:
                companies = md._tpex.listed_companies()  # noqa: SLF001
                row = companies.loc[companies["code"] == code]
                official = row["listing_date"].iloc[0] if not row.empty else None
                first_traded = min(md.raw_daily(ticker)["date"])
                try:
                    derived = md.listing_date(ticker)
                    derived_text = str(derived)
                except DataValidationError as exc:
                    derived, derived_text = None, f"raises {type(exc).__name__}"
                listing_rows.append(
                    {
                        "code": code,
                        "official": official,
                        "derived": derived_text,
                        "first_traded": first_traded,
                        "delisted": code in found.delisted,
                        "diff_days": (derived - official).days if derived and official else None,
                        "both_before_prices": (
                            official is not None
                            and official < finmind.price_history_start
                            and (derived is None or derived < finmind.price_history_start)
                        ),
                    }
                )
                if official is not None and derived is not None:
                    mark(code, min(official, derived))
            except DataValidationError as exc:
                other[f"listing: {type(exc).__name__}"] += 1

        if "per_share.share_basis" in names and prices is not None:
            _, all_events = md._raw_prices(ticker)  # noqa: SLF001
            cuts = all_events.loc[all_events["kind"] == EventKind.CAPITAL_REDUCTION.value]
            for detail, when in zip(cuts["detail"], cuts["date"], strict=True):
                reasons[str(detail)] += 1
                if str(detail) not in ("Making up losses", "Cash refund"):
                    mark(code, pd.Timestamp(when).date())
            raw_div = md._finmind._client.dataset(  # noqa: SLF001
                "TaiwanStockDividend",
                data_id=code,
                start_date=finmind.price_history_start,
                end_date=md.today(),
                allow_empty=True,
            )
            for label in raw_div.get("year", pd.Series(dtype=str)):
                try:
                    parse_fiscal_period(label, code)
                except DataValidationError:
                    labels[repr(label)] += 1
            try:
                md.share_basis(ticker, day)
            except DataValidationError as exc:
                if exc.field == "dividend.stock_ex_date":
                    scheduled = str(exc.actual).removeprefix("no declaration for ex-rights day ")
                    ex = prices.events.loc[
                        prices.events["kind"] == EventKind.EX_RIGHTS_DIVIDEND.value
                    ]
                    labels_on = ex.loc[ex["date"].astype(str) == scheduled, "detail"]
                    undeclared.append(f"{code} {scheduled} label={list(labels_on)}")
                    mark(code, date.fromisoformat(scheduled))
                elif exc.field == "tpex:par_value":
                    other[f"tpex par absent (delisted={code in found.delisted})"] += 1

        if "per_share.share_count" in names:
            try:
                fin = md.financials(ticker, day)
                basis = md.share_basis(ticker, day)
                implied, capital = share_counts(fin, basis)
                both = pd.concat([implied, capital], axis=1, keys=["i", "c"]).dropna()
                positive = both.loc[(both["i"] > 0) & (both["c"] > 0)]
                ratio = (positive["i"] / positive["c"]).median() if len(positive) else np.nan
                share_count_rows.append(
                    f"{code} {md.industry(code)} implied={implied.notna().sum()} "
                    f"capital={capital.notna().sum()} both>0={len(positive)} "
                    f"ratio={ratio:.3f} par={basis.initial_par}"
                )
            except DataValidationError as exc:
                share_count_rows.append(f"{code} raises {type(exc).__name__} {exc.field}")

        if "prices.freshness" in names and prices is not None:
            freshness_rows.append(
                f"{code} last close {max(prices.raw['date'])} checked at {day} "
                f"delisted={found.delisted.get(code)}"
            )
        if "financials.balance_sheet" in names:
            fin = md.financials(ticker, day)
            balance = fin.loc[fin["statement"] == "balance"]
            ends = sorted(set(pd.to_datetime(balance["period_end"]).dt.date))
            need = ("TotalAssets", "Liabilities", "Equity")
            gaps = [
                str(e)
                for e in ends
                if not set(need) <= set(balance.loc[balance["period_end"] == e, "item"])
            ]
            balance_rows.append(f"{code} {len(gaps)} period(s): {' '.join(gaps[:8])}")

    section("prices.completeness")
    for name, counter in completeness.items():
        print(f"[{name}] {sum(counter.values())} stocks", flush=True)
        for key, count in counter.most_common(12):
            print(f"  {count:4d}  {key}", flush=True)
    for column in PRICE_COLUMNS:
        print(f"traded days without {column}: {len(nan_days[column])}  {years(nan_days[column])}")
    print(f"unexplained gap days: {len(gap_days)}  {years(gap_days)}", flush=True)

    section("prices.price_limits: every move beyond the limit")
    limits = pd.DataFrame(limit_rows)
    if not limits.empty:
        print(f"{len(limits)} moves, {limits['code'].nunique()} stocks", flush=True)
        print(pd.crosstab(limits["year"], limits["near"], margins=True).to_string(), flush=True)
        print(pd.crosstab(limits["size"], limits["sign"], margins=True).to_string(), flush=True)
        isolated = limits.loc[limits["near"] == "isolated"].sort_values("date")
        print(f"isolated moves ({len(isolated)}), first 60:", flush=True)
        for r in isolated.head(60).itertuples():
            print(f"  {r.code} {r.date} {r.change:+.1%}", flush=True)
        per_stock = limits.groupby("code").size()
        print("moves per stock:", per_stock.value_counts().sort_index().to_dict(), flush=True)

    section("listing.official_tpex: derived vs official")
    listing = pd.DataFrame(listing_rows)
    if not listing.empty:
        print(f"{len(listing)} stocks; both dates before price_history_start: "
              f"{int(listing['both_before_prices'].sum())}", flush=True)  # fmt: skip
        diff = listing["diff_days"].dropna()
        print("derived - official (days):", flush=True)
        print(pd.cut(diff, [-1e5, -366, -31, -1, 0, 30, 365, 1e5]).value_counts().sort_index()
              .to_string(), flush=True)  # fmt: skip
        listing["official_year"] = [d.year if d else None for d in listing["official"]]
        print(listing.groupby("official_year").size().to_string(), flush=True)
        print(listing["derived"].str.startswith("raises").value_counts().to_string(), flush=True)
        for r in listing.head(40).itertuples():
            print(f"  {r.code} official={r.official} derived={r.derived} "
                  f"first_traded={r.first_traded} delisted={r.delisted}", flush=True)  # fmt: skip

    section("per_share.share_basis")
    print("capital reduction reasons seen:", dict(reasons), flush=True)
    print("unparsed dividend labels:", flush=True)
    for label, count in labels.most_common(40):
        print(f"  {count:4d}  {label}", flush=True)
    print(f"ex-rights days without a declaration ({len(undeclared)}):", flush=True)
    for line in undeclared:
        print(f"  {line}", flush=True)

    section("per_share.share_count")
    for line in share_count_rows:
        print(f"  {line}", flush=True)
    section("prices.freshness")
    for line in freshness_rows:
        print(f"  {line}", flush=True)
    section("financials.balance_sheet")
    for line in balance_rows:
        print(f"  {line}", flush=True)
    section("other")
    print(dict(other), flush=True)

    section("ERROR: missing revenue months")
    errors = rows.loc[rows["status"] == "ERROR"]
    for code in sorted(errors["code"].unique()):
        try:
            revenue = md.monthly_revenue(md.resolve(code), last_day(code))
            months = pd.to_datetime(revenue["period_end"]).dt.to_period("M")
            expected = pd.period_range(months.min(), months.max(), freq="M")
            missing = sorted(set(expected) - set(months))
            print(f"  {code} missing {len(missing)}: {' '.join(str(m) for m in missing[:12])}")
        except DataValidationError as exc:
            print(f"  {code} raises {type(exc).__name__} {exc.field}")

    section("NO_SCORE: listing years")
    no_score = rows.loc[rows["status"] == "NO_SCORE", "code"].unique()
    listed = []
    for code in no_score:
        try:
            when = md.listing_date(md.resolve(code))
            listed.append(when.year if when else "before 2002")
        except DataValidationError:
            listed.append("unknown")
    print(pd.Series(listed).value_counts().sort_index().to_string(), flush=True)

    section("point-in-time scope: rows a dated failure would not reach")
    dated = failed.loc[failed["code"].isin(first_failure)]
    window_end = [b + timedelta(days=366) for b in dated["base_date"]]
    before = [w < first_failure[c] for w, c in zip(window_end, dated["code"], strict=True)]
    print(f"{len(dated)} failed rows with a dated cause ({dated['code'].nunique()} stocks); "
          f"{sum(before)} have a 12-month forward window ending before it", flush=True)  # fmt: skip
    print("earliest failure year per stock:", flush=True)
    print(pd.Series([d.year for d in first_failure.values()]).value_counts().sort_index()
          .to_string(), flush=True)  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
