"""Why calibration rows were excluded (CI only; reads the calibration cache and row artifacts).

The full-market calibration run 39162884540 kept 46,415 liquid stock-months, of which
18,942 were DATA_CHECK_FAILED, 816 ERROR and 926 NO_SCORE. This probe reads that run's row
artifacts (scores and statuses, derived values only) and the calibration cache, and prints:
* status counts by base year;
* DATA_CHECK_FAILED: rows and stocks per failed check, and each failing stock's check details
  re-run from the cache (normalized, with example codes);
* ERROR and NO_SCORE: rows and stocks per normalized reason, by year;
* whether the excluded stocks differ from the scored ones: market, delisted, years since
  listing, industry, liquidity rank and the next 12 months' gross return where computable.
Only counts, check names, codes and derived statistics are printed; no raw FinMind values.
"""

from __future__ import annotations

import re
import sys
import time
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.calibration import pipeline
from stock_research.config import load_config
from stock_research.data.errors import DataValidationError

WORK = Path("calibration/work")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def normalize(text: str, width: int = 140) -> str:
    return NUMBER.sub("#", " ".join(str(text).split()))[:width]


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def table(frame: pd.DataFrame, rows: int = 40) -> None:
    with pd.option_context("display.width", 200, "display.max_colwidth", 140):
        print(frame.head(rows).to_string(), flush=True)


def checks_of(reason: str) -> list[str]:
    return [c.strip() for c in str(reason).removeprefix("data checks failed: ").split(",")]


def main() -> None:
    config = load_config(Path("config.yaml"))
    rows = pd.concat(
        [pd.read_parquet(p) for p in sorted(WORK.glob("rows_*.parquet"))], ignore_index=True
    )
    rows["year"] = pd.to_datetime(rows["base_date"]).dt.year
    print(f"{len(rows)} rows, {rows['code'].nunique()} stocks", flush=True)

    section("status by base year (rows)")
    table(pd.crosstab(rows["year"], rows["status"], margins=True))
    section("status by stock (a stock's rows all share one data-check result)")
    per_stock = rows.groupby("code")["status"].agg(
        lambda s: "DATA_CHECK_FAILED" if (s == "DATA_CHECK_FAILED").all() else "scored"
    )
    print(per_stock.value_counts().to_string(), flush=True)

    failed = rows.loc[rows["status"] == "DATA_CHECK_FAILED"]
    section("DATA_CHECK_FAILED: rows and stocks per failed check")
    exploded = failed.assign(check=failed["reason"].map(checks_of)).explode("check")
    by_check = exploded.groupby("check").agg(rows=("code", "size"), stocks=("code", "nunique"))
    table(by_check.sort_values("rows", ascending=False))
    section("DATA_CHECK_FAILED: combinations of failed checks (stocks)")
    combos = failed.drop_duplicates("code")["reason"].map(
        lambda r: " + ".join(sorted(checks_of(r)))
    )
    print(combos.value_counts().head(25).to_string(), flush=True)

    market_data = pipeline.calibration_market(config, wait_on_quota=False)
    found = pipeline.universe(market_data, config)
    delisted = found.delisted

    section("DATA_CHECK_FAILED: check details re-run from the cache")
    details: dict[str, Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    started = time.monotonic()
    failing_codes = sorted(failed["code"].unique())
    for n, code in enumerate(failing_codes):
        last_day = config.calibration.data_as_of
        if code in delisted:
            last_day = min(last_day, delisted[code] - timedelta(days=1))
        try:
            report = market_data.health_report(code, last_day, cross_source=False)
        except Exception as exc:  # noqa: BLE001 - a probe reports every failure
            details["(health_report raised)"][normalize(f"{type(exc).__name__}: {exc}")] += 1
            continue
        for check in report.checks:
            if check.passed:
                continue
            key = normalize(check.detail)
            details[check.name][key] += 1
            if len(examples[(check.name, key)]) < 6:
                examples[(check.name, key)].append(code)
        if n % 100 == 0:
            print(
                f"  re-checked {n}/{len(failing_codes)} ({time.monotonic() - started:.0f}s)",
                flush=True,
            )
    for name, counter in sorted(details.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"\n[{name}] {sum(counter.values())} stocks", flush=True)
        for key, count in counter.most_common(15):
            print(f"  {count:4d}  {key}  e.g. {' '.join(examples[(name, key)])}", flush=True)

    for status in ("ERROR", "NO_SCORE"):
        subset = rows.loc[rows["status"] == status].copy()
        section(f"{status}: rows and stocks per normalized reason")
        subset["why"] = subset["reason"].map(lambda r: normalize(r, 120))
        grouped = subset.groupby("why").agg(
            rows=("code", "size"),
            stocks=("code", "nunique"),
            first=("base_date", "min"),
            last=("base_date", "max"),
            codes=("code", lambda c: " ".join(sorted(set(c))[:8])),
        )
        table(grouped.sort_values("rows", ascending=False), 30)
        section(f"{status}: rows by base year")
        print(subset["year"].value_counts().sort_index().to_string(), flush=True)

    section("are the excluded stocks different? (one row per stock)")
    liquidity = pd.read_parquet(WORK / pipeline.LIQUIDITY_FILE)
    kept = liquidity.loc[liquidity["kept"]].copy()
    kept["rank"] = kept.groupby("base_date")["liquidity"].rank(ascending=False, pct=True)
    median_rank = kept.groupby("code")["rank"].median()
    attributes = []
    for code in sorted(rows["code"].unique()):
        item = {"code": code, "group": per_stock[code]}
        try:
            ticker = market_data.resolve(code)
            listing = market_data.listing_date(ticker)
            item["market"] = ticker.market.name
            item["listing_year"] = listing.year if listing else np.nan
        except DataValidationError:
            item["market"] = "unknown"
        try:
            item["industry"] = market_data.industry(code)
        except DataValidationError:
            item["industry"] = "unknown"
        item["delisted"] = code in delisted
        item["liquidity_rank"] = median_rank.get(code, np.nan)
        attributes.append(item)
    stocks = pd.DataFrame(attributes)
    table(pd.crosstab(stocks["market"], stocks["group"], margins=True))
    table(pd.crosstab(stocks["delisted"], stocks["group"], margins=True))
    stocks["listed_since"] = pd.cut(stocks["listing_year"], [1900, 2000, 2010, 2015, 2020, 2030])
    table(pd.crosstab(stocks["listed_since"], stocks["group"], margins=True))
    print(stocks.groupby("group")["liquidity_rank"].describe().to_string(), flush=True)
    share = pd.crosstab(stocks["industry"], stocks["group"])
    share["failed_share"] = share.get("DATA_CHECK_FAILED", 0) / share.sum(axis=1)
    table(share.sort_values("failed_share", ascending=False), 45)

    section("next 12 months' gross return of excluded vs scored rows (where computable)")
    returns = []
    sample = failed.groupby("code").head(60)
    for code, group in sample.groupby("code"):
        last_day = config.calibration.data_as_of
        if code in delisted:
            last_day = min(last_day, delisted[code] - timedelta(days=1))
        try:
            prices = market_data.prices(market_data.resolve(code), last_day)
        except Exception:  # noqa: BLE001 - prices themselves may be what failed
            continue
        adjusted = prices.adjusted.set_index("date")["adj_close"].dropna().astype(float)
        for base in group["base_date"]:
            gross, _ = pipeline.forward_gross(adjusted, base, 12, delisted.get(code))
            if gross is not None:
                returns.append({"year": pd.Timestamp(base).year, "gross": gross})
    excluded = pd.DataFrame(returns)
    scored = rows.loc[rows["status"] == "OK", ["year", "gross"]].dropna()
    summary = pd.DataFrame(
        {
            "scored_mean": scored.groupby("year")["gross"].mean(),
            "scored_n": scored.groupby("year")["gross"].size(),
            "excluded_mean": excluded.groupby("year")["gross"].mean() if len(excluded) else np.nan,
            "excluded_n": excluded.groupby("year")["gross"].size() if len(excluded) else 0,
        }
    )
    table(summary)


if __name__ == "__main__":
    sys.exit(main())
