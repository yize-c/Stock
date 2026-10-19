"""Follow-up to probe_calibration_details.py (CI only; reads the calibration cache).

Three open questions from the dated causes of run 39239243546:
* share_count "no comparable quarters" (73 stocks): which income-statement items FinMind has
  for them (item names only), so the parent net income can be found;
* price-limit moves with nothing near them (751, mostly July-August 2007-2009): whether
  TaiwanStockDividend declares an ex-date that day while TaiwanStockDividendResult has no
  event, and whether the move is within the limit against the reference price implied by the
  declaration;
* traded days without any price (9,647 days in 443 stocks): by market and year, how long the
  runs are, and how the volume compares with the stock's median traded day.
Then the two sections the first run could not finish: NO_SCORE listing years and the
point-in-time scope. Only names, dates, counts and derived ratios are printed.
"""

from __future__ import annotations

import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from stock_research.calibration import pipeline
from stock_research.config import load_config
from stock_research.data.errors import DataValidationError
from stock_research.markets.price_limits import limit_on

WORK = Path("calibration/work")


def section(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def main() -> None:
    config = load_config(Path("config.yaml"))
    finmind = config.data_sources.finmind
    rows = pd.concat(
        [pd.read_parquet(p) for p in sorted(WORK.glob("rows_*.parquet"))], ignore_index=True
    )
    md = pipeline.calibration_market(config, wait_on_quota=False)
    found = pipeline.universe(md, config)
    client = md._finmind._client  # noqa: SLF001 - a probe reads internals
    failed = rows.loc[rows["status"] == "DATA_CHECK_FAILED"].drop_duplicates("code")

    def last_day(code: str):
        day = config.calibration.data_as_of
        if code in found.delisted:
            day = min(day, found.delisted[code] - timedelta(days=1))
        return day

    section("share_count: income items of stocks without comparable quarters")
    codes = [c for c, r in zip(failed["code"], failed["reason"], strict=True)
             if "per_share.share_count" in r]  # fmt: skip
    items: Counter = Counter()
    lacking = 0
    for code in codes:
        raw = client.dataset(
            "TaiwanStockFinancialStatements", data_id=code, start_date=finmind.price_history_start,
            end_date=md.today(),
        )  # fmt: skip
        names = set(raw["type"])
        items.update(names)
        if "EquityAttributableToOwnersOfParent" not in names:
            lacking += 1
    print(f"{len(codes)} stocks; {lacking} without EquityAttributableToOwnersOfParent")
    for name, count in items.most_common(60):
        print(f"  {count:4d}  {name}")

    section("isolated price-limit moves vs declared ex-dates")
    matched = unmatched = within = 0
    by_year: Counter = Counter()
    first_result: list[int] = []
    for code, reason in zip(failed["code"], failed["reason"], strict=True):
        if "prices.price_limits" not in reason:
            continue
        try:
            ticker = md.resolve(code)
            prices = md.prices(ticker, last_day(code))
        except DataValidationError:
            continue
        raw = prices.raw.loc[prices.raw["close"].notna()].reset_index(drop=True)
        events = md._raw_prices(ticker)[1]  # noqa: SLF001
        results = events.loc[events["kind"] == "ex_rights_dividend", "date"]
        if len(results):
            first_result.append(pd.Timestamp(min(results)).year)
        declared = client.dataset(
            "TaiwanStockDividend", data_id=code, start_date=finmind.price_history_start,
            end_date=md.today(), allow_empty=True,
        )  # fmt: skip
        ex_days: dict = {}
        for rec in declared.to_dict("records"):
            cash = float(rec["CashEarningsDistribution"]) + float(rec["CashStatutorySurplus"])
            stock = float(rec["StockEarningsDistribution"]) + float(rec["StockStatutorySurplus"])
            for key in ("CashExDividendTradingDate", "StockExDividendTradingDate"):
                day = pd.to_datetime(rec.get(key), errors="coerce")
                if pd.notna(day):
                    c, s = ex_days.get(day.date(), (0.0, 0.0))
                    ex_days[day.date()] = (
                        c + (cash if key.startswith("Cash") else 0.0),
                        s + (stock if key.startswith("Stock") else 0.0),
                    )
        refs = prices.ex_rights_reference()
        closes = raw["close"].astype(float).to_numpy()
        days = list(raw["date"])
        for i in range(1, len(days)):
            change = closes[i] / closes[i - 1] - 1
            limit = limit_on(days[i], config.markets.tw.price_limits)
            if (
                days[i] in refs
                or abs(change) <= limit + config.markets.tw.price_limits.rounding_tolerance
            ):
                continue
            if days[i] in ex_days:
                matched += 1
                by_year[days[i].year] += 1
                cash, stock = ex_days[days[i]]
                implied = (closes[i - 1] - cash) / (1 + stock / 10)
                tolerance = config.markets.tw.price_limits.rounding_tolerance
                if abs(closes[i] / implied - 1) <= limit + tolerance:
                    within += 1
            else:
                unmatched += 1
    print(
        f"moves beyond the limit off ex-rights-result days: {matched + unmatched}; on a "
        f"declared ex-date: {matched} (within the limit vs the implied reference: {within})"
    )
    print("by year:", dict(sorted(by_year.items())))
    print("first DividendResult event year per stock:",
          dict(sorted(Counter(first_result).items())))  # fmt: skip

    section("traded days without any price")
    per_market: Counter = Counter()
    runs: Counter = Counter()
    ratios = []
    for code, reason in zip(failed["code"], failed["reason"], strict=True):
        if "prices.completeness" not in reason:
            continue
        try:
            ticker = md.resolve(code)
            raw = md.raw_daily(ticker)
        except DataValidationError:
            continue
        volume = raw["volume"].astype(float)
        empty = (volume > 0) & raw["close"].isna()
        if not empty.any():
            continue
        per_market[ticker.market.name] += int(empty.sum())
        run = 0
        for flag in empty:
            if flag:
                run += 1
            elif run:
                runs[min(run, 10)] += 1
                run = 0
        normal = volume[(volume > 0) & raw["close"].notna()]
        if len(normal):
            ratios.extend((volume[empty] / normal.median()).tolist())
    print("days by market:", dict(per_market))
    print("run lengths (10 = 10 or more):", dict(sorted(runs.items())))
    if ratios:
        q = np.quantile(ratios, [0.1, 0.5, 0.9])
        print(f"volume / the stock's median traded-day volume: p10 {q[0]:.3f}, "
              f"median {q[1]:.3f}, p90 {q[2]:.3f}")  # fmt: skip

    section("NO_SCORE: listing years")
    listed = []
    for code in rows.loc[rows["status"] == "NO_SCORE", "code"].unique():
        try:
            when = md.listing_date(md.resolve(code))
            listed.append(str(when.year) if when else "before 2002")
        except DataValidationError:
            listed.append("unknown")
    print(pd.Series(listed).value_counts().sort_index().to_string(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
