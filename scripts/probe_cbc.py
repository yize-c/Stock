"""Run the project's CBC rate reader against the real a13rate workbook (CI only).

The development environment cannot reach cbc.gov.tw; a GitHub runner can. Prints the parsed
five-bank one-year rates, checks them against the config's periods (``extend_with_cbc``
raises on any disagreement) and shows what an analysis today would use.
"""

from __future__ import annotations

import sys
from datetime import date

import requests

from stock_research.config import load_config
from stock_research.data.fetchers.cbc import CbcFetcher
from stock_research.data.risk_free import RiskFreeRates, extend_with_cbc


def main() -> int:
    config = load_config()
    risk_free = config.markets.tw.risk_free
    fetched = CbcFetcher(requests.Session(), risk_free.cbc).rates()
    print(f"=== parsed {len(fetched)} periods, {fetched[0].start} .. {fetched[-1].end}")
    for rate in fetched[:3] + fetched[-4:]:
        print(f"  {rate.start} {rate.end} {rate.rate:.6f}")
    merged = extend_with_cbc(risk_free.fallback_rates, fetched)
    later = len(merged) - len(risk_free.fallback_rates)
    print(f"=== agrees with the config; {later} later period(s)")
    for rate in merged[-3:]:
        print(f"  {rate.start} {rate.end} {rate.rate:.6f}")
    rates = RiskFreeRates(tuple(merged), "CBC a13rate", risk_free.carry_forward_max_days)
    today = date.today()
    print("=== 2020-01-02 .. today:", rates.average(date(2020, 1, 2), today))
    return 0


if __name__ == "__main__":
    sys.exit(main())
