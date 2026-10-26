"""Why every stock failed the freshness check in the 2026-10-26 daily export (run 40240981656).

Prints dates only (no prices): the run's clock, the latest close the calendar expects, the last
index close days, FinMind's trading-day schedule around the weekend, and 2330's latest priced
day. Runs on GitHub Actions with FINMIND_TOKEN; a few FinMind requests, no cache.
"""

from __future__ import annotations

import tempfile
from datetime import date, timedelta
from pathlib import Path

from stock_research.config import Secrets, load_config
from stock_research.data.cache import Cache
from stock_research.data.market_data import MarketData


def main() -> None:
    config = load_config()
    with tempfile.TemporaryDirectory() as tmp, Cache(Path(tmp) / "cache.sqlite") as cache:
        market = MarketData.open(config, Secrets(None), cache)
        now = market.now
        today = market.today()
        print("now", now.isoformat(), "today", today)
        calendar = market.calendar()
        print("last historical day", calendar.last_historical_day)
        print("expected latest close", calendar.expected_latest_close(now))
        for offset in range(-6, 8):
            day = today + timedelta(days=offset)
            reason = calendar.closure_reason(day)
            print(" ", day, day.strftime("%a"), calendar.is_trading_day(day), reason or "")
        finmind = market._finmind
        closes = finmind.taiex_closes(today - timedelta(days=14), today)
        print("index close days", [d.isoformat() for d in list(closes["date"])[-6:]])
        schedule = finmind.trading_dates(today - timedelta(days=7), date(today.year, 11, 15))
        print("schedule", [d.isoformat() for d in schedule if d <= today + timedelta(days=10)])
        prices = finmind.prices("2330", today - timedelta(days=14), today)
        priced = prices.loc[prices["close"].notna(), "date"]
        print("2330 priced days", [d.isoformat() for d in list(priced)[-4:]])


if __name__ == "__main__":
    main()
