"""TWSE OpenAPI fetcher (https://openapi.twse.com.tw/v1/): listed companies' listing dates.

Used only as a fallback for the listing date when it cannot be derived from FinMind's
valuation series (stocks that started trading between the emerging board's start and the
valuation series' start). The development environment cannot reach the TWSE (its firewall
answers HTTP 200 with a block page, which ``JsonClient`` rejects), so there the fallback is
simply unavailable. Field names come from a real response read on a GitHub runner:
``公司代號`` and ``上市日期`` (YYYYMMDD).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from stock_research.data.errors import MissingDataError
from stock_research.data.fetchers.tpex import parse_gregorian_date
from stock_research.data.http import JsonClient

SOURCE = "twse"
LISTED_COMPANIES = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"


class TwseFetcher:
    def __init__(self, client: JsonClient, max_age: timedelta) -> None:
        self._client = client
        self._max_age = max_age

    def listing_dates(self) -> pd.DataFrame:
        """code, listing_date for every currently listed TWSE company."""
        result = self._client.get_json(LISTED_COMPANIES, max_age=self._max_age)
        rows = result.data
        if not isinstance(rows, list) or not rows:
            raise MissingDataError(None, f"{SOURCE}:t187ap03_L", rows, "a non-empty list")
        return pd.DataFrame(
            {
                "code": [str(r["公司代號"]).strip() for r in rows],
                "listing_date": [parse_gregorian_date(str(r["上市日期"])) for r in rows],
            }
        )
