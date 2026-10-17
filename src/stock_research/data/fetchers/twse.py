"""TWSE OpenAPI fetcher (https://openapi.twse.com.tw/v1/), government open data.

* Listing dates: a fallback when the listing date cannot be derived from FinMind's valuation
  series (stocks that started trading between the emerging board's start and the valuation
  series' start).
* The latest trading day's close of every listed security (STOCK_DAY_ALL) and the ex-rights /
  ex-dividend preview (TWT48U_ALL), for the portfolio page. TWSE's terms allow reuse of data
  it provides through the open-data platform with the source stated.

The development environment cannot reach the TWSE (its firewall answers HTTP 200 with a block
page, which ``JsonClient`` rejects). Field names come from real responses read on a GitHub
runner.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from stock_research.data.errors import MissingDataError
from stock_research.data.fetchers.tpex import parse_gregorian_date, parse_number, parse_roc_date
from stock_research.data.http import JsonClient

SOURCE = "twse"
LISTED_COMPANIES = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
DAILY_CLOSES = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
EX_RIGHTS_PREVIEW = "https://openapi.twse.com.tw/v1/exchangeReport/TWT48U_ALL"


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

    def _rows(self, url: str, name: str, *, allow_empty: bool = False) -> list[dict]:
        rows = self._client.get_json(url, max_age=self._max_age).data
        if not isinstance(rows, list) or not (rows or allow_empty):
            raise MissingDataError(None, f"{SOURCE}:{name}", rows, "a non-empty list")
        return rows

    def daily_closes(self) -> pd.DataFrame:
        """date, code, name, close: the latest trading day of every listed security (stocks
        and ETFs). A security without trades has no close (NaN), never another day's."""
        rows = self._rows(DAILY_CLOSES, "STOCK_DAY_ALL")
        return pd.DataFrame(
            {
                "date": [parse_roc_date(str(r["Date"])) for r in rows],
                "code": [str(r["Code"]).strip() for r in rows],
                "name": [str(r["Name"]).strip() for r in rows],
                "close": [_number(r["ClosingPrice"], "close") for r in rows],
            }
        ).astype({"close": "float64"})

    def ex_rights_preview(self) -> pd.DataFrame:
        """date (ex-date), code, kind, cash_dividend (NT$/share, NaN until announced),
        stock_ratio_raw: upcoming ex-rights / ex-dividend dates of listed securities.

        The stock dividend ratio's unit has not been confirmed against a real event yet, so
        it is passed on as text and not turned into shares."""
        rows = self._rows(EX_RIGHTS_PREVIEW, "TWT48U_ALL", allow_empty=True)
        return pd.DataFrame(
            {
                "date": [parse_roc_date(str(r["Date"])) for r in rows],
                "code": [str(r["Code"]).strip() for r in rows],
                "kind": [str(r["Exdividend"]).strip() for r in rows],
                "cash_dividend": [_number(r["CashDividend"], "cash_dividend") for r in rows],
                "stock_ratio_raw": [str(r["StockDividendRatio"]).strip() for r in rows],
            }
        ).astype({"cash_dividend": "float64"})


def _number(value: object, field: str) -> float | None:
    """Empty or ``--`` (no trade, not announced) is None; other text must parse."""
    text = str(value).strip()
    if text in ("", "--", "---"):
        return None
    return parse_number(text, field)
