"""TPEx (Taipei Exchange) OpenAPI fetcher: https://www.tpex.org.tw/openapi/

The OpenAPI serves only recent data (the current day or month), so it is used for listing
data, today's quotes, ex-rights reference prices and recent index closes. Field names and
date formats below come from real responses (``tests/fixtures/sources/tpex_*.json``).
Dates are either ROC (``"1151015"`` = 2026-10-15) or Gregorian (``"20261015"``), depending on
the endpoint.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

import pandas as pd

from stock_research.data.errors import AnomalyError, MissingDataError
from stock_research.data.http import JsonClient

SOURCE = "tpex"
BASE_URL = "https://www.tpex.org.tw/openapi/v1"
NO_TRADE = "---"
_PAR_RE = re.compile(r"([\d.]+)\s*元")


def parse_roc_date(value: str) -> date:
    """ROC calendar date ``"1151015"`` (year 115 = 2026) -> ``date(2026, 10, 15)``."""
    text = value.strip()
    if not re.fullmatch(r"\d{6,7}", text):
        raise AnomalyError(None, "roc_date", value, "YYYMMDD in the ROC calendar")
    return date(int(text[:-4]) + 1911, int(text[-4:-2]), int(text[-2:]))


def parse_gregorian_date(value: str) -> date:
    text = value.strip()
    if not re.fullmatch(r"\d{8}", text):
        raise AnomalyError(None, "date", value, "YYYYMMDD")
    return date(int(text[:4]), int(text[4:6]), int(text[6:]))


def parse_number(value: str, field: str) -> float | None:
    """Numeric field; ``---`` (no trade) becomes None. Anything else unparsable raises."""
    text = value.strip().replace(",", "")
    if text == NO_TRADE:
        return None
    try:
        return float(text)
    except ValueError as exc:
        raise AnomalyError(None, field, value, "a number or '---'") from exc


def parse_par_value(value: str) -> float:
    """``"新台幣                 10.0000元"`` -> 10.0. Not every company uses NT$10."""
    match = _PAR_RE.search(value)
    if match is None:
        raise AnomalyError(None, "par_value", value, "'新台幣 <amount>元'")
    return float(match.group(1))


class TpexFetcher:
    def __init__(self, client: JsonClient, max_age: timedelta) -> None:
        self._client = client
        self._max_age = max_age

    def _get(self, endpoint: str, *, allow_empty: bool = False) -> list[dict[str, Any]]:
        result = self._client.get_json(f"{BASE_URL}/{endpoint}", max_age=self._max_age)
        if not isinstance(result.data, list) or not (result.data or allow_empty):
            raise MissingDataError(None, f"{SOURCE}:{endpoint}", result.data, "a non-empty list")
        return result.data

    def listed_companies(self) -> pd.DataFrame:
        rows = self._get("mopsfin_t187ap03_O")
        return pd.DataFrame(
            {
                "code": [r["SecuritiesCompanyCode"].strip() for r in rows],
                "name": [r["CompanyAbbreviation"].strip() for r in rows],
                "full_name": [r["CompanyName"].strip() for r in rows],
                "industry_code": [r["SecuritiesIndustryCode"].strip() for r in rows],
                "listing_date": [parse_gregorian_date(r["DateOfListing"]) for r in rows],
                "par_value": [parse_par_value(r["ParValueOfCommonStock"]) for r in rows],
                "market": "TPEX",
                "data_date": [parse_roc_date(r["Date"]) for r in rows],
                "source": SOURCE,
            }
        )

    def daily_quotes(self) -> pd.DataFrame:
        """Latest trading day's raw quotes for every TPEx security.

        A security without trades has no open/high/low/close (``---``); those are NaN with
        ``no_trade`` set, never filled from another day.
        """
        rows = self._get("tpex_mainboard_daily_close_quotes")
        frame = pd.DataFrame(
            {
                "date": [parse_roc_date(r["Date"]) for r in rows],
                "code": [r["SecuritiesCompanyCode"].strip() for r in rows],
                "name": [str(r.get("CompanyName", "")).strip() for r in rows],
                "open": [parse_number(r["Open"], "open") for r in rows],
                "high": [parse_number(r["High"], "high") for r in rows],
                "low": [parse_number(r["Low"], "low") for r in rows],
                "close": [parse_number(r["Close"], "close") for r in rows],
                "volume": [parse_number(r["TradingShares"], "volume") for r in rows],
                "next_reference_price": [
                    parse_number(r["NextReferencePrice"], "next_reference_price") for r in rows
                ],
                "source": SOURCE,
            }
        )
        for col in ("open", "high", "low", "close", "volume", "next_reference_price"):
            frame[col] = frame[col].astype("float64")
        frame["no_trade"] = frame["close"].isna() & (frame["volume"] == 0)
        return frame

    def ex_rights_today(self) -> pd.DataFrame:
        """Ex-rights / ex-dividend reference prices for the current trading day (often none)."""
        rows = self._get("tpex_exright_daily", allow_empty=True)
        return pd.DataFrame(
            {
                "date": [parse_roc_date(r["Date"]) for r in rows],
                "code": [r["SecuritiesCompanyCode"].strip() for r in rows],
                "close_before": [
                    parse_number(r["ClosePriceBeforeExRightsDiviend"], "close_before") for r in rows
                ],
                "reference_price": [
                    parse_number(r["OpeningReferencePrice"], "reference_price") for r in rows
                ],
                "cash_dividend": [parse_number(r["CashDividend"], "cash_dividend") for r in rows],
                "stock_dividend": [
                    parse_number(r["StockDividend"], "stock_dividend") for r in rows
                ],
                # Shares received per 1,000 shares held.
                "stock_per_thousand": [
                    parse_number(r.get("StockDivdendThousandShares", "0"), "stock_per_thousand")
                    for r in rows
                ],
                "kind": [r["ExRightsDiviend"].strip() for r in rows],
                "source": SOURCE,
            }
        )

    def index_month(self) -> pd.DataFrame:
        """TPEx index OHLC for the current month (trading days with an official close)."""
        rows = self._get("tpex_index")
        return pd.DataFrame(
            {
                "date": [parse_gregorian_date(r["Date"]) for r in rows],
                "close": [parse_number(r["Close"], "close") for r in rows],
                "source": SOURCE,
            }
        )

    def total_return_index_month(self) -> pd.DataFrame:
        rows = self._get("tpex_reward_index")
        return pd.DataFrame(
            {
                "date": [parse_roc_date(r["Date"]) for r in rows],
                "price_index": [parse_number(r["TPExIndex"], "price_index") for r in rows],
                "total_return_index": [
                    parse_number(r["TPExTotalReturnIndex"], "total_return_index") for r in rows
                ],
                "source": SOURCE,
            }
        )
