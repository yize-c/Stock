"""FinMind API v4 client: https://api.finmindtrade.com/api/v4/data

Responses use the envelope ``{"msg": ..., "status": ..., "data": [...]}``. Status 402 means
the request quota is used up (see ``tests/fixtures/sources/finmind_402_limit.json``). The
token is sent as a Bearer header and is never part of the cache key or the stored payload.

Only the envelope is handled here. Per-dataset normalization (prices, statements, monthly
revenue, dividends) is added once real responses for each dataset have been captured as
fixtures; field names are not guessed.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from stock_research.data.errors import MissingDataError
from stock_research.data.http import JsonClient, SourceRateLimitedError, SourceUnavailableError

SOURCE = "finmind"
URL = "https://api.finmindtrade.com/api/v4/data"


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def check_envelope(body: Any) -> None:
    if not isinstance(body, dict) or "status" not in body:
        raise SourceUnavailableError(SOURCE, URL, str(body)[:200], "a FinMind response envelope")
    status = body.get("status")
    if status == 402:
        raise SourceRateLimitedError(
            SOURCE, URL, body.get("msg"), "request quota available (status 402 = limit reached)"
        )
    if status != 200:
        raise SourceUnavailableError(SOURCE, URL, f"{status}: {body.get('msg')}", "status 200")
    if not isinstance(body.get("data"), list):
        raise SourceUnavailableError(SOURCE, URL, type(body.get("data")).__name__, "a data list")


class FinMindClient:
    def __init__(self, client: JsonClient, max_age: timedelta) -> None:
        self._client = client
        self._max_age = max_age

    def dataset(
        self,
        dataset: str,
        *,
        data_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> pd.DataFrame:
        """Raw rows of one dataset as a DataFrame. An empty result raises."""
        params: dict[str, str] = {"dataset": dataset}
        if data_id is not None:
            params["data_id"] = data_id
        if start_date is not None:
            params["start_date"] = start_date.isoformat()
        if end_date is not None:
            params["end_date"] = end_date.isoformat()
        result = self._client.get_json(URL, params, max_age=self._max_age, check=check_envelope)
        rows = result.data["data"]
        if not rows:
            raise MissingDataError(data_id, f"{SOURCE}:{dataset}", "no rows", "at least one row")
        frame = pd.DataFrame(rows)
        frame["source"] = SOURCE
        return frame
