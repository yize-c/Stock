"""Official closes, ex-rights events, score history and parameters for the holdings page.

Everything published here is government open data (TWSE / TPEx OpenAPI, reused with the
source stated, user decision 2026-10-17) or the tool's own computed values; no FinMind record
and never any personal holdings (those stay in the viewer's browser).

The OpenAPIs only give the latest trading day, so the site keeps its own history: each export
reads the archive from the published site (``portfolio.site_url``), adds today's closes,
events and scores, and publishes it again. A missing archive starts a new one only when the
previous site never had one; any other failure stops the export (fail loudly), so history is
never silently lost or restarted.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from stock_research.config import AppConfig

ARCHIVE = "data/market/archive.json.gz"
ARCHIVE_SCHEMA = 1
SOURCE_NOTE = (
    "收盤價與除權息：臺灣證券交易所、證券櫃檯買賣中心 OpenAPI（政府資料開放平臺），"
    "為每日收盤價，不是即時報價。"
)

Fetch = Callable[[str], bytes | None]  # None when the file does not exist (HTTP 404)


class PreviousSiteError(Exception):
    """The previous site's archive could not be read; publishing would lose history."""


@dataclass
class Archive:
    closes: dict[str, dict[str, float]] = field(default_factory=dict)  # code -> date -> close
    names: dict[str, str] = field(default_factory=dict)
    events: dict[str, dict[str, Any]] = field(default_factory=dict)  # "code|date" -> event
    scores: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        body = {
            "schema": ARCHIVE_SCHEMA,
            "closes": self.closes,
            "names": self.names,
            "events": self.events,
            "scores": self.scores,
        }
        return gzip.compress(json.dumps(body, ensure_ascii=False, sort_keys=True).encode())

    @classmethod
    def from_bytes(cls, raw: bytes) -> Archive:
        body = json.loads(gzip.decompress(raw))
        if body.get("schema") != ARCHIVE_SCHEMA:
            raise PreviousSiteError(f"archive schema {body.get('schema')!r}, expected 1")
        return cls(body["closes"], body["names"], body["events"], body["scores"])


def load_previous(fetch: Fetch, site_url: str) -> Archive:
    """The published archive; a new one only if the previous site never published one."""
    archive = fetch(site_url + ARCHIVE)
    if archive is not None:
        return Archive.from_bytes(archive)
    index = fetch(site_url + "data/index.json")
    if index is not None and json.loads(index).get("archive"):
        raise PreviousSiteError(
            f"{site_url}data/index.json lists an archive but {ARCHIVE} is missing"
        )
    return Archive()


def http_fetch(timeout_seconds: float) -> Fetch:
    import requests

    def fetch(url: str) -> bytes | None:
        try:
            response = requests.get(url, timeout=timeout_seconds)
        except requests.RequestException as exc:
            raise PreviousSiteError(f"{url}: {type(exc).__name__}: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise PreviousSiteError(f"{url}: HTTP {response.status_code}")
        return response.content

    return fetch


def _finite(value: Any) -> float | None:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else value


def add_closes(archive: Archive, closes: pd.DataFrame) -> date:
    """Adds one day of closes (date, code, name, close); returns that day. Untraded
    securities get no entry for the day."""
    days = set(closes["date"])
    if len(days) != 1:
        raise ValueError(f"closes from more than one day: {sorted(days)}")
    day = days.pop()
    for code, name, close in zip(closes["code"], closes["name"], closes["close"], strict=True):
        if name:
            archive.names[code] = name
        value = _finite(float(close))
        if value is not None:
            archive.closes.setdefault(code, {})[day.isoformat()] = value
    return day


def add_events(archive: Archive, twse_preview: pd.DataFrame, tpex_today: pd.DataFrame) -> None:
    """Ex-rights / ex-dividend events. A later read of the same event (e.g. the cash amount
    announced after the date was) replaces the earlier one."""
    for row in twse_preview.itertuples():
        note = None
        if row.stock_ratio_raw:
            note = f"無償配股率 {row.stock_ratio_raw}（單位待確認，股數未自動調整）"
        archive.events[f"{row.code}|{row.date.isoformat()}"] = {
            "code": row.code,
            "date": row.date.isoformat(),
            "cash": _finite(row.cash_dividend),
            "stock_per_share": None if row.stock_ratio_raw else 0.0,
            "note": note,
        }
    for row in tpex_today.itertuples():
        archive.events[f"{row.code}|{row.date.isoformat()}"] = {
            "code": row.code,
            "date": row.date.isoformat(),
            "cash": _finite(row.cash_dividend),
            "stock_per_share": (_finite(row.stock_per_thousand) or 0.0) / 1000,
            "note": None,
        }


def add_scores(archive: Archive, as_of: date, rows: list[dict[str, Any]]) -> None:
    """The day's computed scores per tier and aspect verdicts of the watchlist (the holdings
    page compares them with the ones at purchase)."""
    for row in rows:
        if row.get("status") != "ok":
            entry: dict[str, Any] = {"status": row.get("status")}
        else:
            score = row.get("score") or {}
            entry = {
                "status": "ok",
                "tiers": {k: v["value"] for k, v in score.get("tiers", {}).items()},
                "aspects": {k: v["code"] for k, v in row["aspects"].items()},
            }
        archive.scores.setdefault(row["code"], {})[as_of.isoformat()] = entry


def params_payload(config: AppConfig) -> dict[str, Any]:
    """Every number the holdings page computes with; the page hard-codes none."""
    costs, score, portfolio = config.costs, config.research_score, config.portfolio
    return {
        "costs": {
            "fee_rate": costs.fee_rate,
            "default_fee_discount": costs.fee_discount,
            "min_fee": costs.min_fee,
            "tax_stock": costs.tax_stock,
            "tax_etf": costs.tax_etf,
            "board_lot": costs.board_lot,
        },
        "etf_code_prefixes": portfolio.etf_code_prefixes,
        "capital_tiers": [
            {"capital": t.capital, "holdings": t.holdings} for t in score.capital_tiers
        ],
        "labels": [{"min_score": label.min_score, "name": label.name} for label in score.labels],
        "alerts": portfolio.alerts.model_dump(),
        "allocation": portfolio.allocation.model_dump(),
        "backup_reminder_days": portfolio.backup_reminder_days,
        "compare_etf": config.markets.tw.benchmark.compare_etf,
        "note": "損益為稅前，未扣股利所得稅與二代健保補充保費。",
    }


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), "utf-8")


def write_market(out_dir: Path, archive: Archive, closes_day: date, config: AppConfig) -> None:
    """closes.json (latest day, every security), events.json, history/<code>.json and the
    archive itself, plus params.json."""
    day = closes_day.isoformat()
    base = out_dir / "data" / "market"
    _write(
        base / "closes.json",
        {
            "date": day,
            "source": SOURCE_NOTE,
            "closes": {
                code: [archive.names.get(code, ""), series[day]]
                for code, series in sorted(archive.closes.items())
                if day in series
            },
        },
    )
    _write(
        base / "events.json",
        {"source": SOURCE_NOTE, "events": sorted(archive.events.values(), key=_event_key)},
    )
    for code in sorted(set(archive.closes) | set(archive.scores)):
        _write(
            base / "history" / f"{code}.json",
            {
                "code": code,
                "name": archive.names.get(code, ""),
                "closes": sorted(archive.closes.get(code, {}).items()),
                "scores": sorted(archive.scores.get(code, {}).items()),
            },
        )
    (out_dir / ARCHIVE).parent.mkdir(parents=True, exist_ok=True)
    (out_dir / ARCHIVE).write_bytes(archive.to_bytes())
    _write(out_dir / "data" / "params.json", params_payload(config))


def _event_key(event: dict[str, Any]) -> tuple[str, str]:
    return event["date"], event["code"]
