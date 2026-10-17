"""Official closes, events, score history and parameters for the holdings page (no network)."""

from __future__ import annotations

import gzip
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from stock_research.config import load_config
from stock_research.data.errors import SourceMismatchError
from stock_research.data.fetchers.tpex import TpexFetcher
from stock_research.data.fetchers.twse import TwseFetcher
from stock_research.data.http import JsonClient, RateLimiter
from stock_research.data.market_data import MarketData
from stock_research.export import market
from stock_research.export.site import export_site
from tests.conftest import DEFAULT_CONFIG
from tests.fakes import FakeResponse, FakeSession, fixture_text
from tests.test_market_data import TSMC_LISTING, build_market_data

CONFIG = load_config(DEFAULT_CONFIG)
SITE = CONFIG.portfolio.site_url


def _twse(*bodies: str) -> TwseFetcher:
    session = FakeSession([FakeResponse(200, b) for b in bodies])
    return TwseFetcher(JsonClient("twse", session, RateLimiter()), timedelta(hours=1))


def _tpex(*bodies: str) -> TpexFetcher:
    session = FakeSession([FakeResponse(200, b) for b in bodies])
    return TpexFetcher(JsonClient("tpex", session, RateLimiter()), timedelta(hours=1))


def _closes(day: date, rows: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": [day] * len(rows), "code": list(rows), "name": list(rows), "close": rows.values()}
    )


NO_EVENTS = (
    pd.DataFrame(columns=["date", "code", "kind", "cash_dividend", "stock_ratio_raw"]),
    pd.DataFrame(columns=["date", "code", "cash_dividend", "stock_per_thousand"]),
)


# --- previous archive -----------------------------------------------------------------------


def _site(files: dict[str, bytes]):
    return lambda url: files.get(url.removeprefix(SITE))


def test_previous_archive_is_read_or_started_only_when_none_was_published() -> None:
    archive = market.Archive(closes={"2330": {"2026-10-15": 1000.0}})
    loaded = market.load_previous(_site({market.ARCHIVE: archive.to_bytes()}), SITE)
    assert loaded.closes == archive.closes
    assert market.load_previous(_site({}), SITE).closes == {}  # first deploy
    old_site = {"data/index.json": json.dumps({"stocks": []}).encode()}
    assert market.load_previous(_site(old_site), SITE).closes == {}  # site before archives
    listed = {"data/index.json": json.dumps({"archive": market.ARCHIVE}).encode()}
    with pytest.raises(market.PreviousSiteError, match="missing"):
        market.load_previous(_site(listed), SITE)  # it existed: never restart silently
    wrong = gzip.compress(json.dumps({"schema": 99}).encode())
    with pytest.raises(market.PreviousSiteError, match="schema"):
        market.load_previous(_site({market.ARCHIVE: wrong}), SITE)


def test_http_fetch_maps_404_and_fails_on_other_errors(monkeypatch) -> None:
    import requests

    answers = {"a": (200, b"x"), "b": (404, b""), "c": (500, b"")}

    def fake_get(url, timeout):
        status, content = answers[url]
        return type("R", (), {"status_code": status, "content": content})()

    monkeypatch.setattr(requests, "get", fake_get)
    fetch = market.http_fetch(5)
    assert fetch("a") == b"x" and fetch("b") is None
    with pytest.raises(market.PreviousSiteError, match="500"):
        fetch("c")


# --- adding a day ---------------------------------------------------------------------------


def test_closes_events_and_scores_accumulate() -> None:
    archive = market.Archive()
    day = market.add_closes(
        archive, _closes(date(2026, 10, 15), {"2330": 1000.0, "9999": float("nan")})
    )
    market.add_closes(archive, _closes(date(2026, 10, 16), {"2330": 1010.0}))
    assert day == date(2026, 10, 15)
    assert archive.closes == {"2330": {"2026-10-15": 1000.0, "2026-10-16": 1010.0}}  # no NaN
    with pytest.raises(ValueError, match="more than one day"):
        market.add_closes(
            archive,
            pd.concat(
                [_closes(date(2026, 10, 15), {"1": 1.0}), _closes(date(2026, 10, 16), {"2": 1.0})]
            ),
        )
    preview = _twse(fixture_text("twse_twt48u_all.json")).ex_rights_preview()
    tpex = _tpex(fixture_text("tpex_exright_daily.json")).ex_rights_today()
    market.add_events(archive, preview, tpex)
    unannounced = archive.events["00406A|2026-11-03"]
    assert unannounced["cash"] is None and unannounced["stock_per_share"] == 0.0
    announced = preview.assign(cash_dividend=0.2)  # a later read fills the amount in
    market.add_events(archive, announced, tpex.iloc[0:0])
    assert archive.events["00406A|2026-11-03"]["cash"] == 0.2
    tpex_event = next(e for e in archive.events.values() if e["code"] == "6548")
    assert tpex_event["cash"] == pytest.approx(0.451055) and tpex_event["stock_per_share"] == 0.0
    with_stock = preview.assign(stock_ratio_raw=["0.05", ""])
    market.add_events(archive, with_stock, tpex.iloc[0:0])
    flagged = archive.events["00401A|2026-10-15"]
    assert flagged["stock_per_share"] is None and "單位待確認" in flagged["note"]
    aspects = {"revenue": "STABLE", "profitability": "UNSTABLE", "risk": "STABLE"}
    rows = [
        {
            "code": "2330",
            "status": "ok",
            "score": {"available": True, "tiers": {"300000": {"value": 70.0, "label": "中"}}},
            "aspects": {k: {"code": v} for k, v in aspects.items()},
        },
        {"code": "9917", "status": "blocked"},
    ]
    market.add_scores(archive, date(2026, 10, 16), rows)
    assert archive.scores["2330"]["2026-10-16"]["tiers"] == {"300000": 70.0}
    assert archive.scores["2330"]["2026-10-16"]["aspects"]["profitability"] == "UNSTABLE"
    assert archive.scores["9917"]["2026-10-16"] == {"status": "blocked"}


def test_tpex_stock_dividends_become_shares_per_share() -> None:
    archive = market.Archive()
    tpex = _tpex(fixture_text("tpex_exright_daily.json")).ex_rights_today()
    market.add_events(archive, NO_EVENTS[0], tpex.assign(stock_per_thousand=50.0))
    assert all(e["stock_per_share"] == pytest.approx(0.05) for e in archive.events.values())


def test_written_files_and_parameters(tmp_path) -> None:
    archive = market.Archive()
    day = market.add_closes(archive, _closes(date(2026, 10, 16), {"2330": 1010.0, "0050": 109.0}))
    market.write_market(tmp_path, archive, day, CONFIG)
    closes = json.loads((tmp_path / "data" / "market" / "closes.json").read_text("utf-8"))
    assert closes["date"] == "2026-10-16" and closes["closes"]["0050"] == ["0050", 109.0]
    assert "不是即時報價" in closes["source"]
    history = json.loads(
        (tmp_path / "data" / "market" / "history" / "2330.json").read_text("utf-8")
    )
    assert history["closes"] == [["2026-10-16", 1010.0]]
    assert (
        market.Archive.from_bytes((tmp_path / market.ARCHIVE).read_bytes()).closes == archive.closes
    )
    params = json.loads((tmp_path / "data" / "params.json").read_text("utf-8"))
    assert params["costs"]["fee_rate"] == CONFIG.costs.fee_rate
    assert params["costs"]["tax_etf"] == CONFIG.costs.tax_etf
    assert params["alerts"]["score_drop"] == CONFIG.portfolio.alerts.score_drop
    assert params["allocation"]["max_weight"] == CONFIG.portfolio.allocation.max_weight
    assert [t["holdings"] for t in params["capital_tiers"]] == [3, 5, 10, 15]
    assert "稅前" in params["note"]


# --- official sources -----------------------------------------------------------------------


def test_official_closes_need_both_exchanges_on_the_same_day() -> None:
    twse_day = fixture_text("twse_stock_day_all.json")
    tpex_day = fixture_text("tpex_daily_close_quotes.json")  # 2026-10-15 vs TWSE 2026-10-16
    md = MarketData(CONFIG, None, None, _tpex(tpex_day), twse=_twse(twse_day))  # type: ignore[arg-type]
    with pytest.raises(SourceMismatchError):
        md.official_closes()
    same = twse_day.replace("1151016", "1151015")
    md = MarketData(CONFIG, None, None, _tpex(tpex_day), twse=_twse(same))  # type: ignore[arg-type]
    closes = md.official_closes()
    assert {"2330", "0050", "6488"} <= set(closes["code"])
    assert set(closes["date"]) == {date(2026, 10, 15)}


# --- the export -----------------------------------------------------------------------------


def test_export_extends_the_published_history(raw_config, write_config, tmp_path) -> None:
    raw_config["markets"]["tw"]["listing"]["overrides"] = [TSMC_LISTING]
    raw_config["app"]["watchlist"] = ["2330"]
    config = load_config(write_config(raw_config))
    md, _ = build_market_data(config)
    published: dict[str, bytes] = {}

    def run(out: Path, day: date, close: float) -> None:
        md.official_closes = lambda: _closes(day, {"2330": close, "0050": 109.0})  # type: ignore[method-assign]
        md.official_events = lambda: NO_EVENTS  # type: ignore[method-assign]
        export_site(md, config, out, previous_site=_site(published))
        for path in out.rglob("*"):
            if path.is_file():
                published[path.relative_to(out).as_posix()] = path.read_bytes()

    run(tmp_path / "a", date(2026, 10, 15), 1000.0)
    run(tmp_path / "b", date(2026, 10, 16), 1010.0)
    history = json.loads(
        (tmp_path / "b" / "data" / "market" / "history" / "2330.json").read_text("utf-8")
    )
    assert history["closes"] == [["2026-10-15", 1000.0], ["2026-10-16", 1010.0]]
    assert len(history["scores"]) == 1  # one analysis day in this fixture (same as_of twice)
    index = json.loads((tmp_path / "b" / "data" / "index.json").read_text("utf-8"))
    assert index["archive"] == market.ARCHIVE and index["closes_date"] == "2026-10-16"
    assert any("OpenAPI" in line for line in index["disclaimer"])
    # Nothing personal can be in what is published: the export never sees holdings.
    text = " ".join(
        p.decode("utf-8", "replace") for k, p in published.items() if k.endswith(".json")
    )
    assert "trades" not in text and "cash_flows" not in text
