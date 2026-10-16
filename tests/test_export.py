"""Web-app export on fixture data (no network)."""

from __future__ import annotations

import json
from datetime import date

import pytest
from typer.testing import CliRunner

from stock_research import cli
from stock_research.cli import app
from stock_research.config import load_config
from stock_research.export import site
from stock_research.export.site import _clean, export_site
from tests.test_market_data import TSMC_LISTING, base_routes, build_market_data, yahoo_like

runner = CliRunner()


@pytest.fixture
def config(raw_config, write_config, tmp_path):
    raw_config["markets"]["tw"]["listing"]["overrides"] = [TSMC_LISTING]
    raw_config["app"]["watchlist"] = ["2330"]
    return load_config(write_config(raw_config))


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_writes_index_and_detail(config, tmp_path) -> None:
    md, _ = build_market_data(config)
    summary = export_site(md, config, tmp_path / "site")
    assert summary.ok == ["2330"] and not summary.blocked and not summary.errors
    index = _read(tmp_path / "site" / "data" / "index.json")
    assert index["as_of"] == md.today().isoformat()
    assert index["generated_at"].startswith(md.today().isoformat())
    row = index["stocks"][0]
    assert row["status"] == "ok" and row["name"] and row["overall"]["code"]
    assert set(row["aspects"]) == {"revenue", "profitability", "risk"}
    detail = _read(tmp_path / "site" / "data" / "stocks" / "2330.json")
    assert detail["close"]["note"] == "原始收盤價（未還原）"
    assert detail["report_markdown"].startswith("# 2330")
    assert "本工具僅供研究，不構成投資建議。" in detail["disclaimer"]
    assert {f["key"] for f in detail["frameworks"]} >= {"buffett", "warren_chou"}
    assert all("歷史統計值" in r["label"] for r in detail["risk"].values() if r["available"])
    assert {m["key"] for m in detail["metrics"]} >= {"volatility", "roe", "revenue_yoy"}
    text = (tmp_path / "site" / "data" / "stocks" / "2330.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text


def test_blocked_stock_is_published_with_failed_checks(config, tmp_path) -> None:
    routes = base_routes()
    md, _ = build_market_data(
        config, routes, yahoo=yahoo_like("2330", routes, bump={date(2026, 10, 14): 1.02})
    )
    summary = export_site(md, config, tmp_path)
    assert "2330" in summary.blocked and not summary.ok
    detail = _read(tmp_path / "data" / "stocks" / "2330.json")
    assert detail["status"] == "blocked" and "跨來源" in detail["problem"]
    assert not detail["health"]["passed"]
    assert _read(tmp_path / "data" / "index.json")["stocks"][0]["status"] == "blocked"


def test_analysis_errors_are_published_not_hidden(config, tmp_path, monkeypatch) -> None:
    md, _ = build_market_data(config)

    def broken(*_, **__):
        raise ValueError("boom")

    monkeypatch.setattr(site, "run_analysis", broken)
    summary = export_site(md, config, tmp_path)
    assert summary.errors == {"2330": "ValueError: boom"}
    assert _read(tmp_path / "data" / "stocks" / "2330.json")["status"] == "error"


def test_clean_makes_json_safe() -> None:
    assert _clean({"a": float("nan"), "b": [float("inf"), 1.5], "c": date(2026, 1, 2)}) == {
        "a": None,
        "b": [None, 1.5],
        "c": "2026-01-02",
    }


def test_cli_export(config, tmp_path, monkeypatch, write_config, raw_config) -> None:
    md, _ = build_market_data(config)
    monkeypatch.setattr(cli, "open_market_data", lambda cfg: md)
    raw_config["markets"]["tw"]["listing"]["overrides"] = [TSMC_LISTING]
    raw_config["app"]["watchlist"] = ["2330"]
    path = str(write_config(raw_config))
    result = runner.invoke(app, ["export", "--out", str(tmp_path / "out"), "--config", path])
    assert result.exit_code == 0, result.output
    assert "1 ok" in result.output

    def broken(*_, **__):
        raise ValueError("boom")

    monkeypatch.setattr(site, "run_analysis", broken)
    result = runner.invoke(app, ["export", "--out", str(tmp_path / "out2"), "--config", path])
    assert result.exit_code == 1 and "nothing should be published" in result.output


def test_watchlist_is_validated(raw_config, write_config) -> None:
    from stock_research.config import ConfigError

    raw_config["app"]["watchlist"] = ["2330", "2330"]
    with pytest.raises(ConfigError, match="duplicates"):
        load_config(write_config(raw_config))


def test_daily_workflow_runs_the_export_before_the_open() -> None:
    import yaml as _yaml

    from tests.conftest import REPO_ROOT

    workflow = _yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "daily-export.yml").read_text(encoding="utf-8")
    )
    cron = workflow[True]["schedule"][0]["cron"]  # YAML 1.1 reads "on" as True
    minute, hour, *_ = cron.split()
    taipei_hour = (int(hour) + 8) % 24
    assert taipei_hour < 9  # before the 09:00 open
    steps = workflow["jobs"]["export"]["steps"]
    run = next(s["run"] for s in steps if s.get("name") == "Export watchlist")
    assert "stock-research export --out site" in run
    env = next(s["env"] for s in steps if s.get("name") == "Export watchlist")
    assert env["FINMIND_TOKEN"] == "${{ secrets.FINMIND_TOKEN }}"  # never a literal token


def test_web_app_files_exist() -> None:
    from tests.conftest import REPO_ROOT

    web = REPO_ROOT / "web"
    for name in (
        "index.html",
        "app.js",
        "style.css",
        "sw.js",
        "manifest.webmanifest",
        "icon-192.png",
        "icon-512.png",
        "apple-touch-icon.png",
    ):
        assert (web / name).is_file(), name
    assert "data/index.json" in (web / "app.js").read_text(encoding="utf-8")
