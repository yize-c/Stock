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
    # FinMind's terms: no raw data on the public site, and attribution on every page.
    assert "close" not in detail and "close" not in row
    assert any("FinMind" in line for line in detail["disclaimer"])
    assert any("FinMind" in line for line in index["disclaimer"])
    assert detail["report_markdown"].startswith("# 2330")
    assert "本工具僅供研究，不構成投資建議。" in detail["disclaimer"]
    assert {f["key"] for f in detail["frameworks"]} >= {"buffett", "warren_chou"}
    assert all("歷史統計值" in r["label"] for r in detail["risk"].values() if r["available"])
    assert {m["key"] for m in detail["metrics"]} >= {"volatility", "roe", "revenue_yoy"}
    # Spec v2 P1: half-life sensitivity and where the consistency score's evidence comes from.
    sensitivity = detail["half_life_sensitivity"]
    assert sensitivity["multipliers"] == ["0.5 倍", "1 倍", "2 倍"]
    assert len(sensitivity["composite"]["values"]) == 3
    assert "walk-forward" in detail["walk_forward_note"]
    assert "半衰期敏感度（V1）" in detail["report_markdown"]
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
    monkeypatch.setattr(cli, "open_market_data", lambda cfg, **_: md)
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


def test_web_app_is_dark_only_and_cache_is_versioned() -> None:
    """Spec v2 §5.2: dark colours everywhere, a new cache name, reduced motion respected."""
    from tests.conftest import REPO_ROOT

    web = REPO_ROOT / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    assert '<meta name="theme-color" content="#0a0a0b">' in html
    assert 'content="black-translucent"' in html
    manifest = json.loads((web / "manifest.webmanifest").read_text(encoding="utf-8"))
    assert manifest["background_color"] == manifest["theme_color"] == "#0a0a0b"
    assert 'const CACHE = "stock-v2";' in (web / "sw.js").read_text(encoding="utf-8")
    css = (web / "style.css").read_text(encoding="utf-8")
    assert "prefers-reduced-motion" in css and "--bg: #0a0a0b" in css
    assert "prefers-color-scheme" not in css  # one theme only


def _calibration(passed: bool) -> dict:
    tiers = [50000, 300000, 1000000, 3000000]
    stats = {
        "n": 100,
        "mean": 0.08,
        "ci_low": 0.02,
        "ci_high": 0.14,
        "etf_mean": 0.06,
        "rf_mean": 0.015,
    }
    period = {"months": 21, "score_range": [0, 100], "delisted_in_horizon": 0,
              "tiers": {str(t): stats for t in tiers}}  # fmt: skip
    return {
        "passed": passed,
        "gate": {
            "passed": passed,
            "reason": None if passed else "最高分組的樣本外報酬沒有高於最低分組",
        },
        "data_as_of": "2026-10-16",
        "base_dates": {
            "start": "2021-06-30",
            "last": "2025-09-30",
            "out_of_sample_start": "2024-01-31",
        },
        "horizon_months": 12,
        "groups": 10,
        "cutoffs": [10.0 * i for i in range(1, 10)],
        "table": [{"group": g, "in_sample": period, "out_of_sample": period} for g in range(1, 11)],
        "caps": {
            str(t): {str(g): {"cap": None, "reason": None} for g in range(1, 11)} for t in tiers
        },
    }


def test_export_attaches_the_research_score(raw_config, write_config, tmp_path) -> None:
    """The 0-100 score appears only once the calibration passed (spec v2 §4.2)."""
    from stock_research.calibration.summary import save

    metrics = tmp_path / "metric_walk_forward.json"
    metrics.write_text(
        json.dumps({"attempt": 1, "base_dates": ["2023-12-31", "2025-06-30"],
                    "metrics": {"roe": {"positive": True}, "eps": {"positive": True}}}),
        encoding="utf-8",
    )  # fmt: skip
    raw_config["markets"]["tw"]["listing"]["overrides"] = [TSMC_LISTING]
    raw_config["app"]["watchlist"] = ["2330"]
    raw_config["calibration"]["output"] = str(tmp_path / "calibration.json")
    raw_config["research_score"]["consistency"]["walk_forward_results"] = str(metrics)
    # The fixture history is too short for parts B-D; A and E (50 points) must be enough here.
    raw_config["research_score"]["min_available_points"] = 40
    config = load_config(write_config(raw_config))
    md, _ = build_market_data(config)

    export_site(md, config, tmp_path / "a")  # no calibration yet
    research = _read(tmp_path / "a" / "data" / "stocks" / "2330.json")["research_score"]
    assert research == {"available": False, "reason": "全市場歷史校準尚未完成，分數暫不顯示"}

    save(_calibration(False), config.calibration.output)
    export_site(md, config, tmp_path / "b")
    research = _read(tmp_path / "b" / "data" / "stocks" / "2330.json")["research_score"]
    assert not research["available"] and "不上線" in research["reason"]

    passed = _calibration(True) | {"overfit": {"warning": True, "message": "可能過度擬合"}}
    save(passed, config.calibration.output)
    export_site(md, config, tmp_path / "c")
    detail = _read(tmp_path / "c" / "data" / "stocks" / "2330.json")
    research = detail["research_score"]
    assert research["available"], research.get("reason")
    assert "可能過度擬合" in research["notes"]
    walk_forward = {m["key"]: m["walk_forward"] for m in detail["metrics"]}
    assert walk_forward["roe"] == "正向" and walk_forward.get("volatility") is None
    assert "ROE" in detail["walk_forward_note"]
    assert [t["capital"] for t in research["tiers"]] == [50000, 300000, 1000000, 3000000]
    assert all(0 <= t["score"] <= 100 and t["label"] for t in research["tiers"])
    assert {p["name"] for p in research["parts"]} >= {"投資大師原則", "籌碼"}
    assert "不構成投資建議" in research["disclaimer"]
    index = _read(tmp_path / "c" / "data" / "index.json")
    assert index["default_capital"] == 300000
    assert [t["label"] for t in index["capital_tiers"]] == [
        "5 萬・3 檔", "30 萬・5 檔", "100 萬・10 檔", "300 萬・15 檔",
    ]  # fmt: skip
    row = index["stocks"][0]
    default = next(t for t in research["tiers"] if t["capital"] == 300000)
    assert default["capital_label"] == "30 萬・5 檔"
    assert row["score"]["available"] and set(row["score"]["tiers"]) == {
        "50000", "300000", "1000000", "3000000",
    }  # fmt: skip
    assert row["score"]["tiers"]["300000"] == {"value": default["score"], "label": default["label"]}
