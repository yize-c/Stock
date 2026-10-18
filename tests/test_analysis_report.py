"""End-to-end: pipeline, report and CLI on fixture data (no network)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stock_research import cli
from stock_research.analysis.pipeline import AnalysisBlocked, run_analysis
from stock_research.cli import app
from stock_research.config import load_config
from stock_research.data.windows import WindowName
from stock_research.llm.client import LlmClient, RawResponse
from stock_research.report.analysis import money, num, pct, render_analysis, render_comparison
from tests.test_market_data import (
    AS_OF,
    TSMC_LISTING,
    base_routes,
    build_market_data,
    yahoo_like,
)

runner = CliRunner()


@pytest.fixture
def config(raw_config, write_config, tmp_path):
    raw_config["markets"]["tw"]["listing"]["overrides"] = [TSMC_LISTING]
    raw_config["llm"]["documents_dir"] = str(tmp_path / "docs")
    raw_config["llm"]["log_dir"] = str(tmp_path / "logs")
    raw_config["report"]["output_dir"] = str(tmp_path / "reports")
    return load_config(write_config(raw_config))


@pytest.fixture
def analysis(config):
    md, _ = build_market_data(config)
    return run_analysis(md, config, "2330", AS_OF, with_peers=False)


def test_report_has_every_section_and_the_disclaimer(analysis, config) -> None:
    text = render_analysis(analysis, config)
    for n, title in enumerate(
        [
            "摘要",
            "資料健康報告",
            "穩定性總覽",
            "基本面與月營收",
            "投資框架評分卡",
            "風險指標",
            "年報質化分析",
            "技術指標",
            "人工核對連結",
            "資料來源與時間範圍",
            "附錄：未驗證的 LLM 論點",
            "免責聲明",
        ],
        start=1,
    ):
        assert f"## {n}. {title}" in text
    assert text.rstrip().endswith("綜合分數只描述符合這些投資人原則的程度，不是買賣建議。")
    assert "本工具僅供研究，不構成投資建議。" in text
    assert "https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID=2330" in text
    assert "https://tw.stock.yahoo.com/quote/2330.TW" in text
    assert "不代表未來機率" in text
    assert "技術分析的實證效果有爭議" in text
    # Spec v2 P2-10: which principles each investor's rules pass or fail, in words.
    plain = text.split("### 白話解釋")[1].split("## 2.")[0]
    assert "**Warren Buffett**" in plain and "不是買賣建議" in plain
    # Columns are always V1, V3, V2 (reference).
    assert "| 項目 | 判定 | V1 | V3 | V2（參考） |" in text


def test_single_window_and_only_unstable_modes(analysis, config) -> None:
    single = render_analysis(analysis, config, window=WindowName.POST_2020)
    assert "單一版本模式" in single and "V3（2020 前）" not in single.split("## 4.")[1]
    focused = render_analysis(analysis, config, only_unstable=True)
    assert "✅ 穩定 |" not in focused.split("## 3.")[1].split("### 框架彙整")[0]


def test_comparison(analysis) -> None:
    text = render_comparison([analysis, analysis])
    assert text.count("2330 台積電") == 2 and "不是買賣建議" in text


def test_failed_data_checks_block_the_analysis(config) -> None:
    routes = base_routes()
    md, _ = build_market_data(
        config, routes, yahoo=yahoo_like("2330", routes, bump={date(2026, 10, 14): 1.02})
    )
    with pytest.raises(AnalysisBlocked, match="prices.cross_source"):
        run_analysis(md, config, "2330", AS_OF, with_peers=False)


def test_llm_section_and_appendix(config) -> None:
    folder = Path(config.llm.documents_dir) / "2330"
    folder.mkdir(parents=True)
    body = "本公司民國一百一十四年度營收為新台幣38,090億元，主要風險為地緣政治。" * 20
    (folder / "2026-03-31_114年報.txt").write_text(body, encoding="utf-8")
    good = {
        "claim": "114年度營收為38,090億元",
        "quote": "營收為新台幣38,090億元",
        "source_doc": "2026-03-31_114年報",
        "source_date": "2026-03-31",
    }
    bad = good | {"claim": "營收突破五兆元"}
    summary = {"status": "ok", "highlights": [good, bad], "risks": []}
    moat = {"status": "insufficient_data", "claims": []}
    explain = {"status": "ok", "paragraphs": ["這段沒有數字。"]}
    replies = [summary, moat, moat, explain]

    def transport(system, user, schema, cfg):
        reply = replies.pop(0)
        return RawResponse(json.dumps(reply, ensure_ascii=False), "end_turn", "m", {})

    md, _ = build_market_data(config)
    result = run_analysis(
        md,
        config,
        "2330",
        AS_OF,
        with_peers=False,
        with_llm=True,
        llm_client=lambda: LlmClient(config.llm, transport),
    )
    text = render_analysis(result, config)
    assert "114年度營收為38,090億元（引用：「營收為新台幣38,090億元」" in text
    assert "營收突破五兆元" in text.split("## 11.")[1]  # kept in the appendix
    assert "這段沒有數字。" in text
    # V3 has no document published before 2020.
    assert result.llm.tasks[WindowName.PRE_2020]["summary"].status.value == "NO_DOCUMENTS"


def test_llm_unavailable_is_reported(config) -> None:
    md, _ = build_market_data(config)

    def broken():
        raise RuntimeError("no credentials")

    result = run_analysis(
        md, config, "2330", AS_OF, with_peers=False, with_llm=True, llm_client=broken
    )
    assert "LLM 無法執行：RuntimeError: no credentials" in render_analysis(result, config)


def test_formatting_helpers() -> None:
    assert pct(0.1234) == "12.34%" and pct(None) == "—" and pct(float("nan")) == "—"
    assert num(float("inf")) == "∞" and num(1234.5) == "1,234.50"
    assert money(3.5e9) == "35.0 億" and money(12345.0) == "12,345"


def test_cli_analyze_and_breakpoints(config, monkeypatch, tmp_path) -> None:
    md, _ = build_market_data(config)
    monkeypatch.setattr(cli, "open_market_data", lambda cfg: md)
    path = str(tmp_path / "config.yaml")
    result = runner.invoke(app, ["analyze", "2330", "--no-peers", "--config", path, "--compare"])
    assert result.exit_code == 0, result.output
    report = Path(config.report.output_dir) / "2330_2026-10-15.md"
    assert report.exists() and "研究報告" in report.read_text(encoding="utf-8")
    bad = runner.invoke(app, ["analyze", "2330", "--window", "post_2020_early", "--config", path])
    assert bad.exit_code == 2
    result = runner.invoke(app, ["breakpoints", "2330", "--config", path])
    assert result.exit_code == 0 and "insufficient" in result.output


def test_cli_analyze_reports_blocked_stock(config, monkeypatch, tmp_path) -> None:
    routes = base_routes()
    md, _ = build_market_data(
        config, routes, yahoo=yahoo_like("2330", routes, bump={date(2026, 10, 14): 1.02})
    )
    monkeypatch.setattr(cli, "open_market_data", lambda cfg: md)
    result = runner.invoke(
        app, ["analyze", "2330", "--no-peers", "--config", str(tmp_path / "config.yaml")]
    )
    assert result.exit_code == 1 and "未通過" in result.output


def test_report_compares_returns_with_index_0050_and_deposits(analysis, config) -> None:
    text = render_analysis(analysis, config)
    assert "報酬比較" in text and "0050（還原，扣 ETF 來回成本）" in text
    assert "扣一次來回成本，部位 NT$100,000" in text
    assert "一年期定存（期間平均）" in text


def test_report_marks_half_life_sensitive_conclusions(analysis, config) -> None:
    """Spec v2 P1-8: what moves under 0.5x / 2x half-lives is listed with the note."""
    from dataclasses import replace

    from stock_research.analysis.frameworks.base import Outcome
    from stock_research.analysis.sensitivity import ItemSensitivity, ScoreSensitivity

    s = analysis.sensitivity
    moved = replace(
        s,
        composite=ScoreSensitivity({0.5: 0.8, 1.0: 0.6, 2.0: 0.5}, True, True),
        items=[
            ItemSensitivity(
                "buffett", "roe", "ROE", {0.5: Outcome.PASS, 1.0: Outcome.FAIL, 2.0: Outcome.FAIL},
                True,
            ),
            ItemSensitivity(
                "graham", "pe", "本益比", {0.5: Outcome.PASS, 1.0: Outcome.FAIL, 2.0: Outcome.PASS},
                False,
            ),
        ],
    )  # fmt: skip
    text = render_analysis(replace(analysis, sensitivity=moved), config)
    section = text.split("### 半衰期敏感度（V1）")[1].split("## 6.")[0]
    assert "| 綜合分數 | 80% | 60% | 50% | 可能只反映 2023 年後的 AI 多頭行情 |" in section
    assert "| Warren Buffett：ROE | 通過 | 未通過 | 未通過 | 可能只反映" in section
    assert "| Benjamin Graham：本益比 | 通過 | 未通過 | 通過 | 結論隨半衰期改變 |" in section
    plain = render_analysis(analysis, config)
    if not analysis.sensitivity.items:
        assert "所有框架項目的通過／未通過在各半衰期下都相同。" in plain


def test_report_says_which_metrics_count_for_consistency(analysis, config) -> None:
    from dataclasses import replace

    from stock_research.validation.metric_walk_forward import WalkForwardEvidence

    evidence = WalkForwardEvidence(frozenset({"roe", "beta"}), frozenset({"roe"}), "開發期第 1 次")
    text = render_analysis(replace(analysis, walk_forward=evidence), config)
    assert (
        "只採計 walk-forward 正向的指標" in text
        and "ROE；walk-forward 未正向、不採計：Beta" in text
    )
    none = render_analysis(replace(analysis, walk_forward=None), config)
    assert "尚無逐指標 walk-forward 結果" in none


def test_growth_is_described_as_not_a_forecast(analysis, config) -> None:
    text = render_analysis(analysis, config)
    section = text.split("### 成長與風險（最近 3 年）")[1].split("## 2.")[0]
    assert "成長分數：" in section and "不是預測" in section
    assert "| 營收年複合成長率 |" in section and "| 毛利率趨勢 |" in section
    assert "風險分數：" in section and "不是未來機率" in section
    assert "| 過熱程度 |" in section and "| 獲利穩定性 |" in section
