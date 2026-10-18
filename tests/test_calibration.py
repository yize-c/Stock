"""Full-market calibration (spec v2 §4.2): liquidity screen, horizons, groups, caps, gate."""

from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_research.calibration import pipeline, summary
from stock_research.config import load_config
from tests.conftest import DEFAULT_CONFIG

CONFIG = load_config(DEFAULT_CONFIG)


def _config(**changes):
    calibration = CONFIG.calibration.model_copy(update=changes)
    return CONFIG.model_copy(update={"calibration": calibration})


def test_base_dates_are_month_ends_up_to_the_last_one() -> None:
    dates = pipeline.base_dates(CONFIG.calibration)
    assert dates[0] == date(2021, 6, 30) and dates[-1] == date(2025, 9, 30)
    assert len(dates) == 52


class _Market:
    """Two stocks with daily turnover 100 (A) and 10 (B); B is listed in March 2024."""

    def __init__(self) -> None:
        days = [d.date() for d in pd.bdate_range("2023-06-01", "2024-07-10")]
        self.daily = {
            "1111": pd.DataFrame({"date": days, "close": 10.0, "turnover": 100.0}),
            "2222": pd.DataFrame({"date": days, "close": 10.0, "turnover": 10.0}),
        }

    def resolve(self, code):
        return SimpleNamespace(code=code)

    def raw_daily(self, ticker):
        return self.daily[ticker.code]

    def listing_date(self, ticker):
        return date(2024, 3, 1) if ticker.code == "2222" else None


def test_liquidity_screen_keeps_the_most_traded_after_listing() -> None:
    config = _config(
        start=date(2024, 1, 31),
        last_base_date=date(2024, 6, 30),
        out_of_sample_start=date(2024, 3, 31),
    )
    found = pipeline.Universe(["1111", "2222"], {"1111": date(2024, 6, 15)})
    frame = pipeline.liquidity(_Market(), config, found)
    by = {
        (c, d): k for c, d, k in zip(frame["code"], frame["base_date"], frame["kept"], strict=True)
    }
    # 2222 has 60 listed trading days only from late May; 1111 is delisted from 2024-06-15.
    assert ("2222", date(2024, 4, 30)) not in by and ("2222", date(2024, 5, 31)) in by
    assert ("1111", date(2024, 6, 30)) not in by
    assert by[("1111", date(2024, 5, 31))] and not by[("2222", date(2024, 5, 31))]  # top half
    assert frame.loc[frame["code"] == "1111", "liquidity"].iloc[0] == pytest.approx(100.0)


def test_forward_gross_with_halts_and_delistings() -> None:
    days = [d.date() for d in pd.bdate_range("2024-01-01", "2025-03-31")]
    adjusted = pd.Series(np.linspace(100, 150, len(days)), index=days)
    gross, delisted = pipeline.forward_gross(adjusted, date(2024, 1, 31), 12, None)
    assert gross == pytest.approx(
        adjusted.loc[date(2025, 1, 31)] / adjusted.loc[date(2024, 1, 31)] - 1
    )
    assert not delisted
    stopped = adjusted.loc[[d <= date(2024, 9, 30) for d in adjusted.index]]
    gross, delisted = pipeline.forward_gross(stopped, date(2024, 1, 31), 12, date(2024, 10, 1))
    assert delisted and gross == pytest.approx(
        stopped.iloc[-1] / adjusted.loc[date(2024, 1, 31)] - 1
    )
    # Not delisted but no close near the horizon end (a long halt): no outcome.
    assert pipeline.forward_gross(stopped, date(2024, 1, 31), 12, None) == (None, False)
    assert pipeline.forward_gross(adjusted, date(2023, 6, 30), 12, None) == (None, False)


def test_groups_are_assigned_within_each_month() -> None:
    frame = pd.DataFrame(
        {
            "base_date": [date(2024, 1, 31)] * 4 + [date(2024, 2, 29)] * 4,
            "score": [10, 20, 30, 40, 70, 60, 50, 40],
        }
    )
    grouped = summary.assign_groups(frame, 2)
    assert list(grouped["group"]) == [1, 1, 2, 2, 2, 2, 1, 1]


def _rows(slope: float, in_sample_slope: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ten stocks per month; a higher score earns ``slope`` more 12-month gross return
    (``in_sample_slope`` before the out-of-sample start, when given)."""
    rng = np.random.default_rng(0)
    rows, bench = [], []
    oos_start = CONFIG.calibration.out_of_sample_start
    for base in pipeline.base_dates(CONFIG.calibration):
        step = slope if in_sample_slope is None or base >= oos_start else in_sample_slope
        bench.append({"base_date": base, "etf_gross": 0.05, "etf_close": 150.0, "rf": 0.015})
        for i in range(10):
            rows.append(
                {
                    "code": f"{1100 + i}",
                    "base_date": base,
                    "status": "OK",
                    "score": 40 + 4 * i + rng.normal(0, 0.1),
                    "gross": -0.05 + step * i + rng.normal(0, 0.01),
                    "delisted_in_horizon": False,
                }
            )
            rows[-1] |= {  # the aggressive preference ranks the stocks the other way round
                "score_conservative": rows[-1]["score"],
                "score_balanced": rows[-1]["score"] + 1.0,
                "score_aggressive": 80 - 4 * i,
            }
            for tier in CONFIG.research_score.capital_tiers:
                rows[-1][f"net_{tier.capital}"] = pipeline.tier_net(
                    rows[-1]["gross"], 50.0, tier.capital, tier.holdings, False, CONFIG
                )
    rows.append({"code": "9999", "base_date": base, "status": "ERROR", "reason": "boom"})
    return pd.DataFrame(rows), pd.DataFrame(bench)


def _screen(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows.loc[rows["status"] == "OK", ["code", "base_date"]]
    return ok.assign(liquidity=1.0, kept=True)


def test_summary_caps_and_gate_when_higher_scores_earned_more() -> None:
    config = _config(groups=5, bootstrap_draws=50)
    rows, bench = _rows(0.02)
    result = summary.summarize(
        rows, _screen(rows), bench, config, datetime(2026, 10, 17, tzinfo=UTC)
    )
    assert result["passed"] and result["gate"]["rank_correlation"] == pytest.approx(1.0)
    assert len(result["cutoffs"]) == 4 and result["cutoffs"] == sorted(result["cutoffs"])
    caps = result["caps"]["300000"]
    # Group 1 (gross about -4%) is not above the deposit rate; group 3 (about +5%, less
    # costs) trails 0050's 5% less its smaller ETF cost; the top group beats both.
    assert caps["1"]["cap"] == 40 and caps["3"]["cap"] == 55 and caps["5"]["cap"] is None
    top = result["table"][-1]["out_of_sample"]["tiers"]["300000"]
    assert top["ci_low"] < top["mean"] < top["ci_high"] and top["n"] == 2 * 21
    assert result["universe"]["status_counts"] == {"OK": 520, "ERROR": 1}
    assert "不代表未來" in result["disclaimer"]


def test_every_preference_is_calibrated_on_its_own_score() -> None:
    config = _config(groups=5, bootstrap_draws=20)
    rows, bench = _rows(0.02)
    result = summary.summarize(
        rows, _screen(rows), bench, config, datetime(2026, 10, 17, tzinfo=UTC)
    )
    per = result["preferences"]
    assert set(per) == {"conservative", "balanced", "aggressive"}
    assert per["conservative"]["passed"] and per["balanced"]["passed"]
    assert not per["aggressive"]["passed"]  # its higher scores earned less
    assert result["default_preference"] == "conservative"
    assert result["cutoffs"] == per["conservative"]["cutoffs"]
    aggressive = summary.for_preference(result, "aggressive")
    assert aggressive["cutoffs"] == per["aggressive"]["cutoffs"]
    assert aggressive["data_as_of"] == result["data_as_of"]
    assert summary.for_preference({"passed": True}, "aggressive") is None  # older file


def test_rows_scored_with_an_older_definition_are_refused() -> None:
    rows, bench = _rows(0.02)
    old = rows.drop(columns=["score_conservative", "score_balanced", "score_aggressive"])
    with pytest.raises(summary.RowsError, match="older score definition"):
        summary.summarize(old, _screen(rows), bench, CONFIG, datetime(2026, 10, 17, tzinfo=UTC))


def test_overfitting_warning_compares_in_and_out_of_sample_spreads() -> None:
    config = _config(groups=5, bootstrap_draws=20)
    when = datetime(2026, 10, 17, tzinfo=UTC)
    rows, bench = _rows(0.02)
    same = summary.summarize(rows, _screen(rows), bench, config, when)["overfit"]
    assert not same["warning"] and same["times"] == pytest.approx(1.0, abs=0.1)
    rows, bench = _rows(0.02, in_sample_slope=0.05)  # in sample 2.5x the spread
    result = summary.summarize(rows, _screen(rows), bench, config, when)
    overfit = result["overfit"]
    assert overfit["warning"] and overfit["times"] == pytest.approx(2.5, abs=0.15)
    assert "可能過度擬合" in overfit["message"] and result["passed"]  # a warning only


def test_gate_fails_when_scores_do_not_rank_returns() -> None:
    config = _config(groups=5, bootstrap_draws=20)
    rows, bench = _rows(-0.02)
    result = summary.summarize(
        rows, _screen(rows), bench, config, datetime(2026, 10, 17, tzinfo=UTC)
    )
    assert not result["passed"] and "最高分組" in result["gate"]["reason"]


def test_save_and_load(tmp_path) -> None:
    path = tmp_path / "calibration" / "score_calibration.json"
    summary.save({"passed": False}, path)
    assert summary.load(path) == {"passed": False}
    assert summary.load(tmp_path / "missing.json") is None


def test_workflow_matrix_matches_the_configured_shards() -> None:
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "calibration.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    shards = workflow["jobs"]["score"]["strategy"]["matrix"]["shard"]
    assert shards == list(range(CONFIG.calibration.shards))
    run = "\n".join(s.get("run", "") for s in workflow["jobs"]["score"]["steps"])
    assert f"--shards {CONFIG.calibration.shards}" in run
    # The calibration uses the second token, never the daily export's.
    text = path.read_text(encoding="utf-8")
    assert "secrets.SECOND_FINMIND" in text and "secrets.FINMIND_TOKEN" not in text


def test_cli_score_and_summarize(monkeypatch, tmp_path, raw_config, write_config) -> None:
    """The commands wire the stages together (stages replaced by fakes; no network)."""
    from typer.testing import CliRunner

    from stock_research import cli

    raw_config["calibration"]["work_dir"] = str(tmp_path / "work")
    raw_config["calibration"]["output"] = str(tmp_path / "score_calibration.json")
    path = str(write_config(raw_config))
    rows, bench = _rows(0.02)
    monkeypatch.setattr(
        cli,
        "_calibration_market",
        lambda cfg: SimpleNamespace(now=datetime(2026, 10, 17, tzinfo=UTC)),
    )
    monkeypatch.setattr(
        pipeline, "universe", lambda md, cfg: pipeline.Universe(sorted(set(rows["code"])), {})
    )
    monkeypatch.setattr(pipeline, "liquidity", lambda md, cfg, found: _screen(rows))
    monkeypatch.setattr(
        pipeline, "score_shard", lambda md, cfg, kept, delisted, shard, shards, progress: rows
    )
    monkeypatch.setattr(summary, "benchmarks", lambda md, cfg, dates: bench)
    runner = CliRunner()
    assert runner.invoke(cli.app, ["calibration", "liquidity", "--config", path]).exit_code == 0
    result = runner.invoke(
        cli.app, ["calibration", "score", "--shard", "0", "--shards", "1", "--config", path]
    )
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli.app, ["calibration", "summarize", "--config", path])
    assert result.exit_code == 0, result.output
    assert "[conservative] gate passed: True" in result.output
    assert summary.load(tmp_path / "score_calibration.json")["passed"] is True
