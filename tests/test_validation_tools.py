"""Tests for the tool-validation package (spec §14): no network, synthetic data only."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from stock_research import cli
from stock_research.analysis.stability import Verdict
from stock_research.analysis.stability_inputs import metric_specs, stability_report
from stock_research.cli import app
from stock_research.config import AppConfig, ConfigError, add_months, load_config
from stock_research.validation import walk_forward as wf
from stock_research.validation.aspects import (
    Judgement,
    aspect_metrics,
    aspect_verdicts,
    judgement,
    worst,
)
from stock_research.validation.experts import (
    COLUMNS,
    QUESTIONS,
    FormError,
    cohen_kappa,
    export_forms,
    fleiss,
    score_experts,
)
from stock_research.validation.forward_log import compare_entries, read_entries, write_entry
from stock_research.validation.known_cases import (
    confusion,
    load_known_cases,
    run_known_cases,
)
from stock_research.validation.outcomes import normalized_change, outcomes
from stock_research.validation.point_in_time import audit
from stock_research.validation.preregistration import (
    HoldoutBlocked,
    acquire_holdout_lock,
    check_holdout_allowed,
    file_sha256,
)
from stock_research.validation.report import (
    render_experts,
    render_forward,
    render_known_cases,
    render_walk_forward,
)
from stock_research.validation.statistics import auc, block_bootstrap, evaluate, wilson
from tests.conftest import DEFAULT_CONFIG
from tests.test_stability_report import synthetic_stock

CONFIG = load_config(DEFAULT_CONFIG)
TODAY = date(2026, 10, 16)
LATER = synthetic_stock(date(2025, 9, 30))  # one price path shared by decision and outcome
runner = CliRunner()


@pytest.fixture(scope="module")
def report():
    base = date(2024, 6, 30)
    return stability_report(replace(LATER, as_of=base), CONFIG, with_breakpoints=False)


class FakeMarket:
    """The MarketData surface the validation code uses."""

    def __init__(self, healthy: bool = True, listed: date | None = date(2000, 1, 1)) -> None:
        self.healthy, self.listed = healthy, listed

    def today(self) -> date:
        return TODAY

    @property
    def now(self) -> datetime:
        return datetime(2026, 10, 16, 15, 0)

    def resolve(self, code: str) -> str:
        return code

    def listing_date(self, ticker: str) -> date | None:
        return self.listed

    def health_report(self, code: str, as_of: date):
        check = SimpleNamespace(name="prices", passed=self.healthy)
        return SimpleNamespace(passed=self.healthy, checks=[check])

    def stock_info(self, code: str):
        return SimpleNamespace(name="測試公司")


def _fake_loader(market, code, as_of, **_):
    return replace(LATER, as_of=as_of)


# --- config ----------------------------------------------------------------------------------


def test_add_months_keeps_month_ends() -> None:
    assert add_months(date(2023, 11, 30), 3) == date(2024, 2, 29)
    assert add_months(date(2025, 6, 30), 3) == date(2025, 9, 30)
    assert add_months(date(2024, 1, 15), -2) == date(2023, 11, 15)
    assert add_months(date(2024, 1, 31), 1) == date(2024, 2, 29)


def test_config_keeps_dev_outcomes_out_of_the_holdout(raw_config, write_config) -> None:
    raw_config["validation"]["walk_forward"]["horizon_months"] = 4
    with pytest.raises(ConfigError, match="past the quarantine"):
        load_config(write_config(raw_config))
    raw_config["validation"]["walk_forward"]["horizon_months"] = 3
    raw_config["validation"]["walk_forward"]["holdout"]["start"] = date(2025, 9, 30)
    with pytest.raises(ConfigError, match="ordered"):
        load_config(write_config(raw_config))


def test_config_requires_every_aspect(raw_config, write_config) -> None:
    raw_config["validation"]["aspects"]["risk"] = []
    with pytest.raises(ConfigError, match="aspect"):
        load_config(write_config(raw_config))


def test_dev_outcome_limit_is_before_the_holdout() -> None:
    settings = CONFIG.validation.walk_forward
    assert settings.outcome_limit("dev") < settings.holdout.start
    assert settings.outcome_limit("holdout") == date(2026, 9, 30)


# --- aspects ---------------------------------------------------------------------------------


def test_judgement_mapping_and_worst() -> None:
    assert judgement(Verdict.STABLE) is Judgement.UNCHANGED
    assert judgement(Verdict.REGIME_SHIFT) is Judgement.CHANGED
    assert judgement(Verdict.UNSTABLE) is Judgement.CHANGED
    assert judgement(Verdict.INSUFFICIENT) is Judgement.EXCLUDED
    assert worst([Verdict.STABLE, Verdict.INSUFFICIENT]) is Verdict.STABLE
    assert worst([Verdict.REGIME_SHIFT, Verdict.UNSTABLE]) is Verdict.UNSTABLE
    assert worst([Verdict.INSUFFICIENT]) is Verdict.INSUFFICIENT


def test_aspect_verdicts(report) -> None:
    verdicts = aspect_verdicts(report.metrics, CONFIG.validation)
    by_key = {m.spec.key: m.verdict for m in report.metrics}
    assert verdicts["risk"] is worst(by_key[k] for k in ("volatility", "beta", "max_drawdown"))
    assert verdicts["overall"] is worst(verdicts[a] for a in ("revenue", "profitability", "risk"))
    bad = CONFIG.validation.model_copy(
        update={"aspects": {**CONFIG.validation.aspects, "risk": ["nope"]}}
    )
    with pytest.raises(KeyError, match="nope"):
        aspect_verdicts(report.metrics, bad)
    assert aspect_metrics(CONFIG.validation)[0] == "revenue_yoy"


# --- statistics ------------------------------------------------------------------------------


def test_auc_matches_hand_computation() -> None:
    assert auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == 0.75
    assert auc(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == 0.5  # ties count half
    assert auc(np.array([1, 2]), np.array([1, 1])) is None


def test_wilson_interval_known_value() -> None:
    interval = wilson(8, 10, 0.95)
    assert interval.estimate == 0.8
    assert interval.low == pytest.approx(0.4902, abs=1e-4)
    assert interval.high == pytest.approx(0.9433, abs=1e-4)
    assert wilson(0, 0, 0.95).estimate is None


def _simulated(informative: bool, seed: int, stocks: int = 30, months: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(stocks):
        unstable = rng.random() < 0.3  # a stock-level trait, so rows cluster by stock
        for m in range(months):
            base = (pd.Timestamp("2024-01-31") + pd.offsets.MonthEnd(m)).date()
            scale = 2.0 if (unstable and informative) else 1.0
            rows.append(
                {
                    "code": str(s),
                    "base_date": base,
                    "verdict": "UNSTABLE" if unstable else "STABLE",
                    "normalized": abs(rng.normal(0, scale)),
                }
            )
    return pd.DataFrame(rows)


def _evaluate(frame: pd.DataFrame, draws: int = 300):
    return evaluate(
        frame, draws_baseline=draws, draws_bootstrap=draws, confidence=0.95, alpha=0.05, seed=1
    )


def test_evaluation_passes_when_verdicts_are_informative() -> None:
    result = _evaluate(_simulated(True, seed=3))
    assert result.passed is True
    assert result.mwu_p < 0.05 and result.mwu_effect.low > 0.5
    assert result.tool_auc.estimate > result.baseline_b_quantile
    assert result.blocks == 30 * 4  # stock x quarter


def test_evaluation_fails_when_verdicts_are_noise() -> None:
    passes = [_evaluate(_simulated(False, seed=s), draws=200).passed for s in range(10)]
    assert sum(passes) <= 2  # about alpha of the time, not systematically


def test_evaluation_counts_exclusions_and_handles_no_unstable() -> None:
    frame = pd.DataFrame(
        {
            "code": ["1", "1", "2"],
            "base_date": [date(2024, 1, 31)] * 3,
            "verdict": ["STABLE", "INSUFFICIENT", "STABLE"],
            "normalized": [0.5, 2.0, np.nan],
        }
    )
    result = _evaluate(frame, draws=20)
    assert (result.rows, result.excluded_insufficient, result.excluded_no_outcome) == (1, 1, 1)
    assert result.passed is None and result.tool_auc.estimate is None


def test_block_bootstrap_widens_intervals_for_clustered_rows() -> None:
    rng = np.random.default_rng(0)
    values = np.repeat(rng.normal(size=40), 10)  # 40 blocks of 10 identical rows
    blocks = np.repeat(np.arange(40), 10)
    iid = np.arange(400)

    def stat(idx):
        return float(values[idx].mean())

    by_block = block_bootstrap(blocks, stat, 400, np.random.default_rng(1))
    naive = block_bootstrap(iid, stat, 400, np.random.default_rng(1))
    assert by_block.std() > 2.5 * naive.std()  # sqrt(10) ~ 3.2 in theory
    assert by_block.std() == pytest.approx(values.std() / np.sqrt(40), rel=0.25)


# --- outcomes --------------------------------------------------------------------------------


def test_normalized_change_rules() -> None:
    specs = metric_specs(CONFIG)
    assert normalized_change(specs["roe"], 0.20, 0.26) == pytest.approx(2.0)  # 6pp / 3pp
    assert normalized_change(specs["eps"], 10.0, 8.0) == pytest.approx(0.2 / 0.15)
    assert normalized_change(specs["eps"], 0.0, 0.0) == 0.0


def test_outcomes_use_only_the_horizon(report) -> None:
    base = date(2024, 6, 30)
    keys = aspect_metrics(CONFIG.validation)
    specs = {k: metric_specs(CONFIG)[k] for k in keys}
    references = {m.spec.key: m.values["post_2020"] for m in report.metrics}
    found = outcomes(
        specs, references, replace(LATER, as_of=base), LATER, base, 3,
        date(2025, 9, 30), CONFIG, 20,
    )  # fmt: skip
    end = date(2024, 9, 30)
    for key in keys:
        o = found[key]
        assert o.reason is None, (key, o.reason)
        assert base < o.outcome_date <= end or key in (
            "roe",
            "gross_margin",
            "operating_margin",
            "eps",
        )
        assert o.normalized == pytest.approx(normalized_change(specs[key], o.reference, o.future))
    # Next statement after 2024-06-30: Q2 2024, due 2024-08-14.
    assert found["roe"].outcome_date == date(2024, 8, 14)
    # Drawdown is compared with three-month blocks inside V1, not V1's whole drawdown.
    assert found["max_drawdown"].reference < report_value(report, "max_drawdown")
    # Volatility: forward daily returns, annualized.
    prices = LATER.prices.adjusted.set_index("date")["adj_close"]
    r = prices.pct_change().loc[[base < d <= end for d in prices.index]]
    assert found["volatility"].future == pytest.approx(r.std() * np.sqrt(252), rel=1e-9)


def report_value(report, key):
    return next(m.values["post_2020"] for m in report.metrics if m.spec.key == key)


def test_outcomes_refuse_horizons_past_the_limit_and_need_returns() -> None:
    base = date(2025, 7, 31)
    specs = {"volatility": metric_specs(CONFIG)["volatility"]}
    with pytest.raises(ValueError, match="outcome limit"):
        outcomes(specs, {}, LATER, LATER, base, 3, date(2025, 9, 30), CONFIG, 20)
    found = outcomes(
        specs, {"volatility": 0.3}, LATER, LATER, date(2025, 6, 30), 3, date(2025, 9, 30),
        CONFIG, 500,
    )  # fmt: skip
    assert (
        found["volatility"].normalized is None and "forward returns" in found["volatility"].reason
    )


def test_financial_outcome_never_reads_past_the_limit() -> None:
    specs = {"roe": metric_specs(CONFIG)["roe"]}
    # Base 2025-06-30: the next TTM (2025Q2) is due 2025-08-14, inside the dev limit.
    found = outcomes(
        specs, {"roe": 0.2}, LATER, LATER, date(2025, 6, 30), 3, date(2025, 9, 30), CONFIG, 20
    )
    assert found["roe"].outcome_date == date(2025, 8, 14)
    # With a limit before that date there is no outcome rather than a later one.
    found = outcomes(
        specs, {"roe": 0.2}, LATER, LATER, date(2025, 5, 31), 3, date(2025, 8, 31), CONFIG, 20
    )
    assert found["roe"].outcome_date == date(2025, 8, 14)
    with pytest.raises(ValueError):
        outcomes(
            specs, {"roe": 0.2}, LATER, LATER, date(2025, 6, 30), 3, date(2025, 8, 31), CONFIG, 20
        )


def test_unknown_metric_has_no_outcome_definition() -> None:
    with pytest.raises(KeyError, match="pe"):
        outcomes(
            {"pe": metric_specs(CONFIG)["pe"]},
            {},
            LATER,
            LATER,
            date(2025, 3, 31),
            3,
            date(2025, 9, 30),
            CONFIG,
            20,
        )


# --- walk-forward ----------------------------------------------------------------------------


def test_base_dates_skip_horizons_that_have_not_passed() -> None:
    settings = CONFIG.validation.walk_forward
    dev = wf.base_dates(settings.dev, 3, TODAY)
    assert dev[0] == date(2023, 12, 31) and dev[-1] == date(2025, 6, 30) and len(dev) == 19
    holdout = wf.base_dates(settings.holdout, 3, TODAY)
    assert holdout[-1] == date(2026, 6, 30)
    assert wf.base_dates(settings.holdout, 3, date(2026, 3, 15))[-1] == date(2025, 11, 30)


def test_overrides_accept_only_tunable_paths() -> None:
    tunable = CONFIG.validation.walk_forward.tunable
    parsed = wf.parse_overrides(["stability.absolute_tolerance.roe=0.04"], tunable)
    assert parsed == {"stability.absolute_tolerance.roe": 0.04}
    assert wf.apply_overrides(CONFIG, parsed).stability.absolute_tolerance["roe"] == 0.04
    with pytest.raises(ValueError, match="not tunable"):
        wf.parse_overrides(["validation.walk_forward.horizon_months=1"], tunable)
    with pytest.raises(ValueError, match="path=value"):
        wf.parse_overrides(["stability.relative_tolerance"], tunable)
    with pytest.raises(ValueError, match="unknown config path"):
        wf.apply_overrides(CONFIG, {"stability.nope.x": 1})
    with pytest.raises(ConfigError):
        wf.apply_overrides(CONFIG, {"stability.relative_tolerance": "wide"})


def test_tuning_log_is_append_only_and_capped(tmp_path) -> None:
    assert wf.next_attempt(tmp_path, 2) == 1
    wf.append_tuning_log(tmp_path, {"attempt": 1, "overrides": "{}", "tool_auc": 0.6})
    first = (tmp_path / wf.TUNING_LOG).read_text()
    assert wf.next_attempt(tmp_path, 2) == 2
    wf.append_tuning_log(tmp_path, {"attempt": 2, "overrides": "{}", "tool_auc": 0.4})
    after = (tmp_path / wf.TUNING_LOG).read_text()
    assert after.startswith(first)  # earlier attempts are never rewritten
    assert wf.tuning_attempts(tmp_path) == 2
    with pytest.raises(wf.TuningLimitReached):
        wf.next_attempt(tmp_path, 2)


def test_walk_forward_run(monkeypatch, report, tmp_path) -> None:
    config = CONFIG.model_copy(
        update={
            "validation": CONFIG.validation.model_copy(
                update={
                    "walk_forward": CONFIG.validation.walk_forward.model_copy(
                        update={
                            "universe": ["1234", "5678"],
                            "baseline_draws": 50,
                            "bootstrap_draws": 50,
                        }
                    )
                }
            )
        }
    )
    monkeypatch.setattr(wf, "load_stock_data", _fake_loader)
    monkeypatch.setattr(wf, "stability_report", lambda data, cfg, **_: report)
    market = FakeMarket()
    market.listing_date = lambda code: date(2024, 3, 1) if code == "5678" else date(2000, 1, 1)
    run = wf.run_walk_forward(market, config, "dev", attempt=1)
    rows = run.rows
    assert len(run.base_dates) == 19
    early = rows.loc[(rows["code"] == "5678") & (rows["base_date"] < date(2024, 3, 1))]
    assert set(early["status"]) == {"NOT_LISTED"}
    ok = rows.loc[rows["status"] == "OK"]
    assert set(ok["metric"]) == set(aspect_metrics(config.validation))
    assert set(ok["aspect"]) == {"revenue", "profitability", "risk"}
    # Dev outcomes never reach past the quarantine (spec §14.4 step 8).
    dated = ok["outcome_date"].dropna()
    assert max(dated) <= config.validation.walk_forward.quarantine.end
    assert run.evaluation.rows > 0
    path = wf.append_rows(tmp_path, rows)
    wf.append_rows(tmp_path, rows)
    assert len(pd.read_parquet(path)) == 2 * len(rows)
    text = render_walk_forward(run, 0.95, 1)
    assert "開發期" in text and "NOT_LISTED" in text and "不使用股價漲跌" in text
    row = wf.tuning_row(run, {}, market.now, "abc", False)
    assert row["attempt"] == 1 and row["rows"] == run.evaluation.rows


def test_walk_forward_records_failed_checks_and_errors(monkeypatch, report) -> None:
    config = CONFIG.model_copy(
        update={
            "validation": CONFIG.validation.model_copy(
                update={
                    "walk_forward": CONFIG.validation.walk_forward.model_copy(
                        update={"universe": ["1234"], "baseline_draws": 10, "bootstrap_draws": 10}
                    )
                }
            )
        }
    )
    monkeypatch.setattr(wf, "load_stock_data", _fake_loader)
    run = wf.run_walk_forward(FakeMarket(healthy=False), config, "dev", attempt=1)
    assert set(run.rows["status"]) == {"DATA_CHECK_FAILED"}

    def broken(*_, **__):
        raise ValueError("bad data")

    monkeypatch.setattr(wf, "stability_report", broken)
    run = wf.run_walk_forward(FakeMarket(), config, "dev", attempt=2)
    assert set(run.rows["status"]) == {"ERROR"}
    assert "bad data" in render_walk_forward(run, 0.95, 2)


def test_walk_forward_needs_completed_horizons(monkeypatch) -> None:
    market = FakeMarket()
    market.today = lambda: date(2024, 2, 1)
    with pytest.raises(ValueError, match="completed"):
        wf.run_walk_forward(market, CONFIG, "dev", attempt=1)


# --- preregistration -------------------------------------------------------------------------


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path) -> Path:
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "validation").mkdir()
    (tmp_path / "config.yaml").write_text("a: 1\n")
    return tmp_path


def _prereg(repo: Path, text: str) -> None:
    (repo / "validation" / "preregistration.md").write_text(text)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "--allow-empty", "-m", "prereg")


def test_holdout_blocked_until_preregistration_is_final_and_committed(repo) -> None:
    directory, config = repo / "validation", repo / "config.yaml"
    with pytest.raises(HoldoutBlocked, match="does not exist"):
        check_holdout_allowed(directory, config)
    (directory / "preregistration.md").write_text("Status: FINAL\n")
    with pytest.raises(HoldoutBlocked, match="not committed"):
        check_holdout_allowed(directory, config)
    _prereg(repo, "Status: DRAFT\n")
    with pytest.raises(HoldoutBlocked, match="FINAL"):
        check_holdout_allowed(directory, config)
    _prereg(repo, "Status: FINAL\n")
    with pytest.raises(HoldoutBlocked, match="Config SHA-256"):
        check_holdout_allowed(directory, config)
    _prereg(repo, "Status: FINAL\nConfig SHA-256: " + "0" * 64 + "\n")
    with pytest.raises(HoldoutBlocked, match="differs"):
        check_holdout_allowed(directory, config)
    good = f"Status: FINAL\nConfig SHA-256: `{file_sha256(config)}`\n"
    _prereg(repo, good)
    (directory / "preregistration.md").write_text(good + "edited\n")
    with pytest.raises(HoldoutBlocked, match="uncommitted"):
        check_holdout_allowed(directory, config)
    _prereg(repo, good)
    (config).write_text("a: 2\n")
    with pytest.raises(HoldoutBlocked, match="uncommitted"):
        check_holdout_allowed(directory, config)
    _git(repo, "checkout", "config.yaml")
    registration = check_holdout_allowed(directory, config)
    assert len(registration.commit) == 40
    acquire_holdout_lock(directory, registration, datetime(2026, 10, 16))
    with pytest.raises(HoldoutBlocked, match="only once"):
        check_holdout_allowed(directory, config)
    with pytest.raises(FileExistsError):
        acquire_holdout_lock(directory, registration, datetime(2026, 10, 17))


def test_holdout_blocked_outside_git(tmp_path) -> None:
    with pytest.raises(HoldoutBlocked, match="git"):
        check_holdout_allowed(tmp_path / "validation", tmp_path / "config.yaml")


def test_cli_refuses_holdout_without_preregistration(tmp_path, write_config, raw_config) -> None:
    path = write_config(raw_config)
    result = runner.invoke(
        app, ["validation", "run", "walk-forward", "--phase", "holdout", "--config", str(path)]
    )
    assert result.exit_code == 1 and "holdout blocked" in result.output
    result = runner.invoke(
        app,
        [
            "validation",
            "run",
            "walk-forward",
            "--phase",
            "holdout",
            "--set",
            "stability.alpha=0.1",
            "--config",
            str(path),
        ],
    )
    assert result.exit_code == 2
    result = runner.invoke(
        app,
        [
            "validation",
            "run",
            "walk-forward",
            "--phase",
            "dev",
            "--set",
            "general.timezone=UTC",
            "--config",
            str(path),
        ],
    )
    assert result.exit_code == 1 and "not tunable" in result.output
    result = runner.invoke(
        app, ["validation", "run", "walk-forward", "--phase", "x", "--config", str(path)]
    )
    assert result.exit_code == 2


def test_cli_dev_run_logs_the_attempt(
    tmp_path, monkeypatch, report, write_config, raw_config
) -> None:
    raw_config["validation"]["walk_forward"].update(
        {"universe": ["1234"], "baseline_draws": 10, "bootstrap_draws": 10}
    )
    path = write_config(raw_config)
    monkeypatch.setattr(cli, "open_market_data", lambda cfg: FakeMarket())
    monkeypatch.setattr(wf, "load_stock_data", _fake_loader)
    monkeypatch.setattr(wf, "stability_report", lambda data, cfg, **_: report)
    args = ["validation", "run", "walk-forward", "--phase", "dev", "--config", str(path)]
    result = runner.invoke(app, [*args, "--set", "stability.relative_tolerance=0.2"])
    assert result.exit_code == 0, result.output
    with (tmp_path / "validation" / wf.TUNING_LOG).open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["attempt"] == "1" and "relative_tolerance" in rows[0]["overrides"]
    assert (tmp_path / "validation" / "results" / "walk_forward_dev_attempt_01.md").exists()
    assert runner.invoke(app, args).exit_code == 0
    assert wf.tuning_attempts(tmp_path / "validation") == 2


# --- known cases -----------------------------------------------------------------------------


def _cases_file(path: Path, cases: list[dict]) -> Path:
    path.write_text(
        yaml.safe_dump(
            {"labeled_by": "test", "labeled_on": "2026-10-16", "cases": cases}, allow_unicode=True
        ),
        encoding="utf-8",
    )
    return path


def _case(code: str, value: str) -> dict:
    return {
        "code": code,
        "name": "測試",
        "expected": dict.fromkeys(("revenue", "profitability", "risk"), value),
        "reason": "test",
    }


def test_known_cases_file_validation(tmp_path) -> None:
    bad = _case("1", "changed")
    del bad["expected"]["risk"]
    with pytest.raises(ValueError, match="exactly"):
        load_known_cases(_cases_file(tmp_path / "a.yaml", [bad]))
    with pytest.raises(ValueError, match="'changed' or 'unchanged'"):
        load_known_cases(_cases_file(tmp_path / "b.yaml", [_case("1", "excluded")]))
    with pytest.raises(ValueError, match="twice"):
        load_known_cases(_cases_file(tmp_path / "c.yaml", [_case("1", "changed")] * 2))


def test_known_cases_run(monkeypatch, report, tmp_path) -> None:
    from stock_research.validation import known_cases as kc

    monkeypatch.setattr(kc, "load_stock_data", _fake_loader)
    monkeypatch.setattr(kc, "stability_report", lambda data, cfg, **_: report)
    cases = load_known_cases(
        _cases_file(tmp_path / "k.yaml", [_case("1", "changed"), _case("2", "unchanged")])
    )
    result = run_known_cases(FakeMarket(), CONFIG, cases)
    tool = aspect_verdicts(report.metrics, CONFIG.validation)
    decided = [
        a for a in ("revenue", "profitability", "risk") if tool[a] is not Verdict.INSUFFICIENT
    ]
    assert result.aspect_n == 2 * len(decided)
    # The same tool verdicts against opposite labels: exactly half are right.
    assert result.aspect_accuracy.estimate == pytest.approx(0.5)
    assert result.passed is False
    matrix = confusion(result.table)
    assert int(matrix.to_numpy().sum()) == 6
    text = render_known_cases(result, cases.labeled_by)
    assert "混淆矩陣" in text and "Wilson" in text
    failed = run_known_cases(FakeMarket(healthy=False), CONFIG, cases)
    assert set(failed.failures) == {"1", "2"} and failed.passed is None
    assert "無法執行" in render_known_cases(failed, "x")


def test_repository_known_cases_file_is_valid() -> None:
    repo = Path(__file__).resolve().parents[1]
    cases = load_known_cases(repo / CONFIG.validation.known_cases.file)
    assert len(cases.cases) >= 10
    assert {c.code for c in cases.cases} <= set(CONFIG.validation.walk_forward.universe)


# --- forward log -----------------------------------------------------------------------------


def test_forward_log_never_overwrites(tmp_path, report) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("x: 1\n")
    first = write_entry(tmp_path, "1234", report, CONFIG, config_path, datetime(2026, 10, 16))
    text = first.read_text()
    second = write_entry(tmp_path, "1234", report, CONFIG, config_path, datetime(2026, 10, 16, 1))
    assert first.name == "2024-06-30_1234.json" and second.name == "2024-06-30_1234_2.json"
    assert first.read_text() == text
    entries = read_entries(tmp_path)
    assert len(entries) == 2 and entries[0]["config_sha256"] == file_sha256(config_path)
    assert {m["key"] for m in entries[0]["metrics"]} >= set(aspect_metrics(CONFIG.validation))


def test_forward_compare(monkeypatch, report, tmp_path) -> None:
    from stock_research.validation import forward_log as fl

    write_entry(tmp_path, "1234", report, CONFIG, tmp_path / "missing.yaml", datetime(2026, 10, 16))
    monkeypatch.setattr(fl, "load_stock_data", _fake_loader)
    market = FakeMarket()
    market.today = lambda: date(2025, 9, 30)
    comparison = compare_entries(market, CONFIG, tmp_path, 6)
    assert comparison.pending == 0 and not comparison.failures
    assert set(comparison.rows["metric"]) == set(aspect_metrics(CONFIG.validation))
    assert "6 個月後" in render_forward(comparison)
    market.today = lambda: date(2024, 10, 31)
    assert compare_entries(market, CONFIG, tmp_path, 6).pending == 1


def test_cli_analyze_writes_forward_log_only_for_today(
    monkeypatch, tmp_path, raw_config, write_config
) -> None:
    from tests.test_market_data import TSMC_LISTING, build_market_data

    raw_config["markets"]["tw"]["listing"]["overrides"] = [TSMC_LISTING]
    path = write_config(raw_config)
    md, _ = build_market_data(load_config(path))
    monkeypatch.setattr(cli, "open_market_data", lambda cfg: md)
    args = ["analyze", "2330", "--no-peers", "--config", str(path)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    folder = tmp_path / "validation" / "forward_log"
    assert [p.name for p in folder.iterdir()] == [f"{md.today()}_2330.json"]
    assert runner.invoke(app, [*args, "--no-forward-log"]).exit_code == 0
    assert runner.invoke(app, [*args, "--as-of", "2026-10-14"]).exit_code == 0
    assert len(list(folder.iterdir())) == 1


# --- experts ---------------------------------------------------------------------------------


def test_cohen_and_fleiss_kappa() -> None:
    assert cohen_kappa(["是", "否", "是", "否"], ["是", "否", "是", "否"]) == 1.0
    # Classic 2x2 example: po = 0.7, pe = 0.5 -> kappa 0.4.
    a = ["是"] * 25 + ["否"] * 25
    b = ["是"] * 20 + ["否"] * 5 + ["是"] * 10 + ["否"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa(["是"], ["是"]) is None and cohen_kappa([], []) is None
    assert fleiss([["是", "是", "是"], ["否", "否", "否"], ["是", "是", "是"]]) == pytest.approx(
        1.0
    )
    assert fleiss([["是", "是", "是"]]) is None


def _filled(folder: Path, expert: str, answers: dict[str, str]) -> None:
    with (folder / f"filled_{expert}.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for code, value in answers.items():
            writer.writerow({"代號": code, **dict.fromkeys(QUESTIONS.values(), value)})


def test_export_and_score_experts(monkeypatch, report, tmp_path) -> None:
    from stock_research.validation import experts as ex

    monkeypatch.setattr(ex, "load_stock_data", _fake_loader)
    monkeypatch.setattr(ex, "stability_report", lambda data, cfg, **_: report)
    config = CONFIG.model_copy(
        update={"validation": CONFIG.validation.model_copy(update={"directory": tmp_path})}
    )
    folder = export_forms(FakeMarket(), config, ["1", "2", "3", "4"])
    template = (folder / "form_template.csv").read_text(encoding="utf-8-sig")
    assert "STABLE" not in template and "UNSTABLE" not in template  # blind
    with pytest.raises(FileExistsError):
        export_forms(FakeMarket(), config, ["1"])
    with pytest.raises(FormError, match="two"):
        score_experts(folder, config)
    tool = json.loads((folder / "tool_results.json").read_text(encoding="utf-8"))
    changed = judgement(Verdict(tool["verdicts"]["1"]["overall"])) is Judgement.CHANGED
    agree, other = ("是", "否") if changed else ("否", "是")
    _filled(folder, "E1", {"1": agree, "2": agree, "3": other, "4": "不確定"})
    _filled(folder, "E2", {"1": agree, "2": agree, "3": other, "4": agree})
    _filled(folder, "E3", {"1": agree, "2": other, "3": other, "4": other})
    scores = score_experts(folder, config)
    overall = scores.questions[0]
    assert overall.items == 4 and overall.unsure_share == pytest.approx(1 / 12)
    assert overall.ties == 1  # stock 4: one 是, one 否, one 不確定
    assert overall.expert_items == 3 and overall.majority_items == 3
    assert "專家間 kappa" in render_experts(scores)
    _filled(folder, "E4", {"1": "也許"})
    with pytest.raises(FormError, match="是／否／不確定"):
        score_experts(folder, config)


# --- point in time ---------------------------------------------------------------------------


def test_point_in_time_audit_flags_future_rows() -> None:
    base = date(2024, 6, 30)
    assert "prices.raw.date" in " ".join(audit(replace(LATER, as_of=base)))
    clean_prices = LATER.prices.raw.loc[
        pd.to_datetime(LATER.prices.raw["date"]) <= pd.Timestamp(base)
    ]
    clean_adjusted = LATER.prices.adjusted.loc[
        pd.to_datetime(LATER.prices.adjusted["date"]) <= pd.Timestamp(base)
    ]
    benchmark = LATER.benchmark.loc[pd.to_datetime(LATER.benchmark["date"]) <= pd.Timestamp(base)]
    quarterly = LATER.quarterly.loc[
        pd.to_datetime(LATER.quarterly["available_date"]) <= pd.Timestamp(base)
    ]
    revenue = LATER.revenue.loc[
        pd.to_datetime(LATER.revenue["available_date"]) <= pd.Timestamp(base)
    ]
    yoy = LATER.revenue_yoy.loc[
        pd.to_datetime(LATER.revenue_yoy["available_date"]) <= pd.Timestamp(base)
    ]
    dividends = LATER.dividends.loc[
        pd.to_datetime(LATER.dividends["available_date"]) <= pd.Timestamp(base)
    ]
    clean = replace(
        LATER,
        as_of=base,
        prices=replace(LATER.prices, raw=clean_prices, adjusted=clean_adjusted),
        benchmark=benchmark,
        quarterly=quarterly,
        revenue=revenue,
        revenue_yoy=yoy,
        dividends=dividends,
    )
    assert audit(clean) == []


def test_typed_config_round_trip_has_validation_section() -> None:
    assert isinstance(CONFIG, AppConfig)
    assert CONFIG.validation.walk_forward.max_tuning_attempts == 30


# --- remaining CLI commands ------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch, report, raw_config, write_config, tmp_path):
    from stock_research.validation import experts as ex
    from stock_research.validation import forward_log as fl
    from stock_research.validation import known_cases as kc
    from stock_research.validation import point_in_time as pit

    for module in (kc, fl, ex, pit):
        monkeypatch.setattr(module, "load_stock_data", _fake_loader)
    for module in (kc, ex):
        monkeypatch.setattr(module, "stability_report", lambda data, cfg, **_: report)
    monkeypatch.setattr(cli, "open_market_data", lambda cfg: FakeMarket())
    cases = _cases_file(tmp_path / "cases.yaml", [_case("1", "changed")])
    raw_config["validation"]["known_cases"]["file"] = str(cases)
    raw_config["validation"]["walk_forward"]["universe"] = ["1234"]
    return str(write_config(raw_config))


def test_cli_known_cases_and_point_in_time(cli_env, tmp_path) -> None:
    result = runner.invoke(app, ["validation", "run", "known-cases", "--config", cli_env])
    assert result.exit_code == 0, result.output
    assert "已知案例測試" in result.output
    assert list((tmp_path / "validation" / "results").glob("known_cases_*.csv"))
    # The synthetic stock always holds rows up to its own end date: the audit must say so.
    result = runner.invoke(app, ["validation", "run", "point-in-time", "--config", cli_env])
    assert result.exit_code == 1 and "after" in result.output
    result = runner.invoke(
        app,
        ["validation", "run", "point-in-time", "1234", "--date", "2026-10-16", "--config", cli_env],
    )
    assert result.exit_code == 0 and "1234 2026-10-16: OK" in result.output


def test_cli_forward_compare_and_experts(cli_env, tmp_path, report) -> None:
    write_entry(
        tmp_path / "validation", "1234", report, CONFIG, tmp_path / "x", datetime(2026, 10, 16)
    )
    result = runner.invoke(
        app, ["validation", "forward-compare", "--months", "12", "--config", cli_env]
    )
    assert result.exit_code == 0, result.output
    assert "12 個月後" in result.output
    result = runner.invoke(
        app, ["validation", "export-expert-forms", "1", "2", "--config", cli_env]
    )
    assert result.exit_code == 0, result.output
    again = runner.invoke(app, ["validation", "export-expert-forms", "1", "--config", cli_env])
    assert again.exit_code == 1 and "already exist" in again.output
    folder = tmp_path / "validation" / "expert_forms" / TODAY.isoformat()
    result = runner.invoke(app, ["validation", "score-experts", str(folder), "--config", cli_env])
    assert result.exit_code == 1 and "two" in result.output
    _filled(folder, "E1", {"1": "是", "2": "否"})
    _filled(folder, "E2", {"1": "是", "2": "是"})
    result = runner.invoke(app, ["validation", "score-experts", str(folder), "--config", cli_env])
    assert result.exit_code == 0, result.output
    assert (folder / "scores.md").exists()
