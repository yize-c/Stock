from __future__ import annotations

from datetime import date

import pytest
import yaml
from typer.testing import CliRunner

from stock_research import cli
from stock_research.cli import app
from stock_research.config import load_config
from tests.conftest import DEFAULT_CONFIG
from tests.test_market_data import TSMC_LISTING, base_routes, build_market_data, yahoo_like

runner = CliRunner()


def test_config_show_prints_validated_config() -> None:
    result = runner.invoke(app, ["config", "show", "--config", str(DEFAULT_CONFIG)])
    assert result.exit_code == 0, result.output
    config_part = result.output.split("secrets:")[0]
    data = yaml.safe_load(config_part)
    assert data["general"]["regime_split_date"] == "2020-01-01"
    assert "FINMIND_TOKEN: unset" in result.output


def test_config_show_never_prints_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINMIND_TOKEN", "finmind-should-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-leak")
    result = runner.invoke(app, ["config", "show", "--config", str(DEFAULT_CONFIG)])
    assert result.exit_code == 0, result.output
    assert "should-not-leak" not in result.output
    assert "FINMIND_TOKEN: set" in result.output
    assert "ANTHROPIC_API_KEY: set" in result.output


def test_config_show_fails_loudly_on_invalid_config(tmp_path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("general: {}\n", encoding="utf-8")
    result = runner.invoke(app, ["config", "show", "--config", str(bad)])
    assert result.exit_code == 1
    assert "invalid config" in result.output


def test_missing_default_config_fails() -> None:
    # The autouse fixture chdirs into an empty temp dir, so ./config.yaml does not exist.
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 1
    assert "not found" in result.output


@pytest.fixture
def fake_market_data(monkeypatch: pytest.MonkeyPatch, raw_config, write_config):
    """Point ``validate`` at fixture data; returns the config path to pass on the CLI."""

    def _install(overrides=(TSMC_LISTING,), yahoo=None):
        raw_config["markets"]["tw"]["listing"]["overrides"] = list(overrides)
        path = write_config(raw_config)
        market_data, _ = build_market_data(load_config(path), yahoo=yahoo)
        monkeypatch.setattr(cli, "open_market_data", lambda cfg: market_data)
        return str(path)

    return _install


def test_validate_prints_report_and_succeeds(fake_market_data) -> None:
    path = fake_market_data()
    result = runner.invoke(app, ["validate", "2330", "--config", path])
    assert result.exit_code == 0, result.output
    assert "# 資料健康報告：2330" in result.output
    assert "結果：**通過**" in result.output
    assert "本工具僅供研究，不構成投資建議。" in result.output


def test_validate_fails_with_exit_code_1(fake_market_data) -> None:
    bumped = yahoo_like("2330", base_routes(), bump={date(2026, 10, 14): 1.02})
    path = fake_market_data(yahoo=bumped)
    result = runner.invoke(app, ["validate", "2330", "--config", path, "--as-of", "2026-10-15"])
    assert result.exit_code == 1
    assert "未通過" in result.output and "跨來源收盤價比對 | 失敗" in result.output


@pytest.mark.parametrize("as_of", ["2026/10/15", "2026-10-16"])
def test_validate_rejects_bad_or_future_as_of(fake_market_data, as_of) -> None:
    path = fake_market_data()
    result = runner.invoke(app, ["validate", "2330", "--config", path, "--as-of", as_of])
    assert result.exit_code == 2


def test_validate_needs_finmind_token() -> None:
    result = runner.invoke(app, ["validate", "2330", "--config", str(DEFAULT_CONFIG)])
    assert result.exit_code == 2
    assert "FINMIND_TOKEN is not set" in result.output
