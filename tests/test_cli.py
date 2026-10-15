from __future__ import annotations

import pytest
import yaml
from typer.testing import CliRunner

from stock_research.cli import app
from tests.conftest import DEFAULT_CONFIG

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
