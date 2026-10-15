from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from stock_research.config import (
    AppConfig,
    ConfigError,
    MissingSecretError,
    Secrets,
    load_config,
)
from tests.conftest import DEFAULT_CONFIG


def test_default_config_loads() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    assert isinstance(cfg, AppConfig)
    assert cfg.general.regime_split_date == date(2020, 1, 1)
    assert cfg.general.ifrs_start_year == 2013
    assert cfg.markets.tw.price_limits.rules[1].effective_from == date(2015, 6, 1)
    assert cfg.markets.tw.price_limits.rules[1].limit == pytest.approx(0.10)
    assert cfg.markets.tw.filing_deadlines.quarterly["Q4"].year_offset == 1
    assert cfg.weighting.half_life_days.daily_returns == 90
    assert cfg.stability.absolute_tolerance["roe"] == pytest.approx(0.03)


def test_default_scoring_weights_match_spec() -> None:
    groups = load_config(DEFAULT_CONFIG).scoring.groups
    assert groups["taiwan"].weight == pytest.approx(0.6)
    assert groups["international"].weight == pytest.approx(0.4)
    assert set(groups["taiwan"].members) == {"warren_chou", "guyu", "chen_chung_ming"}


def test_config_is_immutable() -> None:
    cfg = load_config(DEFAULT_CONFIG)
    with pytest.raises(ValueError):
        cfg.general.ifrs_start_year = 2000  # type: ignore[misc]


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("general: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_config(path)


def test_non_mapping_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_unknown_key_raises(raw_config, write_config) -> None:
    raw_config["stability"]["relative_tolerence"] = 0.2  # typo
    with pytest.raises(ConfigError, match="relative_tolerence"):
        load_config(write_config(raw_config))


def test_missing_section_raises(raw_config, write_config) -> None:
    del raw_config["weighting"]
    with pytest.raises(ConfigError, match="weighting"):
        load_config(write_config(raw_config))


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
def test_alpha_out_of_range_raises(raw_config, write_config, alpha) -> None:
    raw_config["stability"]["alpha"] = alpha
    with pytest.raises(ConfigError, match="alpha"):
        load_config(write_config(raw_config))


@pytest.mark.parametrize("half_life", [0, -30])
def test_non_positive_half_life_raises(raw_config, write_config, half_life) -> None:
    raw_config["weighting"]["half_life_days"]["daily_returns"] = half_life
    with pytest.raises(ConfigError, match="daily_returns"):
        load_config(write_config(raw_config))


def test_group_weights_must_sum_to_one(raw_config, write_config) -> None:
    raw_config["scoring"]["groups"]["taiwan"]["weight"] = 0.7
    raw_config["scoring"]["groups"]["taiwan"]["members"]["guyu"] = 0.3
    with pytest.raises(ConfigError, match=r"group weights sum to"):
        load_config(write_config(raw_config))


def test_member_weights_must_sum_to_group_weight(raw_config, write_config) -> None:
    raw_config["scoring"]["groups"]["international"]["members"]["munger"] = 0.2
    with pytest.raises(ConfigError, match="member weights sum to"):
        load_config(write_config(raw_config))


def test_every_framework_must_be_scored(raw_config, write_config) -> None:
    members = raw_config["scoring"]["groups"]["international"]["members"]
    del members["munger"]
    members["buffett"] = 0.2
    with pytest.raises(ConfigError, match="missing from scoring groups"):
        load_config(write_config(raw_config))


def test_unknown_framework_in_scoring_raises(raw_config, write_config) -> None:
    members = raw_config["scoring"]["groups"]["international"]["members"]
    del members["munger"]
    members["soros"] = 0.1
    with pytest.raises(ConfigError, match="unknown frameworks"):
        load_config(write_config(raw_config))


def test_price_limit_rules_must_be_ordered(raw_config, write_config) -> None:
    rules = raw_config["markets"]["tw"]["price_limits"]["rules"]
    rules.append({"effective_from": date(2010, 1, 1), "limit": 0.08})
    with pytest.raises(ConfigError, match="strictly increasing"):
        load_config(write_config(raw_config))


def test_first_price_limit_rule_must_be_open_ended(raw_config, write_config) -> None:
    raw_config["markets"]["tw"]["price_limits"]["rules"][0]["effective_from"] = date(2000, 1, 1)
    with pytest.raises(ConfigError, match="effective_from: null"):
        load_config(write_config(raw_config))


def test_filing_deadline_must_be_real_date(raw_config, write_config) -> None:
    raw_config["markets"]["tw"]["filing_deadlines"]["quarterly"]["Q1"]["day"] = 31
    raw_config["markets"]["tw"]["filing_deadlines"]["quarterly"]["Q1"]["month"] = 4
    with pytest.raises(ConfigError, match="invalid deadline"):
        load_config(write_config(raw_config))


def test_all_quarter_deadlines_required(raw_config, write_config) -> None:
    del raw_config["markets"]["tw"]["filing_deadlines"]["quarterly"]["Q3"]
    with pytest.raises(ConfigError, match="Q3"):
        load_config(write_config(raw_config))


def test_manual_closures_parse(raw_config, write_config) -> None:
    raw_config["markets"]["tw"]["trading_calendar"]["manual_closures"] = [
        {"date": date(2024, 7, 25), "reason": "Typhoon Gaemi"}
    ]
    cfg = load_config(write_config(raw_config))
    assert cfg.markets.tw.trading_calendar.manual_closures[0].date == date(2024, 7, 25)


def test_overlapping_risk_free_periods_raise(raw_config, write_config) -> None:
    raw_config["markets"]["tw"]["risk_free"]["fallback_rates"] = [
        {"start": date(2016, 1, 1), "end": date(2019, 12, 31), "rate": 0.01},
        {"start": date(2019, 6, 1), "end": date(2021, 12, 31), "rate": 0.008},
    ]
    with pytest.raises(ConfigError, match="overlap"):
        load_config(write_config(raw_config))


def test_fixed_split_requires_date(raw_config, write_config) -> None:
    raw_config["stability"]["internal_split"]["mode"] = "fixed"
    with pytest.raises(ConfigError, match="requires fixed_date"):
        load_config(write_config(raw_config))


def test_fixed_split_must_be_after_regime_split(raw_config, write_config) -> None:
    raw_config["stability"]["internal_split"] = {"mode": "fixed", "fixed_date": date(2019, 6, 1)}
    with pytest.raises(ConfigError, match="after regime_split_date"):
        load_config(write_config(raw_config))


def test_crash_quarters_format(raw_config, write_config) -> None:
    raw_config["stability"]["crash_exclusion"]["quarters"] = ["2020-Q1"]
    with pytest.raises(ConfigError, match="2020Q1"):
        load_config(write_config(raw_config))


def test_ifrs_start_must_precede_split(raw_config, write_config) -> None:
    raw_config["general"]["ifrs_start_year"] = 2020
    with pytest.raises(ConfigError, match="ifrs_start_year"):
        load_config(write_config(raw_config))


def test_default_market_must_be_enabled(raw_config, write_config) -> None:
    raw_config["general"]["default_market"] = "us"
    with pytest.raises(ConfigError, match="not enabled"):
        load_config(write_config(raw_config))


def test_config_contains_no_secrets() -> None:
    dumped = load_config(DEFAULT_CONFIG).model_dump_json().lower()
    for word in ("token", "api_key", "apikey", "secret", "password"):
        assert word not in dumped


def test_secret_missing_raises() -> None:
    with pytest.raises(MissingSecretError, match="FINMIND_TOKEN"):
        Secrets(env_file=None).get(Secrets.FINMIND_TOKEN)


def test_secret_blank_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    with pytest.raises(MissingSecretError):
        Secrets(env_file=None).get(Secrets.ANTHROPIC_API_KEY)


def test_secret_read_from_env_file(tmp_path: Path) -> None:
    env = tmp_path / "custom.env"
    env.write_text("FINMIND_TOKEN=abc123\n", encoding="utf-8")
    assert Secrets(env_file=env).get(Secrets.FINMIND_TOKEN) == "abc123"


def test_unknown_secret_name_raises() -> None:
    with pytest.raises(KeyError):
        Secrets(env_file=None).get("AWS_SECRET")


def test_secrets_repr_hides_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINMIND_TOKEN", "super-secret-value")
    text = repr(Secrets(env_file=None))
    assert "super-secret-value" not in text
    assert "FINMIND_TOKEN=set" in text
    assert "ANTHROPIC_API_KEY=unset" in text
