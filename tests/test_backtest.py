"""Historical backtest reference for the composite score (no network)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from stock_research import cli
from stock_research.backtest import score as bt
from stock_research.cli import app
from stock_research.config import ConfigError, load_config
from tests.conftest import DEFAULT_CONFIG
from tests.test_stability_report import synthetic_stock
from tests.test_validation_tools import FakeMarket

CONFIG = load_config(DEFAULT_CONFIG)
SETTINGS = CONFIG.backtest
LATER = synthetic_stock(date(2026, 10, 15))
runner = CliRunner()


def test_base_dates_need_the_shortest_horizon_to_have_passed() -> None:
    dates = bt.base_dates(date(2021, 6, 30), date(2026, 10, 16), 3)
    assert dates[0] == date(2021, 6, 30) and dates[-1] == date(2026, 6, 30)
    assert all((d + pd.offsets.MonthEnd(0)).date() == d for d in dates)


def test_forward_excess_is_relative_to_the_total_return_index() -> None:
    base = date(2024, 6, 28)
    found = bt.forward_excess(LATER, base, 6)
    stock = LATER.prices.adjusted.set_index("date")["adj_close"]
    index = LATER.benchmark.set_index("date")["total_return_index"]
    end = date(2024, 12, 28)

    def last(series, day):
        return float(series.loc[[d <= day for d in series.index]].iloc[-1])

    expected = (last(stock, end) / last(stock, base)) / (last(index, end) / last(index, base)) - 1
    assert found == pytest.approx(expected)
    assert bt.forward_excess(LATER, date(2026, 9, 30), 3) is None  # horizon not over yet


def test_v1_composite_uses_only_the_given_data() -> None:
    value = bt.v1_composite(replace(LATER, as_of=date(2024, 6, 30)), CONFIG)
    assert value is None or 0.0 <= value <= 1.0


def _stats(n: int, low: float, high: float) -> dict:
    return {"n": n, "mean": (low + high) / 2, "ci_low": low, "ci_high": high, "hit_rate": 0.5}


def test_label_rule() -> None:
    assert bt.label(_stats(100, 0.01, 0.05), SETTINGS)[0] == bt.BUY
    assert bt.label(_stats(100, -0.05, -0.01), SETTINGS)[0] == bt.SELL
    assert bt.label(_stats(100, -0.01, 0.03), SETTINGS)[0] == bt.HOLD
    name, reason = bt.label(_stats(5, 0.01, 0.05), SETTINGS)
    assert name is None and "樣本數 5" in reason


def test_buckets_cover_zero_to_one() -> None:
    assert bt.bucket_edges(SETTINGS) == [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    assert bt.bucket_of(0.0, SETTINGS) == (0.0, 0.4)
    assert bt.bucket_of(0.6, SETTINGS) == (0.6, 0.8)
    assert bt.bucket_of(1.0, SETTINGS) == (0.8, 1.0)


def _run(seed: int = 0) -> bt.BacktestRun:
    """High scores beat the index, low scores lag it, the middle is noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(20):
        for m in range(36):
            base = (pd.Timestamp("2021-06-30") + pd.offsets.MonthEnd(m)).date()
            score = [0.9, 0.5, 0.2][s % 3]
            drift = {0.9: 0.04, 0.5: 0.0, 0.2: -0.04}[score]
            row = {"code": str(s), "base_date": base, "status": "OK", "composite": score}
            for months in SETTINGS.horizons_months:
                row[f"excess_{months}m"] = drift + rng.normal(0, 0.02)
            rows.append(row)
    rows.append({"code": "x", "base_date": date(2021, 6, 30), "status": "ERROR", "reason": "boom"})
    return bt.BacktestRun(pd.DataFrame(rows), {"9999": "data checks failed"}, date(2026, 10, 16))


def test_summarize_labels_buckets_by_the_fixed_rule() -> None:
    summary = bt.summarize(_run(), CONFIG, datetime(2026, 10, 17, 4, 0))
    by_low = {b["low"]: b for b in summary["buckets"]}
    assert by_low[0.8]["label"] == bt.BUY
    assert by_low[0.0]["label"] == bt.SELL
    assert by_low[0.4]["label"] == bt.HOLD
    assert by_low[0.6]["label"] is None and by_low[0.6]["horizons"]["6"]["n"] == 0
    six = by_low[0.8]["horizons"]["6"]
    assert six["n"] == 7 * 36 and six["ci_low"] < six["mean"] < six["ci_high"]
    assert 0.9 < six["hit_rate"] <= 1.0
    assert summary["row_counts"] == {"OK": 720, "ERROR": 1}
    assert summary["excluded"] == {"9999": "data checks failed"}
    assert "倖存者偏差" in summary["disclaimer"]


def test_save_load_and_reference(tmp_path) -> None:
    run = _run()
    summary = bt.summarize(run, CONFIG, datetime(2026, 10, 17))
    path = tmp_path / "backtest" / "score_backtest.json"
    bt.save(run, summary, path)
    assert path.with_suffix(".parquet").exists()
    loaded = bt.load_summary(path)
    ref = bt.reference_for(0.95, loaded, CONFIG)
    assert ref["available"] and ref["label"] == bt.BUY and ref["bucket"] == [0.8, 1.0]
    assert bt.reference_for(None, loaded, CONFIG)["available"] is False
    assert "尚未產生" in bt.reference_for(0.5, None, CONFIG)["reason"]
    assert bt.load_summary(tmp_path / "missing.json") is None


def test_run_backtest(monkeypatch) -> None:
    monkeypatch.setattr(
        bt, "load_stock_data", lambda md, code, as_of, **_: replace(LATER, as_of=as_of)
    )
    monkeypatch.setattr(bt, "v1_composite", lambda data, config: 0.7)
    market = FakeMarket()
    market.listing_date = lambda code: date(2023, 1, 1) if code == "5678" else date(2000, 1, 1)
    run = bt.run_backtest(market, CONFIG, codes=["1234", "5678"])
    rows = run.rows
    early = rows.loc[(rows["code"] == "5678") & (rows["base_date"] < date(2023, 1, 1))]
    assert set(early["status"]) == {"NOT_LISTED"}
    ok = rows.loc[rows["status"] == "OK"]
    assert set(ok["code"]) == {"1234", "5678"} and (ok["composite"] == 0.7).all()
    # LATER ends 2026-10-15: a 12-month horizon from 2025-12-31 has not ended, 3 months has.
    last = ok.loc[ok["base_date"] == date(2025, 12, 31)].iloc[0]
    assert pd.isna(last["excess_12m"]) and not pd.isna(last["excess_3m"])
    blocked = bt.run_backtest(FakeMarket(healthy=False), CONFIG, codes=["1234"])
    assert "1234" in blocked.excluded and blocked.rows.empty


def test_config_validation(raw_config, write_config) -> None:
    raw_config["backtest"]["score_bins"] = [0.6, 0.4]
    with pytest.raises(ConfigError, match="increasing"):
        load_config(write_config(raw_config))
    raw_config["backtest"]["score_bins"] = [0.4, 0.6]
    raw_config["backtest"]["label_horizon_months"] = 9
    with pytest.raises(ConfigError, match="label_horizon_months"):
        load_config(write_config(raw_config))


def test_cli_backtest(monkeypatch, tmp_path, raw_config, write_config) -> None:
    raw_config["backtest"]["output"] = str(tmp_path / "bt" / "score_backtest.json")
    path = str(write_config(raw_config))
    monkeypatch.setattr(cli, "open_market_data", lambda cfg, **_: FakeMarket())
    monkeypatch.setattr(bt, "run_backtest", lambda md, cfg, **_: _run())
    result = runner.invoke(app, ["backtest", "--config", path])
    assert result.exit_code == 0, result.output
    assert bt.BUY in result.output and (tmp_path / "bt" / "score_backtest.json").exists()


def test_v1_composite_ignores_statements_published_after_the_base_date() -> None:
    """Spec v2 P0-3: filing deadlines hold in the backtest. Statements due after the base
    date (e.g. Q2 2024, due 2024-08-14, for a 2024-06-30 base) must not change the score."""
    base = date(2024, 6, 30)
    full = replace(LATER, as_of=base)
    visible = pd.to_datetime(LATER.quarterly["available_date"]) <= pd.Timestamp(base)
    trimmed = replace(full, quarterly=LATER.quarterly.loc[visible])
    assert (~visible).any()  # the synthetic data does contain later statements
    assert bt.v1_composite(full, CONFIG) == bt.v1_composite(trimmed, CONFIG)
