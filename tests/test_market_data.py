from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd
import pytest

from stock_research.analysis.frameworks import run_all
from stock_research.analysis.inputs import build_inputs
from stock_research.analysis.scoring import composite
from stock_research.config import AppConfig, ManualClosure, Secrets, load_config
from stock_research.data.cache import Cache
from stock_research.data.errors import AnomalyError, MissingDataError
from stock_research.data.fetchers.finmind import FinMindClient, FinMindFetcher
from stock_research.data.fetchers.tpex import BASE_URL, TpexFetcher
from stock_research.data.fetchers.yahoo import YahooFetcher
from stock_research.data.http import JsonClient, RateLimiter
from stock_research.data.market_data import MarketData, calendar_from_sources
from stock_research.data.windows import WindowName
from stock_research.markets.calendar import CalendarConflictError, CalendarCoverageError
from stock_research.markets.tickers import Market, Ticker
from tests.fakes import RoutingSession, fixture_text

# 15:00 in Taipei on 2026-10-15: that day's close is already published.
NOW = datetime(2026, 10, 15, 7, 0, tzinfo=UTC)
AS_OF = date(2026, 10, 15)
TSMC = Ticker("2330", Market.TWSE)


def data(name: str) -> list[dict]:
    return json.loads(fixture_text(name))["data"]


def base_routes() -> dict:
    return {
        ("TaiwanStockInfo", None): "finmind_stock_info.json",
        ("TaiwanStockPrice", "TAIEX"): "finmind_taiex_recent.json",
        ("TaiwanStockTradingDate", None): "finmind_trading_date_future.json",
        ("TaiwanStockTotalReturnIndex", "TAIEX"): "finmind_total_return_index_recent.json",
        ("TaiwanStockPrice", "2330"): "finmind_price_2330_recent.json",
        ("TaiwanStockDividendResult", "2330"): "finmind_dividend_result_2330_recent.json",
        ("TaiwanStockCapitalReductionReferencePrice", "2330"): [],
        ("TaiwanStockSplitPrice", "2330"): [],
        ("TaiwanStockFinancialStatements", "2330"): "finmind_income_2330_2019.json",
        ("TaiwanStockBalanceSheet", "2330"): "finmind_balance_2330_2019Q4.json",
        ("TaiwanStockCashFlowsStatement", "2330"): "finmind_cashflow_2330_2019.json",
        ("TaiwanStockMonthRevenue", "2330"): "finmind_month_revenue_2330_recent.json",
        ("TaiwanStockDividend", "2330"): "finmind_dividend_2330.json",
        # 0050 stand-in for the ETF comparison: 2330's rows, scaled.
        ("TaiwanStockPrice", "0050"): etf_rows(),
        ("TaiwanStockDividendResult", "0050"): [],
        ("TaiwanStockCapitalReductionReferencePrice", "0050"): [],
        ("TaiwanStockSplitPrice", "0050"): [],
    }


def etf_rows() -> list[dict]:
    scale = {"open": 0.1, "max": 0.1, "min": 0.1, "close": 0.1}
    return [
        r | {"stock_id": "0050"} | {k: round(r[k] * v, 2) for k, v in scale.items()}
        for r in data("finmind_price_2330_recent.json")
    ]


def yahoo_like(code: str, routes: dict, bump: dict[date, float] | None = None):
    """A yfinance downloader that echoes FinMind's closes (optionally altered)."""
    source = routes[("TaiwanStockPrice", code)]
    rows = data(source) if isinstance(source, str) else source

    def download(symbol, start, end_exclusive):
        picked = [r for r in rows if start <= date.fromisoformat(r["date"]) < end_exclusive]
        index = pd.DatetimeIndex([r["date"] for r in picked]).tz_localize("Asia/Taipei")
        closes = [r["close"] * (bump or {}).get(date.fromisoformat(r["date"]), 1.0) for r in picked]
        return pd.DataFrame(
            {
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Volume": [r["Trading_Volume"] for r in picked],
                "Stock Splits": 0.0,
            },
            index=index,
        )

    return download


def build_market_data(
    config: AppConfig, routes=None, yahoo=None, tpex_routes=None
) -> tuple[MarketData, RoutingSession]:
    """MarketData over fixture-backed sessions; no request leaves the process."""
    routes = routes or base_routes()
    session = RoutingSession(routes)
    client = FinMindClient(JsonClient("finmind", session, RateLimiter()), timedelta(hours=1))
    finmind = FinMindFetcher(client, config.markets.tw.filing_deadlines)
    yf = YahooFetcher(
        config.markets.tw,
        RateLimiter(),
        max_age=timedelta(hours=1),
        downloader=yahoo or yahoo_like("2330", routes),
    )
    tpex = None
    if tpex_routes is not None:
        tpex = TpexFetcher(
            JsonClient("tpex", RoutingSession(tpex_routes), RateLimiter()), timedelta(hours=1)
        )
    return MarketData(config, finmind, yf, tpex, clock=lambda: NOW), session


@pytest.fixture
def make(raw_config, write_config):
    def _make(routes=None, overrides=(), yahoo=None, tpex_routes=None, tweak=None):
        raw_config["markets"]["tw"]["listing"]["overrides"] = list(overrides)
        if tweak:
            tweak(raw_config)
        config = load_config(write_config(raw_config))
        return build_market_data(config, routes, yahoo, tpex_routes)

    return _make


TSMC_LISTING = {"ticker": "2330", "listing_date": "1994-09-05", "source": "TWSE"}


# --- calendar ---------------------------------------------------------------------------------


def calendar(index_days, scheduled, manual=()):
    return calendar_from_sources(
        index_days,
        scheduled,
        list(manual),
        timezone="Asia/Taipei",
        daily_data_ready_time=time(14, 30),
    )


def test_calendar_history_from_index_and_future_from_schedule() -> None:
    closes = [date.fromisoformat(r["date"]) for r in data("finmind_taiex_recent.json")]
    scheduled = [date.fromisoformat(r["date"]) for r in data("finmind_trading_date_future.json")]
    cal = calendar(closes, scheduled, [ManualClosure(date=date(2026, 11, 2), reason="typhoon")])
    assert cal.last_historical_day == AS_OF
    assert not cal.is_trading_day(date(2026, 10, 9))  # National Day, from the index history
    assert not cal.is_trading_day(date(2026, 10, 26))  # scheduled holiday
    assert not cal.is_trading_day(date(2026, 12, 25))
    assert not cal.is_trading_day(date(2026, 11, 2))  # manual typhoon closure
    assert cal.is_trading_day(date(2026, 10, 16))
    # The schedule ends on 2026-12-31: 2027 is not covered and the calendar says so.
    with pytest.raises(CalendarCoverageError):
        cal.is_trading_day(date(2027, 1, 4))


def test_calendar_rejects_schedule_that_disagrees_with_index_history() -> None:
    days = [date(2026, 10, 14), date(2026, 10, 15)]
    with pytest.raises(CalendarConflictError, match="2026-10-14"):
        calendar(days, [date(2026, 10, 15), date(2026, 10, 16)])
    with pytest.raises(CalendarConflictError, match="no index closes"):
        calendar([], [])


def test_calendar_with_incomplete_schedule_year_has_no_future() -> None:
    cal = calendar([date(2026, 10, 15)], [date(2026, 10, 15), date(2026, 10, 16)])
    with pytest.raises(CalendarCoverageError):
        cal.is_trading_day(date(2026, 10, 16))


def test_calendar_extra_weekend_trading_day() -> None:
    scheduled = [date(2026, 10, 15), date(2026, 10, 17)] + [
        date(2026, 12, 31) - timedelta(days=i) for i in range(4)
    ]
    cal = calendar([date(2026, 10, 15)], scheduled)
    assert cal.is_trading_day(date(2026, 10, 17))  # a Saturday make-up day
    assert not cal.is_trading_day(date(2026, 10, 16))


# --- listing and prices -----------------------------------------------------------------------


def test_stock_info_and_resolve(make) -> None:
    md, _ = make()
    info = md.stock_info("6488")
    assert info.ticker == Ticker("6488", Market.TPEX) and "半導體業" in info.industries
    assert md.resolve("2330") == TSMC
    with pytest.raises(MissingDataError, match="not listed"):
        md.resolve("9999")


def test_stock_info_ambiguous_market_raises(make) -> None:
    routes = base_routes()
    routes[("TaiwanStockInfo", None)] = [
        {"stock_id": "1234", "stock_name": "x", "type": t, "industry_category": "a"}
        | {"date": "2026-10-16"}
        for t in ("twse", "tpex")
    ]
    md, _ = make(routes)
    with pytest.raises(AnomalyError, match="exactly one market"):
        md.resolve("1234")


def test_listing_override_wins(make) -> None:
    md, session = make(overrides=[TSMC_LISTING])
    assert md.listing_date(TSMC) == date(1994, 9, 5)
    assert not any(c["params"].get("dataset") == "TaiwanStockPER" for c in session.calls)


def test_listing_before_emerging_board_is_none(make) -> None:
    routes = base_routes()
    routes[("TaiwanStockPrice", "2330")] = "finmind_price_zero_fields.json"  # starts 1996
    md, _ = make(routes)
    assert md.listing_date(TSMC) is None


def _emerging_then_listed_routes() -> dict:
    routes = base_routes()
    # Valuation days of an earlier company that used the code, then the listing day.
    routes[("TaiwanStockPER", "2330")] = [
        {"date": d, "stock_id": "2330", "dividend_yield": 0.0, "PER": 0.0, "PBR": 0.0}
        for d in ("2008-08-15", "2026-08-03", "2026-08-04")
    ]
    return routes


def test_listing_from_first_valuation_day_after_first_trade(make) -> None:
    md, _ = make(_emerging_then_listed_routes())
    assert md.listing_date(TSMC) == date(2026, 8, 3)
    # Emerging-board prices before listing are dropped.
    prices = md.prices(TSMC, AS_OF)
    assert prices.raw["date"].min() == date(2026, 8, 3)


def test_ambiguous_listing_raises(make) -> None:
    routes = base_routes()
    routes[("TaiwanStockPER", "2330")] = [
        {"date": "2005-09-02", "stock_id": "2330", "dividend_yield": 0.0, "PER": 0.0, "PBR": 0.0}
    ]
    routes[("TaiwanStockPrice", "2330")] = [
        r | {"date": "2004-" + r["date"][5:]} for r in data("finmind_price_2330_recent.json")
    ]
    md, _ = make(routes)
    with pytest.raises(MissingDataError, match="listing.overrides"):
        md.listing_date(TSMC)


def test_prices_start_at_price_history_start(make) -> None:
    md, _ = make(
        overrides=[TSMC_LISTING],
        tweak=lambda c: c["data_sources"]["finmind"].update(price_history_start="2026-09-01"),
    )
    prices = md.prices(TSMC, date(2026, 10, 1))
    assert prices.raw["date"].min() == date(2026, 9, 1)
    assert prices.raw["date"].max() == date(2026, 10, 1)
    assert prices.ex_rights_reference() == {date(2026, 9, 16): 2377.99}


# --- point in time ------------------------------------------------------------------------------


def test_financials_respect_as_of_and_ifrs_start(make) -> None:
    routes = base_routes()
    income = data("finmind_income_2330_2019.json")
    routes[("TaiwanStockFinancialStatements", "2330")] = income + [
        r | {"date": "2012-" + r["date"][5:]} for r in income
    ]
    md, _ = make(routes)
    before = md.financials(TSMC, date(2020, 3, 30))
    on = md.financials(TSMC, date(2020, 3, 31))
    assert date(2019, 12, 31) not in set(before["period_end"])
    assert date(2019, 12, 31) in set(on["period_end"])
    assert min(on["period_start"]) >= date(2013, 1, 1)  # pre-IFRS rows dropped
    assert set(on["statement"]) == {"income", "balance", "cash_flow"}


def test_revenue_dividends_and_benchmark_respect_as_of(make) -> None:
    md, _ = make()
    assert md.monthly_revenue(TSMC, date(2026, 10, 7))["period_end"].max() == date(2026, 8, 31)
    assert md.monthly_revenue(TSMC, date(2026, 10, 8))["period_end"].max() == date(2026, 9, 30)
    # Announced 2019-07-09 and 2019-08-14.
    assert len(md.dividends(TSMC, date(2019, 7, 31))) == 2
    assert md.benchmark(date(2026, 9, 30))["date"].max() == date(2026, 9, 30)


# --- health report --------------------------------------------------------------------------------


def test_health_report_passes_on_consistent_data(make) -> None:
    md, _ = make(overrides=[TSMC_LISTING])
    report = md.health_report("2330", AS_OF)
    failed = [c for c in report.checks if not c.passed]
    assert report.passed, failed
    names = [c.name for c in report.checks]
    assert names == [
        "calendar",
        "listing",
        "prices.fetch_and_adjust",
        "prices.completeness",
        "prices.price_limits",
        "prices.freshness",
        "prices.cross_source",
        "benchmark.fetch",
        "benchmark.completeness",
        "financials.fetch",
        "financials.balance_sheet",
        "per_share.share_basis",
        "per_share.share_count",
        "revenue.fetch",
        "revenue.sanity",
    ]
    sources = {s.source: s.latest_data_date for s in report.sources}
    assert sources["finmind:TaiwanStockPrice"] == AS_OF and sources["yahoo"] == AS_OF
    assert any("FinMind" in limitation for limitation in report.limitations)


def test_health_report_records_failures_and_keeps_checking(make) -> None:
    routes = base_routes()
    md, _ = make(
        routes,
        overrides=[TSMC_LISTING],
        yahoo=yahoo_like("2330", routes, bump={date(2026, 10, 14): 1.02}),
    )
    report = md.health_report("2330", AS_OF)
    result = {c.name: c for c in report.checks}
    assert not report.passed
    assert not result["prices.cross_source"].passed
    assert "2026-10-14" in result["prices.cross_source"].detail
    assert result["revenue.sanity"].passed  # later checks still ran


def test_etf_prices_skip_the_listing_check(make) -> None:
    md, _ = make(overrides=[TSMC_LISTING])
    etf = md.etf_prices("0050", AS_OF)
    assert etf.adjusted["date"].max() == AS_OF
    assert etf.raw["close"].iloc[-1] == pytest.approx(244.0)


def test_price_history_override_moves_the_start(make) -> None:
    def tweak(raw):
        raw["markets"]["tw"]["price_history_overrides"] = [
            {"ticker": "2330", "start": "2026-09-01", "reason": "missing capital change"}
        ]

    md, _ = make(overrides=[TSMC_LISTING], tweak=tweak)
    prices = md.prices(TSMC, AS_OF)
    assert prices.raw["date"].min() == date(2026, 9, 1)
    assert prices.adjusted["date"].min() == date(2026, 9, 1)


def test_unexplained_early_halt_moves_the_price_start(make) -> None:
    """A stock-specific halt before FinMind's capital-reduction coverage (moved into the
    fixture's range here) starts the price history on the resumption day, and says so."""
    routes = base_routes()
    rows = data("finmind_price_2330_recent.json")
    halted = {"2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"}
    routes[("TaiwanStockPrice", "2330")] = [r for r in rows if r["date"] not in halted]

    def tweak(raw):
        raw["data_sources"]["finmind"]["capital_reduction_coverage_start"] = "2027-01-01"

    md, _ = make(routes, overrides=[TSMC_LISTING], tweak=tweak)
    assert md.prices(TSMC, AS_OF).raw["date"].min() == date(2026, 9, 7)
    report = md.health_report("2330", AS_OF)
    assert report.passed, [c for c in report.checks if not c.passed]
    assert any("2026-09-07 前的股價未使用" in line for line in report.limitations)


def test_cross_source_skips_every_day_before_a_yahoo_split(make) -> None:
    """Yahoo adjusts closes before a stock dividend; those days must not be compared, and
    older days must not refill the lookback window (9917, 2026-08-31 stock dividend)."""
    routes = base_routes()
    split_day = date(2026, 10, 1)
    plain = yahoo_like("2330", routes)

    def adjusted(symbol, start, end_exclusive):
        frame = plain(symbol, start, end_exclusive)
        before = frame.index.date < split_day
        frame.loc[before, "Close"] = frame.loc[before, "Close"] / 1.05
        frame.loc[frame.index.date == split_day, "Stock Splits"] = 1.05
        return frame

    md, _ = make(routes, overrides=[TSMC_LISTING], yahoo=adjusted)
    result = {c.name: c for c in md.health_report("2330", AS_OF).checks}
    assert result["prices.cross_source"].passed, result["prices.cross_source"].detail


def test_health_report_stale_data(make) -> None:
    routes = base_routes()
    routes[("TaiwanStockPrice", "2330")] = data("finmind_price_2330_recent.json")[:-1]
    md, _ = make(routes, overrides=[TSMC_LISTING])
    result = {c.name: c for c in md.health_report("2330", AS_OF).checks}
    assert not result["prices.freshness"].passed


def test_health_report_stops_without_listing(make) -> None:
    md, _ = make()
    report = md.health_report("9999", AS_OF)
    assert [c.name for c in report.checks] == ["calendar", "listing"]
    assert not report.passed


def test_health_report_capital_change_halt_is_expected(make) -> None:
    routes = base_routes()
    prices = data("finmind_price_2330_recent.json")
    halted = [r for r in prices if not "2026-09-08" <= r["date"] <= "2026-09-10"]
    routes[("TaiwanStockPrice", "2330")] = halted
    before = next(r for r in prices if r["date"] == "2026-09-07")
    after = next(r for r in prices if r["date"] == "2026-09-11")
    routes[("TaiwanStockCapitalReductionReferencePrice", "2330")] = [
        {
            "date": "2026-09-11",
            "stock_id": "2330",
            "ClosingPriceonTheLastTradingDay": before["close"],
            "PostReductionReferencePrice": after["close"],
            "ReasonforCapitalReduction": "Cash refund",
        }
    ]
    md, _ = make(routes, overrides=[TSMC_LISTING])
    report = md.health_report("2330", AS_OF)
    assert {c.name: c.passed for c in report.checks}["prices.completeness"]
    halt = next(i for i in report.issues if i.check == "known_suspension")
    assert halt.dates == (date(2026, 9, 8), date(2026, 9, 9), date(2026, 9, 10))


def _tpex_routes() -> tuple[dict, dict]:
    routes = base_routes()
    for (dataset, data_id), value in list(routes.items()):
        if data_id == "2330":
            routes[(dataset, "6488")] = value
    tpex = {
        f"{BASE_URL}/mopsfin_t187ap03_O": "tpex_listed_companies.json",
        f"{BASE_URL}/tpex_mainboard_daily_close_quotes": "tpex_daily_close_quotes.json",
    }
    return routes, tpex


def test_tpex_listing_checked_against_official(make) -> None:
    routes, tpex = _tpex_routes()
    good = {"ticker": "6488", "listing_date": "2015-09-25", "source": "TPEx"}
    md, _ = make(routes, overrides=[good], yahoo=yahoo_like("6488", routes), tpex_routes=tpex)
    result = {c.name: c for c in md.health_report("6488", AS_OF).checks}
    assert result["listing.official_tpex"].passed
    # The official TPEx close (923.00) disagrees with these closes, so the check fails.
    assert not result["prices.cross_source"].passed
    assert "tpex" in result["prices.cross_source"].detail

    bad = good | {"listing_date": "2015-10-01"}
    md, _ = make(routes, overrides=[bad], yahoo=yahoo_like("6488", routes), tpex_routes=tpex)
    result = {c.name: c for c in md.health_report("6488", AS_OF).checks}
    assert not result["listing.official_tpex"].passed
    assert "2015-09-25" in result["listing.official_tpex"].detail


def test_health_report_benchmark_gap_and_missing_balance_totals(make) -> None:
    routes = base_routes()
    tri = data("finmind_total_return_index_recent.json")
    routes[("TaiwanStockTotalReturnIndex", "TAIEX")] = [r for r in tri if r["date"] != "2026-08-14"]
    balance = data("finmind_balance_2330_2019Q4.json")
    routes[("TaiwanStockBalanceSheet", "2330")] = [r for r in balance if r["type"] != "Equity"]
    md, _ = make(routes, overrides=[TSMC_LISTING])
    result = {c.name: c for c in md.health_report("2330", AS_OF).checks}
    assert "2026-08-14" in result["benchmark.completeness"].detail
    assert not result["financials.balance_sheet"].passed


def test_open_wires_real_clients_without_network(raw_config, write_config, monkeypatch) -> None:
    monkeypatch.setenv("FINMIND_TOKEN", "tok")
    config = load_config(write_config(raw_config))
    with Cache(":memory:") as cache:
        md = MarketData.open(config, Secrets(None), cache)
        assert isinstance(md, MarketData) and md.today() >= date(2026, 1, 1)


# --- industries and peers -------------------------------------------------------------------------


def info_row(code: str, category: str, kind: str = "twse") -> dict:
    return {
        "stock_id": code,
        "stock_name": code,
        "type": kind,
        "industry_category": category,
        "date": "2026-10-16",
    }


def test_industry_ignores_broad_categories_and_merges_spellings(make) -> None:
    routes = base_routes()
    routes[("TaiwanStockInfo", None)] = [
        info_row("2330", "半導體業"),
        info_row("2330", "電子工業"),
        info_row("6488", "半導體業", "tpex"),
        info_row("3105", "其他電子類", "tpex"),
        info_row("2317", "其他電子業"),
        info_row("2317", "電子工業"),
        info_row("0050", "半導體業"),  # an ETF code: never a peer
        info_row("1234", "半導體業"),
        info_row("1234", "光電業"),
    ]
    md, _ = make(routes)
    assert md.industry("2330") == "半導體業"
    assert md.industry("3105") == md.industry("2317") == "其他電子業"
    assert md.industry_groups("半導體業") == frozenset({"semiconductor"})
    assert [t.code for t in md.peers("半導體業", exclude="2330")] == ["1234", "6488"]
    assert [t.code for t in md.peers("其他電子業", exclude="2317")] == ["3105"]
    with pytest.raises(AnomalyError, match="overrides"):
        md.industry("1234")


def test_industry_override(make) -> None:
    md, _ = make(
        tweak=lambda c: c["markets"]["tw"]["industries"]["overrides"].append(
            {"ticker": "2330", "industry": "光電業"}
        )
    )
    assert md.industry("2330") == "光電業"


def test_peer_financials_are_income_and_balance_only(make) -> None:
    md, session = make()
    rows = md.peer_financials(TSMC, date(2020, 3, 31))
    assert set(rows["statement"]) == {"income", "balance"}
    datasets = {c["params"]["dataset"] for c in session.calls}
    assert "TaiwanStockCashFlowsStatement" not in datasets


def test_build_inputs_end_to_end(make) -> None:
    routes = base_routes()
    for dataset in ("TaiwanStockFinancialStatements", "TaiwanStockBalanceSheet"):
        routes[(dataset, "6488")] = routes[(dataset, "2330")]
    md, _ = make(routes, overrides=[TSMC_LISTING])
    config = md._config
    messages: list[str] = []
    inputs = build_inputs(md, config, "2330", AS_OF, progress=messages.append)
    assert set(inputs) == {WindowName.POST_2020, WindowName.FULL, WindowName.PRE_2020}
    assert messages == ["半導體業: loading 1 peer(s)"]
    v3 = inputs[WindowName.PRE_2020]
    assert v3.industry == "半導體業" and v3.groups == frozenset({"semiconductor"})
    # Prices in the fixtures start in 2026, so V3 has no valuation; the reason is kept.
    assert v3.valuation is None and "raw close" in v3.valuation_problem
    assert v3.peers.peers_total == 1 and v3.peers.values["gross_margin"] is None
    assert "CPI" in v3.cpi_problem
    v1 = inputs[WindowName.POST_2020]
    assert v1.valuation is not None and v1.valuation.price == 2440.0
    for window_inputs in inputs.values():
        results = run_all(window_inputs)
        assert composite(results, config.scoring).window is window_inputs.window.name
