"""Normalizers for each FinMind dataset, run on real (trimmed) responses."""

from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from stock_research.config import load_config
from stock_research.data.cache import Cache
from stock_research.data.errors import AnomalyError, MissingDataError
from stock_research.data.fetchers.finmind import (
    EVENT_COLUMNS,
    EventKind,
    FinMindClient,
    FinMindFetcher,
    Statement,
    normalize_capital_reductions,
    normalize_dividend_results,
    normalize_dividends,
    normalize_index_closes,
    normalize_monthly_revenue,
    normalize_prices,
    normalize_splits,
    normalize_statement,
    normalize_stock_info,
    normalize_total_return_index,
    normalize_trading_dates,
    parse_fiscal_period,
)
from stock_research.data.http import JsonClient, RateLimiter
from stock_research.data.point_in_time import PeriodType
from tests.conftest import DEFAULT_CONFIG
from tests.fakes import RoutingSession, fixture_text

DEADLINES = load_config(DEFAULT_CONFIG).markets.tw.filing_deadlines


def rows(name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads(fixture_text(name))["data"])


def test_stock_info_markets_industries_and_scope() -> None:
    info = normalize_stock_info(rows("finmind_stock_info.json")).set_index("code")
    assert info.loc["2330", "market"] == "TWSE"
    assert info.loc["2330", "industries"] == ("半導體業", "電子工業")
    assert info.loc["6488", "market"] == "TPEX"
    # Stale rows from an earlier listing do not count: 5450 has an old row too.
    assert info.loc["5450", "data_date"] == date(2026, 10, 16)
    # Index rows (date "None") and emerging-market securities are out of scope.
    assert "TAIEX" not in info.index and not (info["listed_types"] == ("emerging",)).any()


def test_stock_info_ambiguous_market_is_none() -> None:
    raw = pd.DataFrame(
        [
            {"stock_id": "1234", "stock_name": "x", "type": t, "industry_category": "a"}
            | {"date": "2026-10-16"}
            for t in ("twse", "tpex")
        ]
    )
    assert normalize_stock_info(raw).loc[0, "market"] is None


def test_prices_raw_columns() -> None:
    df = normalize_prices(rows("finmind_price_2330_2019-06.json"), "2330")
    assert list(df.columns) == [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "no_trade",
        "source",
    ]
    first = df.iloc[0]
    assert first["date"] == date(2019, 6, 17) and first["close"] == 233.0
    assert not df["no_trade"].any()


def test_prices_zero_means_absent_never_a_price() -> None:
    df = normalize_prices(rows("finmind_price_no_trade.json"), "1258").set_index("date")
    # 2011-09-05: no trades; FinMind leaves open=40.3 and zeros elsewhere.
    no_trade = df.loc[date(2011, 9, 5)]
    assert no_trade["no_trade"] and no_trade[["open", "high", "low", "close"]].isna().all()
    df = normalize_prices(rows("finmind_price_zero_fields.json"), "5302").set_index("date")
    # 1996-02-24: traded, but open is 0 -> NaN (validation reports it), others kept.
    traded = df.loc[date(1996, 2, 24)]
    assert not traded["no_trade"] and pd.isna(traded["open"]) and traded["close"] == 12.2


def test_index_series_and_trading_dates() -> None:
    taiex = normalize_index_closes(rows("finmind_taiex_2024-07.json"), "TAIEX")
    days = set(taiex["date"])
    # Typhoon Gaemi closed the market on 2024-07-24 and 07-25.
    assert date(2024, 7, 23) in days and date(2024, 7, 24) not in days
    assert date(2024, 7, 25) not in days and date(2024, 7, 26) in days
    tri = normalize_total_return_index(rows("finmind_total_return_index.json"))
    assert tri.loc[0, "total_return_index"] == 50159.02
    scheduled = normalize_trading_dates(rows("finmind_trading_date_future.json"))
    assert date(2026, 12, 25) not in scheduled and date(2026, 12, 24) in scheduled


@pytest.mark.parametrize("bad", [0.0, None])
def test_index_series_reject_non_positive(bad) -> None:
    raw = pd.DataFrame([{"date": "2024-07-01", "close": bad, "price": bad}])
    with pytest.raises(AnomalyError):
        normalize_index_closes(raw, "TAIEX")
    with pytest.raises(AnomalyError):
        normalize_total_return_index(raw)


def test_bad_date_raises() -> None:
    with pytest.raises(AnomalyError, match="ISO date"):
        normalize_trading_dates(pd.DataFrame([{"date": "2024/07/01"}]))


def test_missing_columns_raise() -> None:
    with pytest.raises(MissingDataError, match="columns"):
        normalize_prices(pd.DataFrame([{"date": "2024-07-01"}]), "2330")


def test_price_events() -> None:
    ex = normalize_dividend_results(rows("finmind_dividend_result_2330.json"), "2330")
    assert ex.loc[0].to_dict() | {} == {
        "date": date(2019, 6, 24),
        "kind": EventKind.EX_RIGHTS_DIVIDEND.value,
        "close_before": 248.5,
        "reference_price": 240.5,
        "detail": "息",
        "source": "finmind",
    }
    cr = normalize_capital_reductions(rows("finmind_capital_reduction_2349.json"), "2349")
    assert cr.loc[0, "close_before"] == 4.1 and cr.loc[0, "reference_price"] == 6.06
    sp = normalize_splits(rows("finmind_split_6415.json"), "6415")
    assert sp.loc[0, "reference_price"] / sp.loc[0, "close_before"] == 0.25
    for normalize in (normalize_dividend_results, normalize_capital_reductions, normalize_splits):
        assert list(normalize(pd.DataFrame(), "2330").columns) == EVENT_COLUMNS


def test_price_event_with_bad_price_raises() -> None:
    raw = rows("finmind_split_6415.json").assign(after_price=-1.0)
    with pytest.raises(AnomalyError, match="positive"):
        normalize_splits(raw, "6415")


def test_income_is_single_quarter_with_statutory_dates() -> None:
    df = normalize_statement(
        rows("finmind_income_2330_2019.json"), "2330", Statement.INCOME, DEADLINES
    )
    eps = df.loc[df["item"] == "EPS"].set_index("period_end")
    assert eps["value"].sum() == pytest.approx(13.32)  # 2019 annual EPS
    q4 = eps.loc[date(2019, 12, 31)]
    assert q4["period_start"] == date(2019, 10, 1) and q4["period_type"] == "quarter"
    assert q4["available_date"] == date(2020, 3, 31)
    assert eps.loc[date(2019, 3, 31), "available_date"] == date(2019, 5, 15)
    assert df["available_date_estimated"].all() and set(df["version_kind"]) == {"latest_only"}


def test_balance_sheet_is_instant_without_percent_rows() -> None:
    df = normalize_statement(
        rows("finmind_balance_2330_2019Q4.json"), "2330", Statement.BALANCE, DEADLINES
    )
    assert not df["item"].str.endswith("_per").any()
    total = df.loc[df["item"] == "TotalAssets"].iloc[0]
    assert total["period_start"] == total["period_end"] == date(2019, 12, 31)
    assert total["period_type"] == PeriodType.INSTANT.value


def test_cash_flow_is_year_to_date() -> None:
    df = normalize_statement(
        rows("finmind_cashflow_2330_2019.json"), "2330", Statement.CASH_FLOW, DEADLINES
    )
    ocf = df.loc[df["item"] == "CashFlowsFromOperatingActivities"].set_index("period_end")
    assert (ocf["period_start"] == date(2019, 1, 1)).all()
    assert ocf.loc[date(2019, 6, 30), "period_type"] == "year_to_date"
    assert ocf.loc[date(2019, 12, 31), "period_type"] == "annual"
    assert ocf["value"].is_monotonic_increasing  # cumulative


def test_statement_rejects_duplicates_and_non_quarter_ends() -> None:
    raw = rows("finmind_income_2330_2019.json")
    with pytest.raises(AnomalyError, match="one value per item"):
        normalize_statement(pd.concat([raw, raw.head(1)]), "2330", Statement.INCOME, DEADLINES)
    with pytest.raises(AnomalyError, match="quarter end"):
        normalize_statement(raw.assign(date="2019-04-30"), "2330", Statement.CASH_FLOW, DEADLINES)


def test_monthly_revenue_periods_and_available_dates() -> None:
    old = normalize_monthly_revenue(
        rows("finmind_month_revenue_2330_2019.json"), "2330", DEADLINES
    ).set_index("period_end")
    dec = old.loc[date(2018, 12, 31)]
    assert dec["period_start"] == date(2018, 12, 1) and dec["value"] == 89830598000
    assert dec["available_date"] == date(2019, 1, 10) and dec["available_date_basis"] == "statutory"

    recent = normalize_monthly_revenue(
        rows("finmind_month_revenue_2330_recent.json"), "2330", DEADLINES
    ).set_index("period_end")
    # Stored 2026-04-21, after the 2026-03-10 deadline: the deadline is the tighter bound.
    assert recent.loc[date(2026, 2, 28), "available_date"] == date(2026, 3, 10)
    # Stored 2026-10-08, before the 2026-10-10 deadline.
    sep = recent.loc[date(2026, 9, 30)]
    assert sep["available_date"] == date(2026, 10, 8)
    assert sep["available_date_basis"] == "source_create_time"
    assert recent["available_date_estimated"].all()


def test_monthly_revenue_inconsistent_rows_raise() -> None:
    raw = rows("finmind_month_revenue_2330_2019.json")
    with pytest.raises(AnomalyError, match="first day after"):
        normalize_monthly_revenue(raw.assign(date="2019-03-01"), "2330", DEADLINES)
    with pytest.raises(AnomalyError, match="one row per month"):
        normalize_monthly_revenue(pd.concat([raw, raw.head(1)]), "2330", DEADLINES)
    with pytest.raises(AnomalyError, match="create_time"):
        normalize_monthly_revenue(raw.assign(create_time="2018-01-01"), "2330", DEADLINES)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("107年", (date(2018, 1, 1), date(2018, 12, 31), PeriodType.ANNUAL)),
        ("108年第3季", (date(2019, 7, 1), date(2019, 9, 30), PeriodType.QUARTER)),
        ("112年前半年度", (date(2023, 1, 1), date(2023, 6, 30), PeriodType.YEAR_TO_DATE)),
        ("112年後半年度", (date(2023, 7, 1), date(2023, 12, 31), PeriodType.YEAR_TO_DATE)),
    ],
)
def test_parse_fiscal_period(label, expected) -> None:
    assert parse_fiscal_period(label) == expected


def test_parse_fiscal_period_rejects_unknown() -> None:
    with pytest.raises(AnomalyError):
        parse_fiscal_period("2019")


def test_dividends() -> None:
    tsmc = normalize_dividends(rows("finmind_dividend_2330.json"), "2330")
    fy2018 = tsmc.loc[tsmc["fiscal_label"] == "107年"].iloc[0]
    assert fy2018["cash_dividend"] == 8.0 and fy2018["stock_dividend"] == 0.0
    assert fy2018["available_date"] == date(2019, 6, 6)
    assert fy2018["cash_ex_date"] == date(2019, 6, 24) and fy2018["stock_ex_date"] is None
    esun = normalize_dividends(rows("finmind_dividend_2884.json"), "2884")
    assert esun.iloc[0]["stock_dividend"] == pytest.approx(0.10015129)
    halves = normalize_dividends(rows("finmind_dividend_8070.json"), "8070")
    # Cash from earnings plus cash from capital surplus.
    assert halves["period_end"].tolist() == [date(2025, 6, 30), date(2025, 12, 31)]
    assert halves.iloc[1]["cash_dividend"] == pytest.approx(0.6825488 + 1.31745119)
    assert normalize_dividends(pd.DataFrame(), "2330").empty


def test_dividend_without_announcement_raises() -> None:
    raw = rows("finmind_dividend_2330.json").assign(AnnouncementDate="")
    with pytest.raises(MissingDataError, match="AnnouncementDate"):
        normalize_dividends(raw, "2330")


def fetcher(routes, snapshots=None) -> tuple[FinMindFetcher, RoutingSession]:
    session = RoutingSession(routes)
    client = FinMindClient(JsonClient("finmind", session, RateLimiter()), timedelta(hours=1))
    return FinMindFetcher(client, DEADLINES, snapshots), session


def test_fetcher_price_events_combines_sources() -> None:
    f, _ = fetcher(
        {
            ("TaiwanStockDividendResult", "2349"): [],
            ("TaiwanStockCapitalReductionReferencePrice", "2349"): (
                "finmind_capital_reduction_2349.json"
            ),
            ("TaiwanStockSplitPrice", "2349"): "finmind_split_6415.json",
        }
    )
    events = f.price_events("2349", date(2003, 1, 1), date(2026, 10, 15))
    assert events["kind"].tolist() == [
        "capital_reduction",
        "capital_reduction",
        "capital_reduction",
        "split",
    ]
    assert events["date"].is_monotonic_increasing


def test_fetcher_price_events_none() -> None:
    f, _ = fetcher(
        {
            (ds, "2330"): []
            for ds in (
                "TaiwanStockDividendResult",
                "TaiwanStockCapitalReductionReferencePrice",
                "TaiwanStockSplitPrice",
            )
        }
    )
    assert f.price_events("2330", date(2003, 1, 1), date(2026, 10, 15)).empty


def test_statements_and_revenue_are_snapshotted(tmp_path) -> None:
    routes = {
        ("TaiwanStockFinancialStatements", "2330"): "finmind_income_2330_2019.json",
        ("TaiwanStockMonthRevenue", "2330"): "finmind_month_revenue_2330_2019.json",
    }
    with Cache(tmp_path / "c.sqlite") as cache:
        f, _ = fetcher(routes, cache)
        f.statement("2330", Statement.INCOME, date(2019, 1, 1), date(2019, 12, 31))
        f.monthly_revenue("2330", date(2019, 1, 1), date(2019, 12, 31))
        q4 = cache.get_snapshots("finmind", "TaiwanStockFinancialStatements", "2330/2019-12-31")
        assert len(q4) == 1 and "EPS" in q4[0].payload and "source" not in q4[0].payload
        assert cache.get_snapshots("finmind", "TaiwanStockMonthRevenue", "2330/2019-02-01")
        # Refetching identical data adds no new versions.
        f2, _ = fetcher(routes, cache)
        f2.statement("2330", Statement.INCOME, date(2019, 1, 1), date(2019, 12, 31))
        assert (
            len(cache.get_snapshots("finmind", "TaiwanStockFinancialStatements", "2330/2019-12-31"))
            == 1
        )


def test_fetcher_simple_datasets() -> None:
    f, session = fetcher(
        {
            ("TaiwanStockInfo", None): "finmind_stock_info.json",
            ("TaiwanStockTradingDate", None): "finmind_trading_date_future.json",
            ("TaiwanStockPrice", "TAIEX"): "finmind_taiex_recent.json",
            ("TaiwanStockPrice", "2330"): "finmind_price_2330_recent.json",
            ("TaiwanStockTotalReturnIndex", "TAIEX"): "finmind_total_return_index_recent.json",
            ("TaiwanStockPER", "6415"): "finmind_per_6415.json",
            ("TaiwanStockDividend", "2330"): "finmind_dividend_2330.json",
        }
    )
    start, end = date(2026, 7, 1), date(2026, 10, 15)
    assert "2330" in set(f.stock_info()["code"])
    assert date(2026, 10, 9) not in f.trading_dates(start, date(2027, 12, 31))
    assert f.taiex_closes(start, end)["date"].max() == end
    assert f.prices("2330", start, end)["date"].max() == end
    assert f.total_return_index(start, end)["date"].min() == start
    assert f.valuation_dates("6415", date(2008, 1, 1), end)[0] == date(2008, 8, 15)
    assert len(f.dividends("2330", date(2019, 1, 1), end)) == 3
    assert session.calls[0]["params"] == {"dataset": "TaiwanStockInfo"}
