from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from stock_research.data.errors import AnomalyError, MissingDataError, SourceMismatchError
from stock_research.data.fetchers.finmind import (
    normalize_capital_reductions,
    normalize_dividend_results,
    normalize_dividends,
    normalize_splits,
)
from stock_research.data.shares import check_share_count, share_basis, share_counts
from tests.fakes import fixture_text

START = date(2013, 1, 1)


def rows(name: str) -> pd.DataFrame:
    return pd.DataFrame(json.loads(fixture_text(name))["data"])


def applied(events: pd.DataFrame, shift: dict[date, date] | None = None) -> pd.DataFrame:
    """Events as ``adjust_prices`` returns them (effective day = scheduled day unless shifted)."""
    shift = shift or {}
    return events.assign(effective_date=[shift.get(d, d) for d in events["date"]])


def esun(**kwargs):
    events = normalize_dividend_results(rows("finmind_dividend_result_2884_2013.json"), "2884")
    # 2024-07-25 was a typhoon closure: that ex-date took effect on 07-26.
    events = applied(events, {date(2024, 7, 25): date(2024, 7, 26)})
    dividends = normalize_dividends(rows("finmind_dividend_2884.json"), "2884")
    options = {"start": START, "initial_par": 10.0, "tolerance": 0.005} | kwargs
    return share_basis("2884", events, dividends, **options)


def test_stock_dividend_multiplier_matches_declared_dividend() -> None:
    basis = esun()
    by_day = basis.multipliers.set_index("effective_date")["multiplier"]
    # 2019: NT$0.711 stock dividend at NT$10 par -> 7.11% more shares.
    assert by_day[date(2019, 7, 25)] == pytest.approx(1.0711, abs=5e-4)
    assert by_day[date(2024, 7, 26)] == pytest.approx(1.02003, abs=5e-4)
    # Rights offerings and a cash-only year.
    assert by_day[date(2023, 4, 7)] == pytest.approx(1.0)
    assert by_day[date(2026, 7, 1)] == 1.0
    assert (basis.multipliers["par_after"] == 10.0).all()


def test_restating_eps_to_a_later_share_basis() -> None:
    basis = esun()
    eps = pd.Series([1.63])
    restated = basis.to_basis(eps, pd.Series([date(2019, 12, 31)]), date(2026, 10, 15))
    expected = 1.63 / basis.factor(date(2019, 12, 31), date(2026, 10, 15))
    assert restated.iloc[0] == pytest.approx(expected)
    # Stock dividends 2020-2025 (7.97%, 6.12%, 6.78%, 3.80%, 2.00%, 1.01%) enlarge the share count.
    assert restated.iloc[0] == pytest.approx(1.245, abs=0.002)
    # On its own basis a figure is unchanged.
    same = basis.to_basis(eps, pd.Series([date(2019, 12, 31)]), date(2019, 12, 31))
    assert same.iloc[0] == 1.63


def test_dividend_on_ex_date_includes_same_day_stock_dividend() -> None:
    basis = esun()
    ex = date(2019, 7, 25)
    after = basis.factor(ex, date(2019, 12, 31))
    including = basis.factor(ex, date(2019, 12, 31), include_start=True)
    assert after == 1.0 and including == pytest.approx(1.0712, abs=5e-4)


def test_wrong_par_is_caught() -> None:
    with pytest.raises(SourceMismatchError, match="par"):
        esun(initial_par=5.0)


def test_figures_before_basis_start_raise() -> None:
    with pytest.raises(MissingDataError, match="on or after 2013-01-01"):
        esun().factor(date(2012, 12, 31), date(2019, 1, 1))


def test_events_before_start_are_ignored() -> None:
    basis = esun(start=date(2020, 1, 1))
    assert min(basis.multipliers["effective_date"]) >= date(2020, 1, 1)


def test_ex_rights_without_declaration_raises() -> None:
    events = applied(
        normalize_dividend_results(rows("finmind_dividend_result_2884_2013.json"), "2884")
    )
    dividends = normalize_dividends(rows("finmind_dividend_2884.json"), "2884")
    with pytest.raises(MissingDataError, match="no declaration"):
        share_basis(
            "2884",
            events,
            dividends.loc[dividends["stock_ex_date"] != date(2019, 7, 25)],
            start=START,
            initial_par=10.0,
            tolerance=0.005,
        )


@pytest.mark.parametrize(("label", "expected"), [("息", 1.0), ("除息", 1.0)])
def test_cash_only_labels(label, expected) -> None:
    events = pd.DataFrame(
        [
            {
                "date": date(2016, 7, 13),
                "effective_date": date(2016, 7, 13),
                "kind": "ex_rights_dividend",
                "close_before": 72.0,
                "reference_price": 67.0,
                "detail": label,
            }
        ]
    )
    basis = share_basis(
        "6488", events, pd.DataFrame(), start=START, initial_par=10.0, tolerance=0.005
    )
    assert basis.multipliers["multiplier"].tolist() == [expected]
    with pytest.raises(AnomalyError, match="權 or 息"):
        share_basis(
            "6488",
            events.assign(detail="?"),
            pd.DataFrame(),
            start=START,
            initial_par=10.0,
            tolerance=0.005,
        )


def test_split_changes_par() -> None:
    events = applied(normalize_splits(rows("finmind_split_6415.json"), "6415"))
    basis = share_basis(
        "6415", events, pd.DataFrame(), start=START, initial_par=10.0, tolerance=0.005
    )
    assert basis.multipliers["multiplier"].tolist() == [4.0]
    assert basis.par_on(date(2022, 7, 12)) == 10.0 and basis.par_on(date(2022, 7, 13)) == 2.5


@pytest.mark.parametrize(
    ("fixture", "chinese"),
    [
        ("finmind_capital_reduction_2349.json", "彌補虧損"),
        ("finmind_capital_reduction_3481.json", "現金減資"),
    ],
)
def test_capital_reduction_reasons_in_chinese(fixture, chinese) -> None:
    code = fixture[-9:-5]
    events = applied(normalize_capital_reductions(rows(fixture), code))
    english = share_basis(code, events, pd.DataFrame(), start=START, initial_par=10.0, tolerance=0)
    renamed = share_basis(
        code, events.assign(detail=chinese), pd.DataFrame(), start=START, initial_par=10.0,
        tolerance=0,
    )  # fmt: skip
    assert renamed.multipliers["multiplier"].tolist() == english.multipliers["multiplier"].tolist()


def test_capital_reductions() -> None:
    loss = applied(
        normalize_capital_reductions(rows("finmind_capital_reduction_2349.json"), "2349")
    )
    basis = share_basis("2349", loss, pd.DataFrame(), start=START, initial_par=10.0, tolerance=0)
    # Offsetting losses: shares shrink by the price ratio.
    assert basis.multipliers["multiplier"].tolist() == pytest.approx(
        [4.1 / 6.06, 10.75 / 14.79, 5.53 / 10.23]
    )
    refund = applied(
        normalize_capital_reductions(rows("finmind_capital_reduction_3481.json"), "3481")
    )
    basis = share_basis("3481", refund, pd.DataFrame(), start=START, initial_par=10.0, tolerance=0)
    # Cash refund at par: r = (P - par) / (reference - par).
    assert basis.multipliers["multiplier"].tolist() == pytest.approx(
        [0.45 / 0.49, 4.35 / 4.57, 4.55 / 5.17]
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"close_before": 9.0}, "between 0 and 1"),
        ({"reference_price": 10.0}, "!= par"),
        ({"detail": "Other"}, "Making up losses"),
    ],
)
def test_capital_reduction_errors(change, match) -> None:
    refund = applied(
        normalize_capital_reductions(rows("finmind_capital_reduction_3481.json"), "3481")
    ).head(1)
    with pytest.raises(AnomalyError, match=match):
        share_basis(
            "3481",
            refund.assign(**change),
            pd.DataFrame(),
            start=START,
            initial_par=10.0,
            tolerance=0,
        )


def test_unknown_event_kind_raises() -> None:
    events = applied(normalize_splits(rows("finmind_split_6415.json"), "6415")).assign(kind="x")
    with pytest.raises(AnomalyError, match="known price-event kind"):
        share_basis("6415", events, pd.DataFrame(), start=START, initial_par=10.0, tolerance=0)


def test_no_events() -> None:
    basis = share_basis(
        "2330", pd.DataFrame(), pd.DataFrame(), start=START, initial_par=10.0, tolerance=0
    )
    assert basis.factor(START, date(2026, 1, 1)) == 1.0 and basis.par_on(START) == 10.0


def test_share_count_check() -> None:
    days = [date(2019, 3, 31), date(2019, 6, 30), date(2019, 9, 30)]
    capital = pd.Series([100.0, 100.0, 100.0], index=days)
    implied = pd.Series([98.0, 120.0, 101.0], index=days)  # one noisy derived quarter
    assert check_share_count("x", implied, capital, 0.1) == pytest.approx(1.01)
    with pytest.raises(SourceMismatchError, match="par value"):
        check_share_count("x", capital * 4, capital, 0.1)
    with pytest.raises(MissingDataError):
        check_share_count("x", pd.Series([-1.0], index=days[:1]), capital, 0.1)


def test_share_counts_from_statements() -> None:
    financials = pd.DataFrame(
        [
            {"statement": "income", "item": "EPS", "period_end": date(2019, 12, 31), "value": 2.0},
            {
                "statement": "income",
                "item": "EquityAttributableToOwnersOfParent",
                "period_end": date(2019, 12, 31),
                "value": 200.0,
            },
            {
                "statement": "balance",
                "item": "OrdinaryShare",
                "period_end": date(2019, 12, 31),
                "value": 1000.0,
            },
        ]
    )
    basis = share_basis(
        "x", pd.DataFrame(), pd.DataFrame(), start=START, initial_par=10.0, tolerance=0
    )
    implied, capital = share_counts(financials, basis)
    assert implied[date(2019, 12, 31)] == 100.0 and capital[date(2019, 12, 31)] == 100.0


def test_share_counts_without_a_parent_line_use_net_income_less_nci() -> None:
    """73 stocks in the full-market calibration report no parent line (no NCI)."""
    end = date(2019, 12, 31)

    def line(statement, item, value):
        return {"statement": statement, "item": item, "period_end": end, "value": value}

    basis = share_basis(
        "x", pd.DataFrame(), pd.DataFrame(), start=START, initial_par=10.0, tolerance=0
    )
    rows = [
        line("income", "EPS", 2.0),
        line("income", "IncomeAfterTaxes", 200.0),
        line("balance", "OrdinaryShare", 1000.0),
    ]
    implied, _ = share_counts(pd.DataFrame(rows), basis)
    assert implied[end] == 100.0
    with_nci = [*rows, line("income", "NoncontrollingInterests", 20.0)]
    implied, _ = share_counts(pd.DataFrame(with_nci), basis)
    assert implied[end] == 90.0


def test_a_quarter_missing_a_reported_nci_line_gets_no_parent_income() -> None:
    """A company that reports an NCI line in some quarter: a quarter without it (and without
    a parent line) is not assumed to have no non-controlling interests."""
    q1, q2 = date(2019, 3, 31), date(2019, 6, 30)

    def line(statement, item, end, value):
        return {"statement": statement, "item": item, "period_end": end, "value": value}

    basis = share_basis(
        "x", pd.DataFrame(), pd.DataFrame(), start=START, initial_par=10.0, tolerance=0
    )
    rows = [
        line("income", "EPS", q1, 2.0), line("income", "EPS", q2, 2.0),
        line("income", "IncomeAfterTaxes", q1, 200.0),
        line("income", "IncomeAfterTaxes", q2, 200.0),
        line("income", "NoncontrollingInterests", q1, 20.0),
    ]  # fmt: skip
    implied, _ = share_counts(pd.DataFrame(rows), basis)
    assert implied[q1] == 90.0 and pd.isna(implied[q2])
