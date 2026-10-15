from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stock_research.config import load_config
from stock_research.data.errors import LookAheadError, MissingDataError
from stock_research.data.point_in_time import (
    as_of_view,
    assert_no_lookahead,
    fill_available_dates,
    restatement_limitations,
    statutory_available_date,
)
from tests.conftest import DEFAULT_CONFIG

DEADLINES = load_config(DEFAULT_CONFIG).markets.tw.filing_deadlines


@pytest.mark.parametrize(
    ("period_end", "ptype", "expected"),
    [
        (date(2024, 3, 31), "quarter", date(2024, 5, 15)),
        (date(2024, 6, 30), "quarter", date(2024, 8, 14)),
        (date(2024, 9, 30), "quarter", date(2024, 11, 14)),
        (date(2024, 12, 31), "quarter", date(2025, 3, 31)),
        (date(2024, 12, 31), "annual", date(2025, 3, 31)),
        (date(2024, 1, 31), "month", date(2024, 2, 10)),
        (date(2024, 12, 31), "month", date(2025, 1, 10)),
    ],
)
def test_statutory_available_date(period_end, ptype, expected) -> None:
    assert statutory_available_date(period_end, ptype, DEADLINES) == expected


def test_statutory_rejects_non_quarter_end() -> None:
    with pytest.raises(ValueError):
        statutory_available_date(date(2024, 5, 31), "quarter", DEADLINES)


def test_fill_available_dates_marks_estimates() -> None:
    df = pd.DataFrame(
        {
            "period_type": ["quarter", "quarter"],
            "period_end": [date(2024, 3, 31), date(2024, 6, 30)],
            "available_date": [pd.Timestamp("2024-05-10"), None],
        }
    )
    out = fill_available_dates(df, DEADLINES)
    assert list(out["available_date"]) == [pd.Timestamp("2024-05-10"), pd.Timestamp("2024-08-14")]
    assert list(out["available_date_estimated"]) == [False, True]
    assert df["available_date"].isna().sum() == 1  # input untouched


def test_fill_keeps_existing_estimated_flags() -> None:
    df = pd.DataFrame(
        {
            "period_type": ["quarter"],
            "period_end": [date(2024, 3, 31)],
            "available_date": [pd.Timestamp("2024-05-15")],
            "available_date_estimated": [True],
        }
    )
    assert bool(fill_available_dates(df, DEADLINES)["available_date_estimated"].iloc[0])


def _versions() -> pd.DataFrame:
    base = {
        "ticker": "2330",
        "item": "eps",
        "period_type": "annual",
        "period_start": date(2018, 1, 1),
        "period_end": date(2018, 12, 31),
    }
    return pd.DataFrame(
        [
            {
                **base,
                "value": 13.54,
                "available_date": date(2019, 3, 30),
                "version_kind": "original",
            },
            {
                **base,
                "value": 13.30,
                "available_date": date(2020, 3, 31),
                "version_kind": "restated",
            },
        ]
    )


def test_as_of_view_uses_version_published_at_the_time() -> None:
    df = _versions()
    assert as_of_view(df, date(2019, 12, 31))["value"].tolist() == [13.54]
    assert as_of_view(df, date(2021, 1, 1))["value"].tolist() == [13.30]
    assert as_of_view(df, date(2019, 1, 1)).empty


def test_assert_no_lookahead() -> None:
    df = _versions()
    assert_no_lookahead(df.iloc[:1], date(2019, 12, 31))
    with pytest.raises(LookAheadError) as exc:
        assert_no_lookahead(df, date(2019, 12, 31), "2330")
    assert exc.value.actual == ["2020-03-31"]


def test_missing_available_date_raises() -> None:
    df = _versions()
    df.loc[0, "available_date"] = None
    with pytest.raises(MissingDataError):
        assert_no_lookahead(df, date(2025, 1, 1))
    with pytest.raises(MissingDataError):
        as_of_view(df, date(2025, 1, 1))


def test_restatement_limitations() -> None:
    assert restatement_limitations(_versions()) == []
    latest = _versions().iloc[1:].assign(version_kind="latest_only", source="finmind")
    notes = restatement_limitations(latest)
    assert len(notes) == 1 and "finmind" in notes[0] and "eps" in notes[0]
    assert restatement_limitations(pd.DataFrame()) == []
