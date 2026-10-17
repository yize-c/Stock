from __future__ import annotations

from datetime import date

import pytest

from stock_research.llm.verification import (
    SourceDocument,
    Status,
    chinese_to_number,
    extract_numbers,
    normalize,
    number_matches,
    quote_found,
    verify_claim,
)

DOC = SourceDocument(
    "2020-03-31_年報",
    "年報",
    date(2020, 3, 31),
    "民國一百零八年度合併營收為新台幣10,699億8,500萬元，較前一年度成長3.7%。\n"
    "先進製程（7奈米及以下）占晶圓銷售金額之百分之三十三。",
)
DOCS = {DOC.doc_id: DOC}


def check(claim: str, quote: str, *, doc=DOC.doc_id, day="2020-03-31", computed=()):
    return verify_claim(claim, quote, doc, day, DOCS, computed, 0.01)


def test_normalisation_without_script_conversion() -> None:
    assert normalize("營收　成長 ３．７％。") == "營收成長3.7%."
    assert quote_found("合併營收為新台幣 10,699 億", DOC.text)
    assert quote_found("「先進製程（7奈米及以下）」", "先進製程(7奈米及以下)占")
    # A Simplified character is a changed quote, not a match.
    assert not quote_found("合并营收", DOC.text)
    assert not quote_found("   ", DOC.text)


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("三千五百億", 3.5e11),
        ("十二", 12),
        ("兩萬零五", 20005),
        ("一百零八", 108),
        ("億", None),
    ],
)
def test_chinese_numerals(text, value) -> None:
    if value is None:
        assert extract_numbers(text) == []
    else:
        assert chinese_to_number(text) == value
    with pytest.raises(ValueError):
        chinese_to_number("三A")


def test_number_extraction() -> None:
    found = {n.text: n.value for n in extract_numbers(DOC.text)}
    assert found["10,699億8,500萬"] == pytest.approx(1.06998500e12)
    assert found["3.7%"] == pytest.approx(0.037)
    assert found["百分之三十三"] == pytest.approx(0.33)
    assert "7" not in found or found["7"] == 7  # an ordinary number is kept
    # Dates, years and quarters are not amounts.
    assert extract_numbers("2019年第4季，2020年3月31日公告，Q3") == []
    # A lone 十 is a word unless it is a percentage.
    assert extract_numbers("十分重要") == []
    assert extract_numbers("V1 與 V3 的 Q4") == []
    assert [n.value for n in extract_numbers("百分之十")] == [pytest.approx(0.1)]
    assert [n.value for n in extract_numbers("1.2兆與3億，5萬")] == [1.2e12, 3e8, 5e4]


def test_number_matching_tolerance() -> None:
    assert number_matches(1.07e12, [1.069985e12], 0.01)
    assert not number_matches(1.2e12, [1.069985e12], 0.01)
    assert number_matches(0.0, [0.0], 0.01)


def test_grounded_claim_is_verified() -> None:
    result = check("108年合併營收約10,700億元，成長3.7%", "合併營收為新台幣10,699億8,500萬元")
    assert result.status is Status.VERIFIED and result.problems == ()


def test_fabricated_quote_is_caught() -> None:
    result = check("營收創新高", "營收創歷史新高並大幅成長")
    assert result.status is Status.UNVERIFIED
    assert "quote not found in the cited document" in result.problems


def test_fabricated_numbers_are_caught_including_chinese_units() -> None:
    result = check("合併營收達一兆兩千億元，成長百分之十", "合併營收為新台幣10,699億8,500萬元")
    assert result.status is Status.UNVERIFIED
    assert len(result.problems) == 2 and all("matches no" in p for p in result.problems)


def test_computed_values_back_numbers() -> None:
    claim = "ROE 為 21.2%"
    assert check(claim, "合併營收為新台幣").status is Status.UNVERIFIED
    assert check(claim, "合併營收為新台幣", computed=[0.2123]).status is Status.VERIFIED


def test_wrong_source_is_caught() -> None:
    unknown = check("營收成長", "成長3.7%", doc="2021-03-31_年報")
    assert unknown.problems == ("cites unknown document '2021-03-31_年報'",)
    wrong_day = check("營收成長3.7%", "成長3.7%", day="2020-04-01")
    assert wrong_day.status is Status.UNVERIFIED and "source_date" in wrong_day.problems[0]
