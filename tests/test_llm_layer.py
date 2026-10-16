from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from stock_research.config import load_config
from stock_research.data.windows import WindowName, build_window
from stock_research.llm import client as client_module
from stock_research.llm.analysis import TaskStatus, explain_metrics, run_document_task
from stock_research.llm.client import LlmClient, LlmFailure, RawResponse, anthropic_transport
from stock_research.llm.documents import (
    check_quality,
    extract_text,
    load_documents,
    select_for_window,
)
from stock_research.llm.prompts import SYSTEM, user_prompt
from stock_research.llm.schemas import DocumentSummary
from stock_research.llm.verification import SourceDocument, Status

AS_OF = date(2026, 10, 15)
BODY_2019 = (
    "本公司民國一百零七年度營收為新台幣10,315億元，毛利率48.3%。主要風險為客戶集中與匯率波動。" * 20
)
BODY_2021 = (
    "本公司民國一百零九年度營收為新台幣13,393億元，先進製程需求強勁，資本支出持續增加。" * 20
)


@pytest.fixture
def llm_config(raw_config, write_config, tmp_path):
    raw_config["llm"]["documents_dir"] = str(tmp_path / "docs")
    raw_config["llm"]["log_dir"] = str(tmp_path / "logs")
    return load_config(write_config(raw_config))


def doc(day: date, body: str) -> SourceDocument:
    return SourceDocument(f"{day}_年報", "年報", day, body)


DOC_2019 = doc(date(2019, 3, 30), BODY_2019)
DOC_2021 = doc(date(2021, 3, 31), BODY_2021)


# --- documents ----------------------------------------------------------------------------------


def test_quality_gates(llm_config) -> None:
    cfg = llm_config.llm
    assert check_quality(BODY_2019, cfg) is None
    assert "characters" in check_quality("太短", cfg)
    assert "CJK" in check_quality("abc " * 400, cfg)
    assert "replacement" in check_quality("營收" * 300 + "�" * 50, cfg)


def test_load_documents_reports_every_rejected_file(llm_config) -> None:
    folder = Path(llm_config.llm.documents_dir) / "2330"
    folder.mkdir(parents=True)
    (folder / "2019-03-30_107年報.txt").write_text(BODY_2019, encoding="utf-8")
    (folder / "2021-03-31_109年報.md").write_text(BODY_2021, encoding="utf-8")
    (folder / "annual.txt").write_text(BODY_2019, encoding="utf-8")
    (folder / "2020-02-30_bad-date.txt").write_text(BODY_2019, encoding="utf-8")
    (folder / "2020-01-01_short.txt").write_text("太短", encoding="utf-8")
    (folder / ".DS_Store").write_text("x")
    docs, problems = load_documents("2330", llm_config.llm)
    assert [(d.doc_id, d.published) for d in docs] == [
        ("2019-03-30_107年報", date(2019, 3, 30)),
        ("2021-03-31_109年報", date(2021, 3, 31)),
    ]
    reasons = {p.path.name: p.reason for p in problems}
    assert "YYYY-MM-DD" in reasons["annual.txt"]
    assert "ValueError" in reasons["2020-02-30_bad-date.txt"]
    assert "characters" in reasons["2020-01-01_short.txt"]
    assert load_documents("9999", llm_config.llm) == ([], [])


def test_pdf_text_is_extracted_page_by_page(monkeypatch, tmp_path) -> None:
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

    class Pdf:
        pages = [Page("第一頁"), Page(None), Page("第三頁")]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import pdfplumber

    monkeypatch.setattr(pdfplumber, "open", lambda path: Pdf())
    assert extract_text(tmp_path / "x.pdf") == "第一頁\n\n第三頁"


def test_v3_reads_only_pre_2020_publications_newest_first(llm_config) -> None:
    v3 = build_window(WindowName.PRE_2020, llm_config, AS_OF)
    v1 = build_window(WindowName.POST_2020, llm_config, AS_OF)
    late_2019 = doc(date(2020, 3, 31), BODY_2019)  # FY2019 report, published in 2020
    older = doc(date(2018, 3, 30), BODY_2019)
    selected, _ = select_for_window([older, DOC_2019, late_2019, DOC_2021], v3, 10**6)
    assert [d.published for d in selected] == [date(2019, 3, 30), date(2018, 3, 30)]
    selected, _ = select_for_window([DOC_2019, late_2019, DOC_2021], v1, 10**6)
    assert [d.published for d in selected] == [date(2021, 3, 31), date(2020, 3, 31)]
    # Over budget: the oldest is dropped first.
    selected, dropped = select_for_window([older, DOC_2019], v3, len(BODY_2019) + 10)
    assert selected == [DOC_2019] and dropped == [older]


def test_prompt_tags_documents_and_computed_values() -> None:
    text = user_prompt("summary", "2330", "pre_2020", [DOC_2019], {"roe": 0.2123})
    assert '<document doc_id="2019-03-30_年報" published="2019-03-30"' in text
    assert "roe: 0.2123" in text
    assert "資料不足" in SYSTEM and "不做股價預測" in SYSTEM
    assert "沒有可用文件" in user_prompt("explain", "2330", "v1", [])


# --- client -------------------------------------------------------------------------------------


def summary_json(**fields) -> str:
    base = {"status": "ok", "highlights": [], "risks": []}
    return json.dumps(base | fields, ensure_ascii=False)


class FakeTransport:
    def __init__(self, *responses: RawResponse | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system, user, schema, config) -> RawResponse:
        self.calls.append((system, user))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def ok(text: str, stop: str = "end_turn") -> RawResponse:
    return RawResponse(text, stop, "claude-opus-5-5", {"input_tokens": 10})


def logged(config) -> list[dict]:
    files = list(Path(config.llm.log_dir).glob("*.jsonl"))
    return [json.loads(line) for f in files for line in f.read_text().splitlines()]


def test_client_parses_and_logs(llm_config) -> None:
    transport = FakeTransport(ok(summary_json()))
    result = LlmClient(llm_config.llm, transport).request("prompt", DocumentSummary, label="t")
    assert result.status == "ok"
    (record,) = logged(llm_config)
    assert record["outcome"] == "ok" and record["user"] == "prompt" and record["system"] == SYSTEM


def test_client_retries_once_then_fails(llm_config) -> None:
    transport = FakeTransport(ok("not json"), ok(summary_json()))
    LlmClient(llm_config.llm, transport).request("p", DocumentSummary, label="t")
    assert len(transport.calls) == 2
    failing = FakeTransport(ok("{}"), ok(summary_json(), stop="refusal"))
    with pytest.raises(LlmFailure, match="refusal"):
        LlmClient(llm_config.llm, failing).request("p", DocumentSummary, label="t")
    outcomes = [r["outcome"] for r in logged(llm_config) if "outcome" in r]
    assert outcomes[-2].startswith("failed: ValidationError")


def test_client_treats_api_errors_as_failures(llm_config) -> None:
    import anthropic
    import httpx2

    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://x"))
    transport = FakeTransport(error, ok(summary_json()))
    assert LlmClient(llm_config.llm, transport).request("p", DocumentSummary, label="t")


def test_sdk_request_shape(monkeypatch, llm_config) -> None:
    """The real transport sends structured output, effort and default fallbacks; no
    temperature (this model rejects sampling parameters)."""
    sent = {}

    class Block:
        type = "text"
        text = summary_json()

    class Usage:
        def to_dict(self):
            return {"output_tokens": 5}

    class Message:
        content = [Block()]
        stop_reason = "end_turn"
        model = "claude-opus-5-5"
        usage = Usage()

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return Message()

    class FakeAnthropic:
        def __init__(self, api_key=None):
            self.beta = self
            self.messages = self

        def stream(self, **kwargs):
            sent.update(kwargs)
            return Stream()

    monkeypatch.setattr(client_module.anthropic, "Anthropic", FakeAnthropic)
    raw = anthropic_transport("key")("system", "user", {"type": "object"}, llm_config.llm)
    assert raw.text == summary_json() and raw.usage == {"output_tokens": 5}
    assert sent["model"] == "claude-opus-5-5" and "temperature" not in sent
    assert sent["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": {"type": "object"}},
    }
    assert sent["fallbacks"] == "default" and sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}


# --- end to end with fabricated output ----------------------------------------------------------


def claim(text: str, quote: str, source=DOC_2019) -> dict:
    return {
        "claim": text,
        "quote": quote,
        "source_doc": source.doc_id,
        "source_date": source.published.isoformat(),
    }


def test_fabrications_are_caught_and_kept_for_the_appendix(llm_config) -> None:
    v3 = build_window(WindowName.PRE_2020, llm_config, AS_OF)
    output = summary_json(
        highlights=[
            claim("107年營收約10,315億元", "營收為新台幣10,315億元"),
            claim("營收突破一兆兩千億元", "營收為新台幣10,315億元"),  # fabricated number
            claim("獲利創歷史新高", "獲利創歷史新高"),  # fabricated quote
        ],
        risks=[
            claim("客戶集中是主要風險", "主要風險為客戶集中與匯率波動"),
            claim("疫情衝擊", "先進製程需求強勁", source=DOC_2021),  # a document V3 cannot see
        ],
    )
    client = LlmClient(llm_config.llm, FakeTransport(ok(output)))
    result = run_document_task("summary", "2330", v3, [DOC_2019, DOC_2021], client, llm_config.llm)
    assert result.status is TaskStatus.OK and result.documents == (DOC_2019.doc_id,)
    assert [c.claim for c in result.verified] == ["107年營收約10,315億元", "客戶集中是主要風險"]
    assert [c.claim for c in result.sections["risks"]] == ["客戶集中是主要風險"]
    problems = {c.claim: c.problems for c in result.unverified}
    assert any("一兆兩千億" in p for p in problems["營收突破一兆兩千億元"])
    assert "quote not found in the cited document" in problems["獲利創歷史新高"]
    assert any("unknown document" in p for p in problems["疫情衝擊"])
    assert all(c.status is Status.UNVERIFIED for c in result.unverified)
    verification = [r for r in logged(llm_config) if "verification" in r]
    assert len(verification[0]["verification"]) == 5


def test_task_outcomes_without_claims(llm_config) -> None:
    cfg = llm_config.llm
    v3 = build_window(WindowName.PRE_2020, llm_config, AS_OF)
    none = run_document_task("moat", "2330", v3, [DOC_2021], LlmClient(cfg, FakeTransport()), cfg)
    assert none.status is TaskStatus.NO_DOCUMENTS
    insufficient = json.dumps({"status": "insufficient_data", "claims": []})
    client = LlmClient(cfg, FakeTransport(ok(insufficient)))
    result = run_document_task("moat", "2330", v3, [DOC_2019], client, cfg)
    assert result.status is TaskStatus.INSUFFICIENT_DATA
    broken = LlmClient(cfg, FakeTransport(ok("x"), ok("y")))
    failed = run_document_task("inversion", "2330", v3, [DOC_2019], broken, cfg)
    assert failed.status is TaskStatus.FAILED and "ValidationError" in failed.detail
    moat = json.dumps(
        {"status": "ok", "claims": [claim("轉換成本高", "主要風險為客戶集中")]}, ensure_ascii=False
    )
    result = run_document_task(
        "moat", "2330", v3, [DOC_2019], LlmClient(cfg, FakeTransport(ok(moat))), cfg
    )
    assert result.sections["claims"][0].claim == "轉換成本高"


def test_explanations_may_only_use_computed_numbers(llm_config) -> None:
    cfg = llm_config.llm
    output = json.dumps(
        {
            "status": "ok",
            "paragraphs": ["V1 的 ROE 為 31.7%，高於 V3 的 24.5%。", "波動度為 45%。"],
        },
        ensure_ascii=False,
    )
    computed = {"roe_v1": 0.3169, "roe_v3": 0.2447, "volatility_v1": 0.3018}
    result = explain_metrics(
        "2330", "post_2020", computed, LlmClient(cfg, FakeTransport(ok(output))), cfg
    )
    assert result.paragraphs == ("V1 的 ROE 為 31.7%，高於 V3 的 24.5%。",)
    assert result.flagged == (("波動度為 45%。", ("45%",)),)
    assert (
        explain_metrics(
            "2330", "v1", computed, LlmClient(cfg, FakeTransport(ok("x"), ok("y"))), cfg
        ).status
        is TaskStatus.FAILED
    )
    insufficient = json.dumps({"status": "insufficient_data", "paragraphs": []})
    assert (
        explain_metrics(
            "2330", "v1", computed, LlmClient(cfg, FakeTransport(ok(insufficient))), cfg
        ).status
        is TaskStatus.INSUFFICIENT_DATA
    )
