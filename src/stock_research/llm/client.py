"""Anthropic API client for the LLM layer.

* Structured output: the response is constrained to the pydantic schema's JSON schema
  (``output_config.format``) and validated again with pydantic.
* ``claude-opus-5-5`` rejects sampling parameters, so there is no temperature; thinking is
  always on and ``effort`` (config) sets its depth.
* Refusals: ``fallbacks: "default"`` lets the API re-run a declined request on Anthropic's
  recommended fallback model; a final ``refusal`` stop reason is still a failure.
* Streaming, because inputs can be long documents.
* One retry (``max_retries``) after any failure, then ``LlmFailure``. Every attempt, the
  prompt, the raw response and the outcome are appended to a JSONL log in ``log_dir``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from stock_research.config import LlmConfig, MissingSecretError, Secrets
from stock_research.llm.prompts import SYSTEM

FALLBACK_BETA = "server-side-fallback-2026-07-01"
Schema = TypeVar("Schema", bound=BaseModel)


class LlmFailure(Exception):
    """The call failed after all retries; the reason of the last attempt is kept."""


@dataclass(frozen=True)
class RawResponse:
    text: str
    stop_reason: str | None
    model: str
    usage: dict[str, Any]


# (system, user, schema, config) -> raw response; replaced by a fake in tests.
Transport = Callable[[str, str, dict[str, Any], LlmConfig], RawResponse]


def anthropic_transport(api_key: str | None = None) -> Transport:
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def send(system: str, user: str, schema: dict[str, Any], config: LlmConfig) -> RawResponse:
        with client.beta.messages.stream(
            model=config.model,
            max_tokens=config.max_output,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": config.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            message = stream.get_final_message()
        text = "".join(b.text for b in message.content if b.type == "text")
        return RawResponse(text, message.stop_reason, message.model, message.usage.to_dict())

    return send


def default_transport() -> Transport:
    """Uses ANTHROPIC_API_KEY when set (from .env); otherwise the SDK's own credential chain."""
    secrets = Secrets()
    try:
        key = secrets.get(Secrets.ANTHROPIC_API_KEY)
    except MissingSecretError:
        key = None
    return anthropic_transport(key)


class LlmClient:
    def __init__(self, config: LlmConfig, transport: Transport | None = None) -> None:
        self._config = config
        self._transport = transport or default_transport()

    def request(self, user: str, schema: type[Schema], *, label: str) -> Schema:
        json_schema = schema.model_json_schema()
        last_error = "no attempt made"
        for attempt in range(self._config.max_retries + 1):
            record: dict[str, Any] = {
                "time": datetime.now(UTC).isoformat(),
                "label": label,
                "attempt": attempt + 1,
                "model": self._config.model,
                "effort": self._config.effort,
                "system": SYSTEM,
                "user": user,
            }
            try:
                raw = self._transport(SYSTEM, user, json_schema, self._config)
                record.update(
                    response=raw.text,
                    stop_reason=raw.stop_reason,
                    served_by=raw.model,
                    usage=raw.usage,
                )
                if raw.stop_reason != "end_turn":
                    raise LlmFailure(f"stop_reason {raw.stop_reason!r}")
                parsed = schema.model_validate_json(raw.text)
            except (anthropic.APIError, LlmFailure, ValidationError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                record["outcome"] = f"failed: {last_error}"
                self._log(record)
                continue
            record["outcome"] = "ok"
            self._log(record)
            return parsed
        raise LlmFailure(f"{label}: {last_error}")

    def log_verification(self, label: str, results: list[dict[str, Any]]) -> None:
        self._log({"time": datetime.now(UTC).isoformat(), "label": label, "verification": results})

    def _log(self, record: dict[str, Any]) -> None:
        folder = Path(self._config.log_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
