"""OpenAI-compatible chat client (vLLM / OpenAI / 千问AI / local gateways).

**Unstable** - configure via ``fallback.llm`` / env vars. Built-in client uses
stdlib ``urllib`` (no ``[llm]`` extra required). See ``API.md``.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
_OBJECT = re.compile(r"\{.*\}", re.S)


class LLMClientError(RuntimeError):
    """HTTP or parse failure talking to an OpenAI-compatible endpoint."""


def env_llm_config() -> dict[str, str | None]:
    base = (
        os.environ.get("LAYA_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("VLLM_BASE_URL")
        or os.environ.get("DASHSCOPE_BASE_URL")
    )
    key = (
        os.environ.get("LAYA_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
    )
    model = (
        os.environ.get("LAYA_LLM_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("VLLM_MODEL")
        or "gpt-4o-mini"
    )
    return {"base_url": base, "api_key": key, "model": model}


def llm_is_configured(base_url: str | None = None, api_key: str | None = None) -> bool:
    env = env_llm_config()
    url = base_url or env["base_url"]
    key = api_key or env["api_key"]
    if url:
        return True
    if key:
        return True
    return False


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMClientError("empty LLM response")
    m = _FENCE.search(text)
    blob = m.group(1) if m else None
    if blob is None:
        m2 = _OBJECT.search(text)
        blob = m2.group(0) if m2 else None
    if not blob:
        raise LLMClientError(f"no JSON object in LLM response: {text[:200]}")
    try:
        data = json.loads(blob)
    except json.JSONDecodeError as e:
        raise LLMClientError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise LLMClientError("JSON root is not an object")
    return data


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        *,
        enable_thinking: bool | None = False,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        env = env_llm_config()
        raw_url = (base_url or env["base_url"] or "https://api.openai.com/v1").rstrip("/")
        if raw_url.endswith("/chat/completions"):
            self.endpoint = raw_url
        else:
            self.endpoint = raw_url + "/chat/completions"
        self.api_key = api_key or env["api_key"]
        self.model = model or env["model"] or "gpt-4o-mini"
        self.timeout_s = timeout_s
        self.enable_thinking = enable_thinking
        self.extra_body = dict(extra_body or {})

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        max_tokens: int = 256,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if self.enable_thinking is not None:
            body["enable_thinking"] = self.enable_thinking
        body.update(self.extra_body)

        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.endpoint, data=data, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:400]
            raise LLMClientError(f"HTTP {e.code}: {detail}") from e
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise LLMClientError(str(e)) from e

        try:
            msg = payload["choices"][0]["message"]
            content = msg.get("content") or ""
            if not str(content).strip():
                # Some thinking models put text only in reasoning_content
                content = msg.get("reasoning_content") or ""
            return str(content)
        except (KeyError, IndexError, TypeError) as e:
            raise LLMClientError(f"unexpected chat payload: {payload!r}"[:400]) from e

    def chat_json(self, system: str, user: str) -> dict[str, Any]:
        # Prefer structured JSON; fall back to free-form if the gateway rejects
        # response_format (common on some DeepSeek-V4 / thinking endpoints).
        try:
            text = self.chat(
                system,
                user,
                response_format={"type": "json_object"},
                max_tokens=320,
            )
            return parse_json_object(text)
        except Exception as first:
            try:
                text = self.chat(system, user, max_tokens=320)
                return parse_json_object(text)
            except Exception as second:
                raise LLMClientError(
                    f"chat_json failed: {first}; retry: {second}"
                ) from second
