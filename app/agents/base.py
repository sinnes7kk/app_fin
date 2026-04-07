"""Shared base class for all V3 agents.

Provides: OpenAI structured output, SHA-256 response caching, Langfuse
tracing (optional), and retry logic.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"

_LANGFUSE_AVAILABLE = False
try:
    from langfuse import Langfuse  # type: ignore[import-untyped]

    _LANGFUSE_AVAILABLE = bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )
except ImportError:
    pass


class BaseAgent:
    """Thin wrapper around the OpenAI chat completions API.

    Features:
    - Pydantic structured outputs via ``response_format``
    - Deterministic caching (SHA-256 of serialised input)
    - Optional Langfuse tracing when keys are configured
    - Exponential-backoff retry (3 attempts)
    """

    def __init__(
        self,
        *,
        name: str,
        model: str = "gpt-4o-mini",
        system_prompt_path: Path | str,
        schema: type[T],
        cache_dir: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.name = name
        self.model = model
        self.schema = schema
        self.temperature = temperature

        prompt_path = Path(system_prompt_path)
        if not prompt_path.is_absolute():
            prompt_path = Path(__file__).resolve().parent / prompt_path
        self._system_prompt = prompt_path.read_text().strip()

        cache_name = cache_dir or name
        self._cache_dir = _DATA_ROOT / "agent_cache" / cache_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._shadow_dir = _DATA_ROOT / "agent_shadow" / name
        self._shadow_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self._langfuse = None

    def _get_client(self):
        """Lazy-init the OpenAI client so import doesn't fail without a key."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def _get_langfuse(self):
        if self._langfuse is None and _LANGFUSE_AVAILABLE:
            self._langfuse = Langfuse()
        return self._langfuse

    @staticmethod
    def _hash_input(input_dict: dict) -> str:
        raw = json.dumps(input_dict, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_path(self, h: str) -> Path:
        return self._cache_dir / f"{h}.json"

    def _read_cache(self, h: str) -> dict | None:
        p = self._cache_path(h)
        if p.exists():
            return json.loads(p.read_text())
        return None

    def _write_cache(self, h: str, input_dict: dict, output_dict: dict) -> None:
        payload = {"input": input_dict, "output": output_dict}
        self._cache_path(h).write_text(json.dumps(payload, default=str))

    def call(self, input_dict: dict) -> T:
        """Synchronous call: check cache -> call OpenAI -> parse -> cache."""
        h = self._hash_input(input_dict)
        cached = self._read_cache(h)
        if cached is not None:
            return self.schema.model_validate(cached["output"])

        user_content = json.dumps(input_dict, indent=2, default=str)
        result = self._call_with_retry(user_content, h, input_dict)
        return result

    def _call_with_retry(
        self, user_content: str, cache_hash: str, input_dict: dict,
        max_retries: int = 3,
    ) -> T:
        client = self._get_client()
        last_err: Exception | None = None

        for attempt in range(max_retries):
            t0 = time.monotonic()
            try:
                response = client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": self._system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    response_format=self.schema,
                )
                latency_ms = int((time.monotonic() - t0) * 1000)
                parsed = response.choices[0].message.parsed
                output_dict = parsed.model_dump()

                self._write_cache(cache_hash, input_dict, output_dict)
                self._trace(input_dict, output_dict, latency_ms, response.usage)

                return parsed

            except Exception as e:
                last_err = e
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    time.sleep(wait)

        raise RuntimeError(
            f"Agent {self.name} failed after {max_retries} attempts: {last_err}"
        ) from last_err

    def _trace(
        self, input_dict: dict, output_dict: dict, latency_ms: int, usage,
    ) -> None:
        lf = self._get_langfuse()
        if lf is None:
            return
        try:
            trace = lf.trace(name=f"agent_{self.name}", input=input_dict, output=output_dict)
            trace.generation(
                name=f"{self.name}_call",
                model=self.model,
                input=input_dict,
                output=output_dict,
                usage={
                    "input": getattr(usage, "prompt_tokens", 0),
                    "output": getattr(usage, "completion_tokens", 0),
                },
                metadata={"latency_ms": latency_ms},
            )
        except Exception:
            pass

    def log_shadow(self, ticker: str, direction: str, output: dict) -> None:
        """Write agent output to the shadow log directory for offline analysis."""
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self._shadow_dir / f"{ticker}_{direction}_{ts}.json"
        payload = {
            "ticker": ticker,
            "direction": direction,
            "agent": self.name,
            "timestamp": ts,
            "output": output,
        }
        path.write_text(json.dumps(payload, default=str, indent=2))
