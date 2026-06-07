"""
OpenRouter Model Rotation
Automatically cycles through free models when a rate-limit (429) is hit.
"""

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class ModelRouter:
    """
    Manages OpenRouter API calls with transparent model rotation.

    When the active model returns HTTP 429 (rate-limited), the router
    cycles to the next available free model and retries automatically.
    After exhausting all models it waits briefly and resets the failed
    set so the cycle can repeat.
    """

    # Ordered list of free-tier models on OpenRouter (verified April 2025).
    # Larger/stronger models first; smaller fallbacks at the end.
    FREE_MODELS: List[str] = [
        "openai/gpt-oss-120b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-4-31b-it:free",
        "google/gemma-4-26b-a4b-it:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "google/gemma-3-27b-it:free",
        "openai/gpt-oss-20b:free",
        "google/gemma-3-12b-it:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "google/gemma-3-4b-it:free",
    ]

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        api_key: str,
        preferred_model: Optional[str] = None,
    ) -> None:
        self.api_key = api_key
        self._failed: set[int] = set()
        self._idx = 0
        self._lock = threading.Lock()   # protects _idx and _failed across threads
        self._tls = threading.local()   # per-request state (was_exhausted)
        if preferred_model and preferred_model in self.FREE_MODELS:
            self._idx = self.FREE_MODELS.index(preferred_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_model(self) -> str:
        with self._lock:
            return self.FREE_MODELS[self._idx]

    # was_exhausted is per-thread so concurrent requests don't clobber each other
    @property
    def was_exhausted(self) -> bool:
        return getattr(self._tls, "exhausted", False)

    @was_exhausted.setter
    def was_exhausted(self, value: bool) -> None:
        self._tls.exhausted = value

    def reset_failures(self) -> None:
        """Clear the failed-model set (e.g. after a cooldown period). Caller must hold _lock."""
        self._failed.clear()

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.35,
        max_tokens: int = 1200,
        retries: int = len(FREE_MODELS) + 2,
        usage_accumulator: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send a chat-completion request.  Returns the assistant text.
        Rotates models automatically on 429 responses.

        If ``usage_accumulator`` (a plain dict owned by the caller) is supplied,
        token usage and the answering model are folded into it on success. The
        dict is per-request/caller-owned so this stays thread-safe even though
        the router instance is shared across requests.
        """
        last_err: Optional[Exception] = None

        for _ in range(retries):
            with self._lock:
                model = self.FREE_MODELS[self._idx]
            try:
                resp = requests.post(
                    self.BASE_URL,
                    headers=self._headers(),
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=90,
                )

                if resp.status_code == 429:
                    logger.warning("Rate-limit on %s — rotating model.", model)
                    with self._lock:
                        rotated = self._rotate()
                    if not rotated:
                        logger.info("All models exhausted — waiting 8 s then resetting.")
                        self.was_exhausted = True
                        time.sleep(8)
                        with self._lock:
                            self.reset_failures()
                    continue

                resp.raise_for_status()
                data = resp.json()
                if usage_accumulator is not None:
                    u = data.get("usage") or {}
                    usage_accumulator["prompt_tokens"] = (
                        usage_accumulator.get("prompt_tokens", 0)
                        + int(u.get("prompt_tokens") or 0)
                    )
                    usage_accumulator["completion_tokens"] = (
                        usage_accumulator.get("completion_tokens", 0)
                        + int(u.get("completion_tokens") or 0)
                    )
                    usage_accumulator["total_tokens"] = (
                        usage_accumulator.get("total_tokens", 0)
                        + int(u.get("total_tokens") or 0)
                    )
                    usage_accumulator["calls"] = usage_accumulator.get("calls", 0) + 1
                    usage_accumulator["model"] = model
                return data["choices"][0]["message"]["content"]

            except requests.RequestException as exc:
                last_err = exc
                logger.error("Request error on %s: %s", model, exc)
                with self._lock:
                    rotated = self._rotate()
                if not rotated:
                    time.sleep(3)
                    with self._lock:
                        self.reset_failures()

        raise RuntimeError(
            f"All model-rotation attempts failed. Last error: {last_err}"
        )

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 400,
        usage_accumulator: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Like ``chat`` but parses the response as JSON.
        Strips markdown code fences if present.
        """
        raw = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            usage_accumulator=usage_accumulator,
        )
        raw = raw.strip()
        # Strip ``` or ```json fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) >= 3 else parts[-1].strip()
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/RoboSmartInvestment",
            "X-Title": "RoboSmartInvestment",
        }

    def _rotate(self) -> bool:
        """Advance to the next non-failed model. Returns False when exhausted. Caller must hold _lock."""
        self._failed.add(self._idx)
        for i in range(len(self.FREE_MODELS)):
            if i not in self._failed:
                self._idx = i
                logger.info("Switched to model: %s", self.FREE_MODELS[self._idx])
                return True
        return False
