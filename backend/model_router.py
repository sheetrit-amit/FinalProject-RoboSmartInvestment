"""
OpenRouter Model Rotation
Automatically cycles through free models when a rate-limit (429) is hit.
"""

import json
import logging
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
        if preferred_model and preferred_model in self.FREE_MODELS:
            self._idx = self.FREE_MODELS.index(preferred_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def current_model(self) -> str:
        return self.FREE_MODELS[self._idx]

    def reset_failures(self) -> None:
        """Clear the failed-model set (e.g. after a cooldown period)."""
        self._failed.clear()

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.35,
        max_tokens: int = 1200,
        retries: int = len(FREE_MODELS) + 2,
    ) -> str:
        """
        Send a chat-completion request.  Returns the assistant text.
        Rotates models automatically on 429 responses.
        """
        last_err: Optional[Exception] = None

        for _ in range(retries):
            model = self.current_model
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
                    if not self._rotate():
                        logger.info("All models exhausted — waiting 8 s then resetting.")
                        time.sleep(8)
                        self.reset_failures()
                    continue

                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.RequestException as exc:
                last_err = exc
                logger.error("Request error on %s: %s", model, exc)
                if not self._rotate():
                    time.sleep(3)
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
    ) -> Dict[str, Any]:
        """
        Like ``chat`` but parses the response as JSON.
        Strips markdown code fences if present.
        """
        raw = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
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
        """Advance to the next non-failed model.  Returns False when exhausted."""
        self._failed.add(self._idx)
        for i in range(len(self.FREE_MODELS)):
            if i not in self._failed:
                self._idx = i
                logger.info("Switched to model: %s", self.current_model)
                return True
        return False
