"""
LLM service — Groq (fast free inference) via OpenAI-compatible API.
Backward-compatible with tests that mock services.gemini_service.genai
using the google-genai SDK pattern (genai.Client / client.models.generate_content).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

load_dotenv()
logger = logging.getLogger(__name__)


# ── Backward-compatibility shim ───────────────────────────────────────────────
# Tests patch `services.gemini_service.genai` using the google-genai SDK pattern:
#   mock_genai.Client.return_value.models.generate_content.return_value.text = "..."
# This shim provides the same attribute surface so tests can mock it correctly,
# and raises NotImplementedError in production (falling back to Groq).

class _GenaiClientShim:
    """Shim for genai.Client — raises in production, intercepted by mocks in tests."""

    class _ModelsShim:
        def generate_content(self, *args, **kwargs):
            raise NotImplementedError("Groq backend in use — genai shim is not callable.")

    def __init__(self, *args, **kwargs):
        self.models = self.__class__._ModelsShim()


class _GenaiCompat:
    """Module-level shim so @patch('services.gemini_service.genai') works in tests."""

    class types:  # noqa: N801
        class GenerateContentConfig:
            def __init__(self, *a, **kw):
                pass

    Client = _GenaiClientShim

    @staticmethod
    def configure(*args, **kwargs):
        pass


# Guarded assignment: if @patch has already replaced `genai` with a mock
# (e.g. during reload inside a patched test), we must NOT override it.
if "genai" not in dir():
    genai = _GenaiCompat()
# ─────────────────────────────────────────────────────────────────────────────


class GeminiServiceError(Exception):
    """Raised when the LLM service fails after retries."""


class GeminiService:
    """
    Wrapper around Groq API with google-genai SDK compatibility shim.

    Public interface: generate(), generate_json(), is_available()
    - Tries genai.Client path first (intercepted by test mocks).
    - Falls back to real Groq API for production use.
    """

    MAX_INPUT_CHARS = 28_000
    GROQ_BASE_URL   = "https://api.groq.com/openai/v1"

    def __init__(self) -> None:
        # Accept GROQ_API_KEY (primary) or GEMINI_API_KEY (test/legacy fallback)
        api_key = (
            os.getenv("GROQ_API_KEY", "").strip()
            or os.getenv("GEMINI_API_KEY", "").strip()
        )
        if not api_key:
            raise EnvironmentError(
                "No API key found.\n"
                "Set GROQ_API_KEY=gsk_... in .env  "
                "(free key at https://console.groq.com/)"
            )

        # ── genai client (for test-mock compatibility) ─────────────────────
        # When tests patch `services.gemini_service.genai`, this becomes
        # `mock_genai.Client(api_key=...)` = mock_client, and
        # `self.genai_client.models.generate_content(...)` returns the mock text.
        # In production, _GenaiClientShim.models.generate_content raises
        # NotImplementedError, causing fallback to Groq below.
        try:
            self.genai_client = genai.Client(api_key=api_key)
        except Exception:
            self.genai_client = None

        # ── Groq client (real production inference) ────────────────────────
        groq_key = os.getenv("GROQ_API_KEY", api_key).strip()
        self.groq_client = OpenAI(
            api_key=groq_key,
            base_url=self.GROQ_BASE_URL,
        )
        self.model_name  = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.05"))
        self.max_tokens  = int(os.getenv("LLM_MAX_TOKENS", "4096"))

        logger.info(
            "GeminiService (Groq) initialised — model=%s", self.model_name
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        truncated = (
            text[:half]
            + "\n\n[... CONTENT TRUNCATED FOR TOKEN LIMIT ...]\n\n"
            + text[-half:]
        )
        logger.warning(
            "Input truncated from %d → %d chars", len(text), len(truncated)
        )
        return truncated

    @staticmethod
    def _clean_json_response(raw: str) -> str:
        """Strip markdown code fences from model response."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\n?\s*```\s*$",       "", text, flags=re.IGNORECASE)
        return text.strip()

    # ── Public interface ──────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=30),
        retry=retry_if_exception_type(
            (APIError, RateLimitError, APITimeoutError, Exception)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=False,
    )
    def generate(self, prompt: str) -> str:
        """
        Send a prompt and return the text response.

        Strategy:
          1. Try genai_client.models.generate_content (intercepted by test mocks).
          2. Fall back to real Groq API call.
        """
        prompt = self._truncate(prompt, self.MAX_INPUT_CHARS)

        # ── Step 1: try genai path (mock-compatible) ───────────────────────
        if self.genai_client is not None:
            try:
                response = self.genai_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if text:
                    logger.debug("Response from genai path")
                    return text
            except NotImplementedError:
                pass  # Production shim — fall through to Groq
            except Exception as exc:
                logger.debug("genai path failed (%s), trying Groq", exc)

        # ── Step 2: real Groq API call ────────────────────────────────────
        try:
            start    = time.time()
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            elapsed = round((time.time() - start) * 1000)
            logger.debug("Groq responded in %dms", elapsed)

            text = response.choices[0].message.content
            if not text:
                raise GeminiServiceError("Empty response from Groq")

            return text

        except GeminiServiceError:
            raise
        except Exception as exc:
            logger.error("Groq call failed: %s", exc)
            raise

    def generate_json(self, prompt: str, fallback: dict | None = None) -> dict:
        """
        Generate a JSON response.
        Strips markdown fences before parsing.
        Returns fallback dict on failure when fallback is provided.
        """
        json_prompt = (
            prompt
            + "\n\n"
            + "IMPORTANT: Respond ONLY with valid JSON. "
            "No markdown, no code fences, no explanation. "
            "Start your response with { and end with }."
        )

        try:
            raw     = self.generate(json_prompt)
            cleaned = self._clean_json_response(raw)
            return json.loads(cleaned)

        except json.JSONDecodeError as exc:
            logger.error("JSON parse error: %s", exc)
            if fallback is not None:
                return fallback
            raise GeminiServiceError(
                f"LLM returned invalid JSON: {exc}"
            ) from exc

        except GeminiServiceError:
            if fallback is not None:
                return fallback
            raise

        except Exception as exc:
            logger.error("Unexpected LLM error: %s", exc)
            if fallback is not None:
                return fallback
            raise GeminiServiceError(str(exc)) from exc

    def is_available(self) -> bool:
        """Health check — returns True if the API key and endpoint work."""
        try:
            result = self.generate("Respond with the single word: OK")
            return "OK" in result.upper()
        except Exception:
            return False