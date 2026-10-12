"""HTTP client abstraction and implementation for LLM completions."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Protocol

from pydantic import SecretStr

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Exception raised when an LLM completion request fails."""


class LLMTimeoutError(LLMClientError, TimeoutError):
    """Exception raised when an LLM request exceeds the allotted timeout."""


class LLMClientProtocol(Protocol):
    """Protocol for LLM completion providers."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generates a text completion given system and user prompts."""
        ...


class HttpLLMClient:
    """Lightweight HTTP client for OpenAI-compatible chat completions using standard library."""

    def __init__(
        self,
        api_key: SecretStr,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 15.0,
    ) -> None:
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def __repr__(self) -> str:
        return (
            f"HttpLLMClient(base_url={self.base_url!r}, "
            f"model={self.model!r}, timeout_seconds={self.timeout_seconds})"
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Executes a chat completion request and returns the assistant text."""
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                "User-Agent": "xau-bot/0.1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                status_code = resp.getcode()
                body_bytes = resp.read()
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8", errors="replace")
                err_json = json.loads(err_body)
                err_msg = str(err_json.get("error", {}).get("message", err_body))
            except Exception:
                err_msg = str(e.reason)
            safe_msg = err_msg[:200]
            raise LLMClientError(f"HTTP {e.code} error from LLM endpoint: {safe_msg}") from e
        except urllib.error.URLError as e:
            raise LLMClientError(f"Connection failure to LLM endpoint: {e.reason}") from e
        except TimeoutError as e:
            raise LLMTimeoutError(
                f"LLM request timed out after {self.timeout_seconds} seconds"
            ) from e
        except Exception as e:
            raise LLMClientError(f"Unexpected error communicating with LLM: {e}") from e

        if status_code != 200:
            raise LLMClientError(f"Unexpected status code {status_code} from LLM endpoint")

        try:
            resp_data = json.loads(body_bytes.decode("utf-8"))
            content = resp_data["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise ValueError("Expected string content in message")
            return content.strip()
        except Exception as e:
            raise LLMClientError(f"Failed to parse LLM response payload: {e}") from e
