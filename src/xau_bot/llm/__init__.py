"""LLM Narrator subpackage for qualitative post-trade and learning synthesis."""

from __future__ import annotations

from xau_bot.llm.client import (
    HttpLLMClient,
    LLMClientError,
    LLMClientProtocol,
    LLMTimeoutError,
)
from xau_bot.llm.models import NarrativeReport, NarratorContext, NarratorMode
from xau_bot.llm.narrator import LLMNarrator
from xau_bot.llm.veto import LLMVetoDecision, LLMVetoPromptBuilder, LLMVetoService

__all__ = [
    "HttpLLMClient",
    "LLMClientError",
    "LLMClientProtocol",
    "LLMTimeoutError",
    "LLMNarrator",
    "LLMVetoDecision",
    "LLMVetoPromptBuilder",
    "LLMVetoService",
    "NarrativeReport",
    "NarratorContext",
    "NarratorMode",
]
