"""Data models and enums for the LLM Narrator subpackage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class NarratorMode(StrEnum):
    """Operation mode for narrative generation."""

    POST_MORTEM = "POST_MORTEM"
    LEARNING_REVIEW = "LEARNING_REVIEW"
    HUMAN_DISCRETION = "HUMAN_DISCRETION"


@dataclass(frozen=True)
class NarratorContext:
    """Encapsulates context metadata passed to narrative generation."""

    mode: NarratorMode
    prompt_payload: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class NarrativeReport:
    """Structured report produced by the LLM Narrator."""

    summary: str
    key_insights: tuple[str, ...]
    suggested_action: str | None
    model: str
    elapsed_seconds: float
    mode: NarratorMode
    is_fallback: bool = False
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the report to a dictionary."""
        return {
            "summary": self.summary,
            "key_insights": list(self.key_insights),
            "suggested_action": self.suggested_action,
            "model": self.model,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "mode": self.mode.value,
            "is_fallback": self.is_fallback,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NarrativeReport:
        """Constructs a NarrativeReport from serialized dictionary data."""
        return cls(
            summary=str(data["summary"]),
            key_insights=tuple(str(x) for x in data.get("key_insights", ())),
            suggested_action=data.get("suggested_action"),
            model=str(data["model"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
            mode=NarratorMode(data["mode"]),
            is_fallback=bool(data.get("is_fallback", False)),
            error_message=data.get("error_message"),
        )
