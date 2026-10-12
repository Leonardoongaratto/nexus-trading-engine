"""LLM-based setup veto service operating in shadow mode with deterministic fail-safe."""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from xau_bot.domain.models import TradeSetup
from xau_bot.llm.client import LLMClientProtocol

logger = logging.getLogger("xau_bot.llm.veto")


@dataclass(frozen=True, slots=True)
class LLMVetoDecision:
    """Audit-trail decision record produced by the LLM Veto Service."""

    setup_id: str
    veto: bool
    reason: str
    confidence: float
    model: str
    elapsed_seconds: float
    shadow_mode: bool
    timeout_triggered: bool = False
    raw_response: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence):
            raise ValueError(f"confidence must be finite, got {self.confidence}")
        if not math.isfinite(self.elapsed_seconds):
            raise ValueError(f"elapsed_seconds must be finite, got {self.elapsed_seconds}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict[str, Any]:
        """Serializes veto decision to a dictionary."""
        return {
            "setup_id": self.setup_id,
            "veto": self.veto,
            "reason": self.reason,
            "confidence": round(self.confidence, 4),
            "model": self.model,
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "shadow_mode": self.shadow_mode,
            "timeout_triggered": self.timeout_triggered,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMVetoDecision:
        """Constructs an LLMVetoDecision from serialized dictionary data."""
        return cls(
            setup_id=str(data["setup_id"]),
            veto=bool(data["veto"]),
            reason=str(data["reason"]),
            confidence=float(data["confidence"]),
            model=str(data["model"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
            shadow_mode=bool(data["shadow_mode"]),
            timeout_triggered=bool(data.get("timeout_triggered", False)),
            raw_response=data.get("raw_response"),
        )


class LLMVetoPromptBuilder:
    """Builds deterministic, structured prompts for LLM trade setup veto analysis."""

    def build_system_prompt(self) -> str:
        """Generates the system prompt instructing the model on SMC veto criteria."""
        return (
            "Você é um auditor quantitativo e gestor de risco sênior especializado na metodologia "
            "Smart Money Concepts (SMC) para o ativo XAU/USD (Ouro spot).\n"
            "Sua única função é avaliar propostas de entrada e decidir se deve vetar (veto=True) "
            "ou aprovar (veto=False).\n"
            "Critérios de veto: contra-tendência sem ChoCH confirmado, entrada em área "
            "adversária de Premium/Discount, confluências fracas ou ausentes, "
            "risco excessivo de volatilidade.\n\n"
            "Responda ESTRITAMENTE em formato JSON com o seguinte esquema:\n"
            "{\n"
            '  "veto": true|false,\n'
            '  "reason": "justificativa concisa em português (máximo 150 caracteres)",\n'
            '  "confidence": float entre 0.0 e 1.0\n'
            "}"
        )

    def build_user_prompt(self, setup: TradeSetup, atr: float | None = None) -> str:
        """Generates the user prompt formatting the setup SMC metrics."""
        confluences_str = ", ".join(setup.confluences) if setup.confluences else "Nenhuma"
        atr_str = f"{atr:.2f}" if atr is not None else "N/A"
        return (
            f"Avalie a seguinte proposta de setup SMC:\n"
            f"- ID: {setup.id}\n"
            f"- Direção: {setup.direction.value}\n"
            f"- Entrada: {setup.entry_price:.2f}\n"
            f"- Stop Loss: {setup.stop_loss:.2f}\n"
            f"- Take Profit: {setup.take_profit:.2f}\n"
            f"- Risco/Retorno: {setup.risk_reward_ratio:.2f}\n"
            f"- Gatilho: {setup.trigger_type.value}\n"
            f"- Timeframe: {setup.timeframe.value}\n"
            f"- Confluências: {confluences_str}\n"
            f"- ATR: {atr_str}\n"
            f"- Horário: {setup.time.isoformat()}\n"
        )


class LLMVetoService:
    """Service evaluating trade proposals using an LLM with strict shadow-mode and fail-safe."""

    def __init__(
        self,
        client: LLMClientProtocol,
        model: str = "gpt-4o-mini-2024-07-18",
        shadow_mode: bool = True,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._client = client
        self.model = model
        self.shadow_mode = shadow_mode
        self.timeout_seconds = timeout_seconds
        self._prompt_builder = LLMVetoPromptBuilder()

    def evaluate_setup(self, setup: TradeSetup, atr: float | None = None) -> LLMVetoDecision:
        """Evaluates a trade setup and returns an auditable veto decision."""
        start_time = time.perf_counter()
        system_prompt = self._prompt_builder.build_system_prompt()
        user_prompt = self._prompt_builder.build_user_prompt(setup, atr=atr)

        try:
            raw_response = self._client.complete(system_prompt, user_prompt)
            elapsed = time.perf_counter() - start_time
            if elapsed > self.timeout_seconds:
                raise TimeoutError(
                    f"Tempo limite excedido ({elapsed:.2f}s > {self.timeout_seconds:.2f}s)"
                )
            parsed = self._parse_response(raw_response)
            decision = LLMVetoDecision(
                setup_id=setup.id,
                veto=bool(parsed["veto"]),
                reason=str(parsed["reason"]),
                confidence=float(parsed["confidence"]),
                model=self.model,
                elapsed_seconds=elapsed,
                shadow_mode=self.shadow_mode,
                timeout_triggered=False,
                raw_response=raw_response,
            )
            logger.info(
                "Parecer LLM Veto para %s: veto=%s (shadow=%s, conf=%.2f): %s",
                setup.id,
                decision.veto,
                decision.shadow_mode,
                decision.confidence,
                decision.reason,
                extra={"event": "llm_veto_decision", **decision.to_dict()},
            )
            return decision

        except TimeoutError as exc:
            elapsed = time.perf_counter() - start_time
            decision = LLMVetoDecision(
                setup_id=setup.id,
                veto=True,
                reason=f"Fail-safe: Timeout na chamada ao modelo LLM ({exc}).",
                confidence=1.0,
                model=self.model,
                elapsed_seconds=elapsed,
                shadow_mode=self.shadow_mode,
                timeout_triggered=True,
                raw_response=None,
            )
            logger.warning(
                "Fail-safe acionado por timeout para %s: %s",
                setup.id,
                decision.reason,
                extra={"event": "llm_veto_decision", **decision.to_dict()},
            )
            return decision

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            decision = LLMVetoDecision(
                setup_id=setup.id,
                veto=True,
                reason=f"Fail-safe: Erro operacional ao consultar LLM ({exc}).",
                confidence=1.0,
                model=self.model,
                elapsed_seconds=elapsed,
                shadow_mode=self.shadow_mode,
                timeout_triggered=True,
                raw_response=None,
            )
            logger.warning(
                "Fail-safe acionado por erro para %s: %s",
                setup.id,
                decision.reason,
                extra={"event": "llm_veto_decision", **decision.to_dict()},
            )
            return decision

    def _parse_response(self, text: str) -> dict[str, Any]:
        """Safely parses JSON payload from raw model response, extracting markdown blocks."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```").removesuffix("```").strip()

        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")
        if "veto" not in data or not isinstance(data["veto"], bool):
            raise ValueError("Field 'veto' must be an explicit boolean")
        if (
            "reason" not in data
            or not isinstance(data["reason"], str)
            or not data["reason"].strip()
        ):
            raise ValueError("Field 'reason' must be a non-empty string")
        if (
            "confidence" not in data
            or isinstance(data["confidence"], bool)
            or not isinstance(data["confidence"], (int, float))
        ):
            raise ValueError("Field 'confidence' must be a numeric float")
        confidence = float(data["confidence"])
        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Field 'confidence' must be within [0.0, 1.0], got {confidence}")
        return data
