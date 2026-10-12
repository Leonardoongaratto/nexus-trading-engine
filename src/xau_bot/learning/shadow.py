"""Shadow execution engine for observing candidate parameter filters non-invasively."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from xau_bot.domain.models import Direction, TradeSetup

if TYPE_CHECKING:
    from xau_bot.journal.repository import JournalRepository


class ShadowDivergence(StrEnum):
    ALIGNED = "ALIGNED"
    SHADOW_BLOCKED = "SHADOW_BLOCKED"
    SHADOW_ALLOWED = "SHADOW_ALLOWED"


@dataclass(frozen=True, slots=True)
class ShadowEvaluation:
    setup_id: str
    direction: Direction
    timestamp: datetime
    active_allowed: bool
    shadow_allowed: bool
    divergence: ShadowDivergence
    active_reason: str = ""
    shadow_reason: str = ""
    is_divergent: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "setup_id": self.setup_id,
            "direction": self.direction.value,
            "timestamp": self.timestamp.isoformat(),
            "active_allowed": self.active_allowed,
            "shadow_allowed": self.shadow_allowed,
            "divergence": self.divergence.value,
            "active_reason": self.active_reason,
            "shadow_reason": self.shadow_reason,
            "is_divergent": self.is_divergent,
        }


class ShadowEngine:
    """Evaluates filter proposals under active vs shadow parameter configurations."""

    def evaluate_filter_decision(
        self,
        setup_id: str,
        direction: Direction,
        active_allowed: bool,
        shadow_allowed: bool,
        active_reason: str = "",
        shadow_reason: str = "",
        timestamp: datetime | None = None,
    ) -> ShadowEvaluation:
        now = timestamp or datetime.now(UTC)

        if active_allowed == shadow_allowed:
            divergence = ShadowDivergence.ALIGNED
            is_div = False
        elif active_allowed and not shadow_allowed:
            divergence = ShadowDivergence.SHADOW_BLOCKED
            is_div = True
        else:
            divergence = ShadowDivergence.SHADOW_ALLOWED
            is_div = True

        return ShadowEvaluation(
            setup_id=setup_id,
            direction=direction,
            timestamp=now,
            active_allowed=active_allowed,
            shadow_allowed=shadow_allowed,
            divergence=divergence,
            active_reason=active_reason,
            shadow_reason=shadow_reason,
            is_divergent=is_div,
        )

    def evaluate_setup(
        self,
        setup: TradeSetup,
        active_params: dict[str, Any],
        shadow_params: dict[str, Any],
        spread: float | None = None,
        atr: float | None = None,
        timestamp: datetime | None = None,
    ) -> ShadowEvaluation:
        """Avalia um setup de trade comparando a política ativa com a política sombra.

        Observação metodológica: a avaliação não-invasiva de min_displacement_multiplier
        avalia o displacement_atr_ratio do POI OrderBlock contra o limiar sombra (shadow_min).
        Na detecção causal em produção, confluências secundárias de vela única também podem
        ser consideradas; além disso, a sombra avalia os setups já gerados pelo pipeline ativo
        (evitando recomputação redundante de confluências em voo).
        """
        setup_id = setup.id
        direction = setup.direction
        now = timestamp or setup.confirmed_time

        active_allowed = True
        active_reasons: list[str] = []
        shadow_allowed = True
        shadow_reasons: list[str] = []

        disp_ratio: float | None = None
        poi = getattr(setup, "poi", None)
        if poi is not None and hasattr(poi, "displacement_atr_ratio"):
            disp_ratio = float(poi.displacement_atr_ratio)

        if disp_ratio is not None:
            active_min = float(active_params.get("min_displacement_multiplier", 0.0))
            shadow_min = float(shadow_params.get("min_displacement_multiplier", 0.0))
            if active_min > 0.0 and disp_ratio < active_min:
                active_allowed = False
                active_reasons.append(
                    f"INSUFFICIENT_DISPLACEMENT_ATR: {disp_ratio:.2f} < {active_min:.2f}"
                )
            if shadow_min > 0.0 and disp_ratio < shadow_min:
                shadow_allowed = False
                shadow_reasons.append(
                    f"INSUFFICIENT_DISPLACEMENT_ATR: {disp_ratio:.2f} < {shadow_min:.2f}"
                )

        if spread is not None and atr is not None and atr > 0:
            spread_ratio = spread / atr
            active_max_ratio = float(active_params.get("max_spread_atr_ratio", 1.0))
            shadow_max_ratio = float(shadow_params.get("max_spread_atr_ratio", 1.0))
            if spread_ratio > active_max_ratio:
                active_allowed = False
                active_reasons.append(
                    f"SPREAD_ABOVE_THRESHOLD: {spread_ratio:.4f} > {active_max_ratio:.4f}"
                )
            if spread_ratio > shadow_max_ratio:
                shadow_allowed = False
                shadow_reasons.append(
                    f"SPREAD_ABOVE_THRESHOLD: {spread_ratio:.4f} > {shadow_max_ratio:.4f}"
                )

        return self.evaluate_filter_decision(
            setup_id=str(setup_id),
            direction=direction,
            active_allowed=active_allowed,
            shadow_allowed=shadow_allowed,
            active_reason="; ".join(active_reasons),
            shadow_reason="; ".join(shadow_reasons),
            timestamp=now,
        )

    def record_evaluation(self, repo: JournalRepository, evaluation: ShadowEvaluation) -> None:
        repo.save_shadow_evaluation(evaluation)
