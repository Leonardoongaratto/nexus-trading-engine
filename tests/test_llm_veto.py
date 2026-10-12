"""Tests for LLM Veto Service in Shadow Mode and Fail-Safe Governance."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from xau_bot.advisor.modes import AdvisorDispatcher, ExecutionMode, ProposalManager, ProposalStatus
from xau_bot.domain.models import (
    Direction,
    OrderIntent,
    SetupStatus,
    SetupTriggerType,
    Timeframe,
    TradeSetup,
)
from xau_bot.execution.executor import OrderExecutor
from xau_bot.llm.client import HttpLLMClient, LLMClientError, LLMClientProtocol, LLMTimeoutError
from xau_bot.llm.veto import (
    LLMVetoDecision,
    LLMVetoPromptBuilder,
    LLMVetoService,
)


def _create_dummy_setup(
    setup_id: str = "setup_veto_1",
    direction: Direction = Direction.BULLISH,
) -> TradeSetup:
    now = datetime(2026, 10, 9, 14, 0, tzinfo=UTC)
    return TradeSetup(
        id=setup_id,
        direction=direction,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2660.0,
        time=now,
        confirmed_time=now + timedelta(minutes=5),
        risk_reward_ratio=2.0,
        confluences=("H4_OB", "M15_CHOCH"),
        trigger_type=SetupTriggerType.FVG_CE,
        timeframe=Timeframe.M5,
        status=SetupStatus.PENDING,
    )


def test_llm_veto_decision_model_invariants_and_serialization() -> None:
    decision = LLMVetoDecision(
        setup_id="setup_1",
        veto=True,
        reason="Contra-tendência em H4 sem deslocamento comprovado.",
        confidence=0.85,
        model="gpt-4o-mini-2024-07-18",
        elapsed_seconds=0.45,
        shadow_mode=True,
        timeout_triggered=False,
    )

    d = decision.to_dict()
    assert d["setup_id"] == "setup_1"
    assert d["veto"] is True
    assert d["confidence"] == 0.85
    assert d["shadow_mode"] is True

    reconstructed = LLMVetoDecision.from_dict(d)
    assert reconstructed == decision

    with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
        LLMVetoDecision(
            setup_id="setup_1",
            veto=False,
            reason="ok",
            confidence=1.5,
            model="gpt-4o-mini-2024-07-18",
            elapsed_seconds=0.1,
            shadow_mode=True,
            timeout_triggered=False,
        )

    with pytest.raises(ValueError, match="confidence must be finite"):
        LLMVetoDecision(
            setup_id="setup_1",
            veto=False,
            reason="ok",
            confidence=float("nan"),
            model="gpt-4o-mini-2024-07-18",
            elapsed_seconds=0.1,
            shadow_mode=True,
            timeout_triggered=False,
        )

    with pytest.raises(ValueError, match="elapsed_seconds must be finite"):
        LLMVetoDecision(
            setup_id="setup_1",
            veto=False,
            reason="ok",
            confidence=0.8,
            model="gpt-4o-mini-2024-07-18",
            elapsed_seconds=float("inf"),
            shadow_mode=True,
            timeout_triggered=False,
        )


def test_llm_veto_prompt_builder() -> None:
    setup = _create_dummy_setup()
    builder = LLMVetoPromptBuilder()

    system_prompt = builder.build_system_prompt()
    assert "SMC" in system_prompt
    assert "JSON" in system_prompt

    user_prompt = builder.build_user_prompt(setup=setup, atr=4.5)
    assert "setup_veto_1" in user_prompt
    assert "2650.0" in user_prompt
    assert "BULLISH" in user_prompt
    assert "FVG_CE" in user_prompt
    assert "4.5" in user_prompt


def test_llm_veto_service_evaluates_approval_and_veto(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_client = MagicMock(spec=LLMClientProtocol)
    service = LLMVetoService(
        client=mock_client,
        model="gpt-4o-mini-2024-07-18",
        shadow_mode=True,
        timeout_seconds=5.0,
    )
    setup = _create_dummy_setup()

    mock_client.complete.return_value = json.dumps(
        {
            "veto": False,
            "reason": "Excelente confluência alinhada com viés diário.",
            "confidence": 0.90,
        }
    )
    with caplog.at_level(logging.INFO, logger="xau_bot.llm.veto"):
        caplog.clear()
        decision = service.evaluate_setup(setup)

    assert decision.veto is False
    assert decision.confidence == 0.90
    assert "Excelente confluência" in decision.reason
    assert decision.shadow_mode is True
    assert decision.timeout_triggered is False

    records = [r for r in caplog.records if getattr(r, "event", None) == "llm_veto_decision"]
    assert len(records) == 1
    assert getattr(records[0], "setup_id", None) == setup.id
    assert getattr(records[0], "veto", None) is False

    mock_client.complete.return_value = json.dumps(
        {
            "veto": True,
            "reason": "Entrada em Premium de H4 em range lateral.",
            "confidence": 0.82,
        }
    )
    decision_veto = service.evaluate_setup(setup)
    assert decision_veto.veto is True
    assert decision_veto.confidence == 0.82
    assert "Premium" in decision_veto.reason


def test_llm_veto_service_fail_safe_on_timeout_and_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mock_client = MagicMock(spec=LLMClientProtocol)
    service = LLMVetoService(
        client=mock_client,
        model="gpt-4o-mini-2024-07-18",
        shadow_mode=True,
        timeout_seconds=2.0,
    )
    setup = _create_dummy_setup()

    mock_client.complete.side_effect = LLMTimeoutError("Request timed out")
    with caplog.at_level(logging.WARNING, logger="xau_bot.llm.veto"):
        caplog.clear()
        decision_timeout = service.evaluate_setup(setup)

    assert decision_timeout.veto is True
    assert decision_timeout.timeout_triggered is True
    assert decision_timeout.confidence == 1.0
    assert decision_timeout.reason.startswith("Fail-safe: Timeout")

    timeout_records = [
        r for r in caplog.records if getattr(r, "event", None) == "llm_veto_decision"
    ]
    assert len(timeout_records) == 1
    assert getattr(timeout_records[0], "timeout_triggered", None) is True

    mock_client.complete.side_effect = LLMClientError("500 Server Error")
    decision_err = service.evaluate_setup(setup)
    assert decision_err.veto is True
    assert decision_err.timeout_triggered is True
    assert decision_err.confidence == 1.0
    assert decision_err.reason.startswith("Fail-safe: Erro operacional")

    mock_client.complete.side_effect = None
    mock_client.complete.return_value = "Non-json text returned by model"
    decision_bad_json = service.evaluate_setup(setup)
    assert decision_bad_json.veto is True
    assert decision_bad_json.timeout_triggered is True
    assert decision_bad_json.confidence == 1.0
    assert decision_bad_json.reason.startswith("Fail-safe: Erro operacional")

    for invalid_payload in [
        "{}",
        json.dumps({"veto": "false", "reason": "invalid type", "confidence": 0.5}),
        json.dumps({"veto": True, "reason": "", "confidence": 0.5}),
        json.dumps({"veto": True, "reason": "ok", "confidence": 2.0}),
        json.dumps({"veto": True, "reason": "ok", "confidence": True}),
    ]:
        mock_client.complete.return_value = invalid_payload
        dec = service.evaluate_setup(setup)
        assert dec.veto is True
        assert dec.timeout_triggered is True
        assert dec.reason.startswith("Fail-safe:")


def test_llm_veto_service_with_real_client_timeout() -> None:
    real_client = HttpLLMClient(
        api_key=SecretStr("sk-test-secret-key"),
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini-2024-07-18",
        timeout_seconds=0.1,
    )
    service = LLMVetoService(
        client=real_client,
        model="gpt-4o-mini-2024-07-18",
        shadow_mode=True,
        timeout_seconds=0.1,
    )
    setup = _create_dummy_setup()

    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        decision = service.evaluate_setup(setup)

    assert decision.veto is True
    assert decision.timeout_triggered is True
    assert decision.confidence == 1.0
    assert decision.reason.startswith("Fail-safe: Timeout")


def test_llm_veto_service_triggers_timeout_when_elapsed_exceeds_threshold() -> None:
    mock_client = MagicMock(spec=LLMClientProtocol)
    mock_client.complete.return_value = json.dumps(
        {
            "veto": False,
            "reason": "Resposta tardia.",
            "confidence": 0.88,
        }
    )
    service = LLMVetoService(
        client=mock_client,
        model="gpt-4o-mini-2024-07-18",
        shadow_mode=True,
        timeout_seconds=2.0,
    )
    setup = _create_dummy_setup()

    with patch("time.perf_counter", side_effect=[0.0, 3.5, 3.5]):
        decision = service.evaluate_setup(setup)

    assert decision.veto is True
    assert decision.timeout_triggered is True
    assert decision.confidence == 1.0
    assert decision.reason.startswith("Fail-safe: Timeout")


def test_llm_veto_integration_with_proposal_manager_modes() -> None:
    mock_client = MagicMock(spec=LLMClientProtocol)
    setup = _create_dummy_setup()

    mock_client.complete.return_value = json.dumps(
        {
            "veto": True,
            "reason": "Veto de sombra em teste.",
            "confidence": 0.75,
        }
    )
    shadow_service = LLMVetoService(
        client=mock_client,
        model="gpt-4o-mini-2024-07-18",
        shadow_mode=True,
    )
    shadow_manager = ProposalManager(
        default_timeout_s=300.0,
        veto_service=shadow_service,
    )
    shadow_prop = shadow_manager.create_proposal(setup=setup, mode=ExecutionMode.SEMIAUTO)
    assert shadow_prop.llm_veto is not None
    assert shadow_prop.llm_veto.veto is True
    assert shadow_prop.llm_veto.shadow_mode is True
    assert shadow_prop.status == ProposalStatus.PENDING_APPROVAL

    active_service = LLMVetoService(
        client=mock_client,
        model="gpt-4o-mini-2024-07-18",
        shadow_mode=False,
    )
    active_manager = ProposalManager(
        default_timeout_s=300.0,
        veto_service=active_service,
    )
    active_rejected_prop = active_manager.create_proposal(setup=setup, mode=ExecutionMode.SEMIAUTO)
    assert active_rejected_prop.llm_veto is not None
    assert active_rejected_prop.llm_veto.veto is True
    assert active_rejected_prop.status == ProposalStatus.REJECTED
    assert active_rejected_prop.reason is not None
    assert active_rejected_prop.reason.startswith("Veto ativo pelo LLM:")

    mock_client.complete.return_value = json.dumps(
        {
            "veto": False,
            "reason": "Setup aprovado com confluência.",
            "confidence": 0.95,
        }
    )
    active_approved_prop = active_manager.create_proposal(setup=setup, mode=ExecutionMode.SEMIAUTO)
    assert active_approved_prop.llm_veto is not None
    assert active_approved_prop.llm_veto.veto is False
    assert active_approved_prop.status == ProposalStatus.PENDING_APPROVAL


@pytest.mark.asyncio
async def test_llm_veto_auto_mode_inhibits_order_execution_on_active_veto_or_timeout() -> None:
    mock_client = MagicMock(spec=LLMClientProtocol)
    active_service = LLMVetoService(
        client=mock_client,
        model="gpt-4o-mini-2024-07-18",
        shadow_mode=False,
    )
    proposal_manager = ProposalManager(
        default_timeout_s=300.0,
        veto_service=active_service,
    )
    mock_executor = AsyncMock(spec=OrderExecutor)
    mock_alert_manager = MagicMock()
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        executor=mock_executor,
        alert_manager=mock_alert_manager,
        mode=ExecutionMode.AUTO,
    )
    setup = _create_dummy_setup()
    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.02,
        price=2650.0,
        stop_loss=2645.0,
        take_profit=2660.0,
        magic=123456,
        comment="test_auto_veto",
    )

    mock_client.complete.return_value = json.dumps(
        {
            "veto": True,
            "reason": "Contra-fluxo institucional em H1.",
            "confidence": 0.89,
        }
    )
    prop_vetoed = await dispatcher.handle_setup(setup=setup, intent=intent)
    assert prop_vetoed is not None
    assert prop_vetoed.status == ProposalStatus.REJECTED
    assert prop_vetoed.reason is not None
    assert prop_vetoed.reason.startswith("Veto ativo pelo LLM:")
    assert mock_executor.execute_intent.await_count == 0
    assert mock_alert_manager.notify.called
    assert "[VETADO PELO LLM]" in mock_alert_manager.notify.call_args.kwargs["title"]
    assert mock_alert_manager.notify.call_args.kwargs["level"] == "WARNING"

    mock_client.complete.side_effect = LLMTimeoutError("Tempo esgotado")
    prop_timeout = await dispatcher.handle_setup(setup=setup, intent=intent)
    assert prop_timeout is not None
    assert prop_timeout.status == ProposalStatus.REJECTED
    assert prop_timeout.reason is not None
    assert prop_timeout.reason.startswith("Veto ativo pelo LLM: Fail-safe: Timeout")
    assert mock_executor.execute_intent.await_count == 0


@pytest.mark.asyncio
async def test_llm_veto_auto_mode_handles_approve_proposal_failure() -> None:
    proposal_manager = ProposalManager(default_timeout_s=300.0)
    mock_executor = AsyncMock(spec=OrderExecutor)
    mock_alert_manager = MagicMock()
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        executor=mock_executor,
        alert_manager=mock_alert_manager,
        mode=ExecutionMode.AUTO,
    )
    setup = _create_dummy_setup()
    intent = OrderIntent(
        symbol="XAUUSD",
        direction=Direction.BULLISH,
        order_type="BUY_LIMIT",
        volume=0.02,
        price=2650.0,
        stop_loss=2645.0,
        take_profit=2660.0,
        magic=123456,
        comment="test_auto_fail",
    )

    with patch.object(proposal_manager, "approve_proposal", return_value=(False, "expired", None)):
        prop = await dispatcher.handle_setup(setup=setup, intent=intent)

    assert prop is not None
    assert mock_executor.execute_intent.await_count == 0
    assert mock_alert_manager.notify.called
    assert "[PROPOSTA NÃO APROVADA]" in mock_alert_manager.notify.call_args.kwargs["title"]
    assert mock_alert_manager.notify.call_args.kwargs["level"] == "WARNING"


@pytest.mark.asyncio
async def test_llm_veto_async_dispatcher_offload_and_lock_freedom() -> None:
    mock_client = MagicMock(spec=LLMClientProtocol)
    mock_client.complete.return_value = json.dumps(
        {
            "veto": False,
            "reason": "Aprovado em modo assíncrono.",
            "confidence": 0.91,
        }
    )
    veto_service = LLMVetoService(
        client=mock_client,
        model="gpt-4o-mini-2024-07-18",
        shadow_mode=True,
    )
    proposal_manager = ProposalManager(
        default_timeout_s=300.0,
        veto_service=veto_service,
    )
    dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=ExecutionMode.SEMIAUTO,
    )
    setup = _create_dummy_setup()

    evaluated_lock_available = False
    evaluated_in_worker_thread = False

    def spy_evaluate_setup(s: TradeSetup, atr: float | None = None) -> LLMVetoDecision:
        nonlocal evaluated_lock_available, evaluated_in_worker_thread
        evaluated_in_worker_thread = threading.current_thread() is not threading.main_thread()
        acquired = proposal_manager._lock.acquire(blocking=False)
        if acquired:
            evaluated_lock_available = True
            proposal_manager._lock.release()
        return LLMVetoDecision(
            setup_id=s.id,
            veto=False,
            reason="Aprovado no spy.",
            confidence=0.90,
            model="gpt-4o-mini-2024-07-18",
            elapsed_seconds=0.01,
            shadow_mode=True,
            timeout_triggered=False,
        )

    with patch.object(veto_service, "evaluate_setup", side_effect=spy_evaluate_setup):
        prop = await dispatcher.handle_setup(setup=setup, atr=3.5)
    assert prop is not None
    assert evaluated_lock_available is True
    assert evaluated_in_worker_thread is True
    assert prop.llm_veto is not None
    assert prop.llm_veto.reason == "Aprovado no spy."
