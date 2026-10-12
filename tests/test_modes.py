import argparse
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Generator
from datetime import UTC, datetime, timedelta
from http import HTTPStatus
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from xau_bot.advisor.modes import (
    AdvisorDispatcher,
    ExecutionMode,
    ProposalManager,
    ProposalStatus,
    TradeProposal,
)
from xau_bot.cli.run import build_run_parser, run_bot_cli
from xau_bot.config import Settings
from xau_bot.domain.models import Direction, OrderIntent, SetupTriggerType, Timeframe, TradeSetup
from xau_bot.execution.models import ExecutionResult, ExecutionStatus
from xau_bot.server import HealthStatusState, create_server


def make_test_setup(
    setup_id: str = "setup_test_1",
    direction: Direction = Direction.BULLISH,
    entry: float = 2650.0,
    sl: float = 2640.0,
    tp: float = 2670.0,
) -> TradeSetup:
    t0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    rr = abs(tp - entry) / abs(entry - sl)
    return TradeSetup(
        id=setup_id,
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        time=t0,
        confirmed_time=t0 + timedelta(minutes=5),
        risk_reward_ratio=rr,
        trigger_type=SetupTriggerType.FVG_CE,
        timeframe=Timeframe.M5,
        confluences=("SWEEP", "CHOCH"),
    )


def make_test_intent(setup: TradeSetup, volume: float = 0.1) -> OrderIntent:
    return OrderIntent(
        symbol="XAUUSD",
        direction=setup.direction,
        order_type="BUY_LIMIT" if setup.direction == Direction.BULLISH else "SELL_LIMIT",
        volume=volume,
        price=setup.entry_price,
        stop_loss=setup.stop_loss,
        take_profit=setup.take_profit,
        magic=202610,
        comment="test_intent",
    )


@pytest.fixture
def running_server_with_manager() -> Generator[
    tuple[str, HealthStatusState, ProposalManager], None, None
]:
    manager = ProposalManager(default_timeout_s=300.0)
    state = HealthStatusState(execution_mode="semiauto", proposal_manager=manager)
    server = create_server(host="127.0.0.1", port=0, state=state)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{actual_port}"
    try:
        yield base_url, state, manager
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_settings_execution_mode_and_timeout() -> None:
    settings = Settings()
    assert settings.execution_mode == ExecutionMode.SEMIAUTO
    assert settings.proposal_timeout_s == 300.0

    custom = Settings(execution_mode=ExecutionMode.ANALYST, proposal_timeout_s=120.0)
    assert custom.execution_mode == ExecutionMode.ANALYST
    assert custom.proposal_timeout_s == 120.0


def test_cli_run_mode_argument() -> None:
    parser = argparse.ArgumentParser()
    build_run_parser(parser)

    args_analyst = parser.parse_args(["--mode", "analyst"])
    assert args_analyst.mode == "analyst"

    args_semiauto = parser.parse_args(["--mode", "semiauto"])
    assert args_semiauto.mode == "semiauto"

    args_auto = parser.parse_args(["--mode", "auto"])
    assert args_auto.mode == "auto"

    with pytest.raises(SystemExit):
        parser.parse_args(["--mode", "invalid_mode"])


def test_trade_proposal_validation_and_serialization() -> None:
    t0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    t_exp = t0 + timedelta(minutes=5)
    setup = make_test_setup()

    prop = TradeProposal(
        id="prop_001",
        setup_id=setup.id,
        symbol="XAUUSD",
        direction=setup.direction,
        entry_price=setup.entry_price,
        stop_loss=setup.stop_loss,
        take_profit=setup.take_profit,
        volume=0.1,
        risk_usd=100.0,
        risk_pct=1.0,
        created_at=t0,
        expires_at=t_exp,
        status=ProposalStatus.PENDING_APPROVAL,
        mode=ExecutionMode.SEMIAUTO,
    )

    data = prop.to_dict()
    assert data["id"] == "prop_001"
    assert data["direction"] == "BUY"
    assert data["raw_direction"] == "BULLISH"
    assert data["status"] == "PENDING_APPROVAL"
    assert data["mode"] == "semiauto"
    assert data["volume"] == 0.1

    with pytest.raises(ValueError, match="created_at must be timezone-aware"):
        TradeProposal(
            id="bad_1",
            setup_id=setup.id,
            symbol="XAUUSD",
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            volume=0.1,
            risk_usd=100.0,
            risk_pct=1.0,
            created_at=datetime(2026, 10, 4, 12, 0),
            expires_at=t_exp,
        )

    with pytest.raises(ValueError, match="expires_at must be strictly greater than created_at"):
        TradeProposal(
            id="bad_2",
            setup_id=setup.id,
            symbol="XAUUSD",
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            volume=0.1,
            risk_usd=100.0,
            risk_pct=1.0,
            created_at=t0,
            expires_at=t0,
        )


def test_proposal_manager_lifecycle() -> None:
    manager = ProposalManager(default_timeout_s=300.0)
    setup = make_test_setup("setup_lifecycle_1")
    t0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)

    prop = manager.create_proposal(
        setup=setup,
        volume=0.15,
        risk_usd=150.0,
        risk_pct=1.5,
        mode=ExecutionMode.SEMIAUTO,
        now=t0,
    )
    assert prop.status == ProposalStatus.PENDING_APPROVAL
    assert manager.get_proposal(prop.id) is prop
    assert len(manager.get_proposals(ProposalStatus.PENDING_APPROVAL)) == 1

    ok, reason, approved_prop = manager.approve_proposal(prop.id, now=t0 + timedelta(seconds=10))
    assert ok is True
    assert reason == "approved"
    assert approved_prop is not None
    assert approved_prop.status == ProposalStatus.APPROVED

    ok_again, reason_again, _ = manager.approve_proposal(prop.id)
    assert ok_again is False
    assert reason_again == "invalid_state"

    dispatched = manager.mark_dispatched(prop.id, order_ticket=999888)
    assert dispatched is not None
    assert dispatched.status == ProposalStatus.DISPATCHED
    assert dispatched.order_ticket == 999888

    stats = manager.get_stats()
    assert stats["total"] == 1
    assert stats["dispatched"] == 1
    assert stats["pending"] == 0


def test_proposal_manager_rejection_and_not_found() -> None:
    manager = ProposalManager()
    setup = make_test_setup("setup_rej_1")

    prop = manager.create_proposal(setup=setup)
    ok_nf, reason_nf, _ = manager.reject_proposal("non_existent_id")
    assert ok_nf is False
    assert reason_nf == "not_found"

    ok_nf_app, reason_nf_app, _ = manager.approve_proposal("non_existent_id")
    assert ok_nf_app is False
    assert reason_nf_app == "not_found"

    ok, reason, rej_prop = manager.reject_proposal(prop.id, reason="market_too_volatile")
    assert ok is True
    assert reason == "rejected"
    assert rej_prop is not None
    assert rej_prop.status == ProposalStatus.REJECTED
    assert rej_prop.reason == "market_too_volatile"

    ok_again, reason_again, _ = manager.reject_proposal(prop.id)
    assert ok_again is False
    assert reason_again == "invalid_state"


def test_proposal_manager_expiration() -> None:
    manager = ProposalManager(default_timeout_s=60.0)
    setup = make_test_setup("setup_exp_1")
    t0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)

    prop = manager.create_proposal(setup=setup, timeout_s=60.0, now=t0)
    assert prop.status == ProposalStatus.PENDING_APPROVAL

    expired_list = manager.check_expirations(now=t0 + timedelta(seconds=30))
    assert len(expired_list) == 0

    expired_list = manager.check_expirations(now=t0 + timedelta(seconds=65))
    assert len(expired_list) == 1
    assert expired_list[0].id == prop.id
    assert expired_list[0].status == ProposalStatus.EXPIRED

    setup2 = make_test_setup("setup_exp_2")
    prop2 = manager.create_proposal(setup=setup2, timeout_s=60.0, now=t0)
    ok_app, reason_app, _ = manager.approve_proposal(prop2.id, now=t0 + timedelta(seconds=70))
    assert ok_app is False
    assert reason_app == "expired"
    refreshed_prop2 = manager.get_proposal(prop2.id)
    assert refreshed_prop2 is not None
    assert refreshed_prop2.status == ProposalStatus.EXPIRED


def test_proposal_manager_cancellation_by_price_and_setup() -> None:
    manager = ProposalManager()
    t0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)

    bullish_setup = make_test_setup(
        "bullish_1", Direction.BULLISH, entry=2650.0, sl=2640.0, tp=2670.0
    )
    prop_bull = manager.create_proposal(setup=bullish_setup, now=t0)

    bearish_setup = make_test_setup(
        "bearish_1", Direction.BEARISH, entry=2650.0, sl=2660.0, tp=2630.0
    )
    prop_bear = manager.create_proposal(setup=bearish_setup, now=t0)

    cancelled = manager.cancel_by_price(symbol="XAUUSD", current_price=2645.0)
    assert len(cancelled) == 0

    cancelled_bull = manager.cancel_by_price(symbol="XAUUSD", current_price=2639.5)
    assert len(cancelled_bull) == 1
    assert cancelled_bull[0].id == prop_bull.id
    assert prop_bull.status == ProposalStatus.CANCELLED

    cancelled_bear = manager.cancel_by_price(symbol="XAUUSD", current_price=2661.0)
    assert len(cancelled_bear) == 1
    assert cancelled_bear[0].id == prop_bear.id
    assert prop_bear.status == ProposalStatus.CANCELLED

    setup_inv = make_test_setup("setup_inv_1")
    prop_inv = manager.create_proposal(setup=setup_inv, now=t0)
    cancelled_by_setup = manager.cancel_invalidated(setup_inv.id, reason="bos_failed")
    assert len(cancelled_by_setup) == 1
    assert cancelled_by_setup[0].id == prop_inv.id
    assert prop_inv.status == ProposalStatus.CANCELLED
    assert prop_inv.reason == "bos_failed"


@pytest.mark.asyncio
async def test_advisor_dispatcher_analyst_mode() -> None:
    manager = ProposalManager()
    mock_alert_mgr = MagicMock()
    mock_alert_mgr.notify = AsyncMock()
    mock_executor = AsyncMock()

    dispatcher = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.ANALYST,
        alert_manager=mock_alert_mgr,
        executor=mock_executor,
    )

    setup = make_test_setup("setup_analyst_1")
    intent = make_test_intent(setup)

    prop = await dispatcher.handle_setup(
        setup=setup, intent=intent, volume=0.2, risk_usd=200.0, risk_pct=2.0
    )
    assert prop is not None
    assert prop.status == ProposalStatus.ADVISORY
    mock_executor.execute_intent.assert_not_called()
    mock_alert_mgr.notify.assert_awaited_once()
    _, kwargs = mock_alert_mgr.notify.call_args
    assert "[SUGESTÃO ANALISTA]" in kwargs["title"]
    assert "SL: 2640.00" in kwargs["message"]

    ok, reason, _ = manager.approve_proposal(prop.id)
    assert ok is False
    assert reason == "invalid_mode"


@pytest.mark.asyncio
async def test_advisor_dispatcher_semiauto_mode() -> None:
    manager = ProposalManager()
    mock_alert_mgr = MagicMock()
    mock_alert_mgr.notify = AsyncMock()
    mock_executor = AsyncMock()

    dispatcher = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.SEMIAUTO,
        alert_manager=mock_alert_mgr,
        executor=mock_executor,
    )

    setup = make_test_setup("setup_semiauto_1")
    intent = make_test_intent(setup)

    prop = await dispatcher.handle_setup(
        setup=setup, intent=intent, volume=0.1, risk_usd=100.0, risk_pct=1.0
    )
    assert prop is not None
    assert prop.status == ProposalStatus.PENDING_APPROVAL
    mock_executor.execute_intent.assert_not_called()
    mock_alert_mgr.notify.assert_awaited_once()
    _, kwargs = mock_alert_mgr.notify.call_args
    assert "[APROVAÇÃO NECESSÁRIA]" in kwargs["title"]


@pytest.mark.asyncio
async def test_advisor_dispatcher_auto_mode() -> None:
    manager = ProposalManager()
    mock_alert_mgr = MagicMock()
    mock_alert_mgr.notify = AsyncMock()
    mock_executor = AsyncMock()
    mock_executor.execute_intent.return_value = ExecutionResult(
        success=True,
        status=ExecutionStatus.FILLED,
        ticket=123456,
        price=2650.0,
        volume=0.1,
        retcode=10009,
    )

    dispatcher = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.AUTO,
        alert_manager=mock_alert_mgr,
        executor=mock_executor,
    )

    setup = make_test_setup("setup_auto_1")
    intent = make_test_intent(setup)

    prop = await dispatcher.handle_setup(
        setup=setup, intent=intent, volume=0.1, risk_usd=100.0, risk_pct=1.0
    )
    assert prop is not None
    assert prop.status == ProposalStatus.DISPATCHED
    assert prop.order_ticket == 123456
    mock_executor.execute_intent.assert_called_once_with(intent)
    mock_alert_mgr.notify.assert_awaited_once()
    _, kwargs = mock_alert_mgr.notify.call_args
    assert "[ORDEM AUTOMÁTICA ENVIADA]" in kwargs["title"]


def test_server_status_and_proposals_api(
    running_server_with_manager: tuple[str, HealthStatusState, ProposalManager],
) -> None:
    base_url, state, manager = running_server_with_manager
    setup = make_test_setup("setup_api_1")
    prop = manager.create_proposal(setup=setup)

    with urllib.request.urlopen(f"{base_url}/status") as resp:
        assert resp.status == HTTPStatus.OK
        data = json.loads(resp.read().decode("utf-8"))
        assert data["execution_mode"] == "semiauto"
        assert data["proposals"]["total"] == 1
        assert data["proposals"]["pending"] == 1

    with urllib.request.urlopen(f"{base_url}/api/proposals") as resp:
        assert resp.status == HTTPStatus.OK
        props_data = json.loads(resp.read().decode("utf-8"))
        assert len(props_data["proposals"]) == 1
        assert props_data["proposals"][0]["id"] == prop.id

    req_approve = urllib.request.Request(
        f"{base_url}/api/proposals/{prop.id}/approve",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_approve) as resp:
        assert resp.status == HTTPStatus.OK
        app_res = json.loads(resp.read().decode("utf-8"))
        assert app_res["status"] == "approved"
        assert app_res["proposal_id"] == prop.id
        assert app_res["order_dispatched"] is False

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req_approve)
    assert exc_info.value.code == HTTPStatus.CONFLICT
    conflict_body = json.loads(exc_info.value.read().decode("utf-8"))
    assert conflict_body["error"] == "proposal already expired or rejected"

    req_not_found = urllib.request.Request(
        f"{base_url}/api/proposals/missing_prop/approve",
        data=b"",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_nf:
        urllib.request.urlopen(req_not_found)
    assert exc_nf.value.code == HTTPStatus.NOT_FOUND

    prop2 = manager.create_proposal(setup=make_test_setup("setup_api_2"))
    req_reject = urllib.request.Request(
        f"{base_url}/api/proposals/{prop2.id}/reject",
        data=json.dumps({"reason": "high_spread"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req_reject) as resp_rej:
        assert resp_rej.status == HTTPStatus.OK
        rej_res = json.loads(resp_rej.read().decode("utf-8"))
        assert rej_res["status"] == "rejected"
        assert rej_res["proposal_id"] == prop2.id
        assert prop2.status == ProposalStatus.REJECTED
        assert prop2.reason == "high_spread"

    with pytest.raises(urllib.error.HTTPError) as exc_rej_conflict:
        urllib.request.urlopen(req_reject)
    assert exc_rej_conflict.value.code == HTTPStatus.CONFLICT


def test_proposal_manager_thread_safety() -> None:
    manager = ProposalManager()
    proposals: list[TradeProposal] = []

    for i in range(50):
        s = make_test_setup(f"setup_ts_{i}")
        proposals.append(manager.create_proposal(setup=s))

    errors: list[Exception] = []

    def _worker(start_idx: int) -> None:
        try:
            for j in range(start_idx, start_idx + 10):
                p = proposals[j]
                if j % 2 == 0:
                    manager.approve_proposal(p.id)
                else:
                    manager.reject_proposal(p.id, reason="thread_test")
                manager.get_stats()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i * 10,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    stats = manager.get_stats()
    assert stats["total"] == 50
    assert stats["pending"] == 0
    assert stats["approved"] == 25
    assert stats["rejected"] == 25


def test_proposal_edge_cases() -> None:
    manager = ProposalManager()
    assert manager.mark_dispatched("non_existent_id") is None

    t0 = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
    setup = make_test_setup()
    prop = manager.create_proposal(setup=setup, now=t0)
    manager.approve_proposal(prop.id)
    manager.mark_dispatched(prop.id, order_ticket=123, now=t0 + timedelta(seconds=10))
    d = prop.to_dict()
    assert d["dispatched_at"] is not None
    assert d["order_ticket"] == 123

    with pytest.raises(ValueError, match="expires_at must be timezone-aware"):
        TradeProposal(
            id="bad_exp_tz",
            setup_id="s1",
            symbol="XAUUSD",
            direction=Direction.BULLISH,
            entry_price=2650.0,
            stop_loss=2640.0,
            take_profit=2670.0,
            volume=0.1,
            risk_usd=100.0,
            risk_pct=1.0,
            created_at=t0,
            expires_at=datetime(2026, 10, 4, 12, 5),
        )

    prop_other = manager.create_proposal(setup=setup, now=t0)
    prop_other.symbol = "EURUSD"
    cancelled = manager.cancel_by_price(symbol="XAUUSD", current_price=2630.0)
    assert prop_other not in cancelled


@pytest.mark.asyncio
async def test_advisor_dispatcher_auto_mode_failure() -> None:
    manager = ProposalManager()
    failing_executor = AsyncMock()
    failing_executor.execute_intent.return_value = ExecutionResult(
        success=False,
        status=ExecutionStatus.REJECTED,
        retcode=10013,
        error_message="Invalid volume",
    )
    mock_alert_mgr = MagicMock()
    mock_alert_mgr.notify = AsyncMock()

    dispatcher = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.AUTO,
        alert_manager=mock_alert_mgr,
        executor=failing_executor,
    )

    setup = make_test_setup("setup_fail_1")
    intent = make_test_intent(setup)
    prop = await dispatcher.handle_setup(setup=setup, intent=intent)
    assert prop is not None
    assert prop.status == ProposalStatus.DISPATCH_FAILED
    assert prop.order_ticket is None
    assert "Invalid volume" in str(prop.reason)
    mock_alert_mgr.notify.assert_awaited_once()
    _, kwargs = mock_alert_mgr.notify.call_args
    assert "[FALHA NO DISPARO AUTOMÁTICO]" in kwargs["title"]


@pytest.mark.asyncio
async def test_advisor_dispatcher_resilience() -> None:
    manager = ProposalManager()
    failing_executor = AsyncMock()
    failing_executor.execute_intent.side_effect = RuntimeError("Broker connection dropped")
    mock_alert_mgr = MagicMock()
    mock_alert_mgr.notify = AsyncMock()

    dispatcher = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.AUTO,
        alert_manager=mock_alert_mgr,
        executor=failing_executor,
    )

    setup = make_test_setup("setup_fail_2")
    intent = make_test_intent(setup)
    prop = await dispatcher.handle_setup(setup=setup, intent=intent)
    assert prop is not None
    assert prop.status == ProposalStatus.DISPATCH_FAILED
    assert prop.order_ticket is None
    assert "Broker connection dropped" in str(prop.reason)
    mock_alert_mgr.notify.assert_awaited_once()
    _, kwargs = mock_alert_mgr.notify.call_args
    assert "[FALHA NO DISPARO AUTOMÁTICO]" in kwargs["title"]


def test_server_http_dispatch_execution_and_errors() -> None:
    manager = ProposalManager(default_timeout_s=300.0)
    mock_executor = AsyncMock()
    mock_executor.execute_intent.return_value = ExecutionResult(
        success=True,
        status=ExecutionStatus.FILLED,
        ticket=888999,
        retcode=10009,
    )

    state = HealthStatusState(
        execution_mode="semiauto",
        proposal_manager=manager,
        order_executor=mock_executor,
    )
    server = create_server(host="127.0.0.1", port=0, state=state)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{actual_port}"

    try:
        setup_ok = make_test_setup("prop_http_ok")
        prop_ok = manager.create_proposal(setup=setup_ok, mode=ExecutionMode.SEMIAUTO)
        prop_ok.intent = make_test_intent(setup_ok)

        req_ok = urllib.request.Request(
            f"{base_url}/api/proposals/{prop_ok.id}/approve",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_ok) as resp:
            assert resp.status == HTTPStatus.OK
            data = json.loads(resp.read().decode("utf-8"))
            assert data["status"] == "approved"
            assert data["order_dispatched"] is True
            assert prop_ok.status == ProposalStatus.DISPATCHED
            assert prop_ok.order_ticket == 888999

        setup_fail = make_test_setup("prop_http_fail")
        prop_fail = manager.create_proposal(setup=setup_fail, mode=ExecutionMode.SEMIAUTO)
        prop_fail.intent = make_test_intent(setup_fail)

        mock_executor.execute_intent.return_value = ExecutionResult(
            success=False,
            status=ExecutionStatus.REJECTED,
            retcode=10013,
            error_message="Insufficient margin",
        )
        req_fail = urllib.request.Request(
            f"{base_url}/api/proposals/{prop_fail.id}/approve",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_fail:
            urllib.request.urlopen(req_fail)
        assert exc_fail.value.code == HTTPStatus.BAD_GATEWAY
        fail_body = json.loads(exc_fail.value.read().decode("utf-8"))
        assert fail_body["status"] == "dispatch_failed"
        assert "Insufficient margin" in fail_body["error"]
        assert prop_fail.status == ProposalStatus.DISPATCH_FAILED

        setup_analyst = make_test_setup("prop_analyst_test")
        prop_analyst = manager.create_proposal(setup=setup_analyst, mode=ExecutionMode.ANALYST)
        prop_analyst.intent = make_test_intent(setup_analyst)
        req_analyst = urllib.request.Request(
            f"{base_url}/api/proposals/{prop_analyst.id}/approve",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_analyst:
            urllib.request.urlopen(req_analyst)
        assert exc_analyst.value.code == HTTPStatus.CONFLICT
        analyst_body = json.loads(exc_analyst.value.read().decode("utf-8"))
        assert "analyst" in analyst_body["error"].lower()

    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_server_reject_invalid_body_and_malformed_headers(
    running_server_with_manager: tuple[str, HealthStatusState, ProposalManager],
) -> None:
    base_url, _, manager = running_server_with_manager
    setup = make_test_setup("prop_bad_body")
    prop = manager.create_proposal(setup=setup)

    req_bad_json = urllib.request.Request(
        f"{base_url}/api/proposals/{prop.id}/reject",
        data=b"{not_valid_json",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_json:
        urllib.request.urlopen(req_bad_json)
    assert exc_json.value.code == HTTPStatus.BAD_REQUEST

    req_bad_cl = urllib.request.Request(
        f"{base_url}/api/proposals/{prop.id}/reject",
        data=b"{}",
        headers={"Content-Type": "application/json", "Content-Length": "not_an_int"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_cl:
        urllib.request.urlopen(req_bad_cl)
    assert exc_cl.value.code == HTTPStatus.BAD_REQUEST


def test_server_status_calls_check_expirations(
    running_server_with_manager: tuple[str, HealthStatusState, ProposalManager],
) -> None:
    base_url, _, manager = running_server_with_manager
    t0 = datetime.now(UTC) - timedelta(seconds=100)
    setup = make_test_setup("prop_status_exp")
    manager.create_proposal(setup=setup, timeout_s=10.0, now=t0)

    with urllib.request.urlopen(f"{base_url}/status") as resp:
        assert resp.status == HTTPStatus.OK
        data = json.loads(resp.read().decode("utf-8"))
        assert data["proposals"]["expired"] >= 1
        assert data["proposals"]["pending"] == 0


def test_run_bot_cli_applies_mode_to_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_settings: list[Settings] = []

    def mock_run_server(settings: Settings, **kwargs: Any) -> None:
        captured_settings.append(settings)

    with (
        patch("xau_bot.cli.run.run_server", side_effect=mock_run_server),
        patch("xau_bot.cli.run.create_broker_adapter"),
        patch("xau_bot.cli.run.Watchdog"),
        patch("xau_bot.cli.run.SessionFilter"),
    ):
        parser = argparse.ArgumentParser()
        build_run_parser(parser)
        parsed = parser.parse_args(["--broker", "sim", "--mode", "analyst"])
        code = run_bot_cli(parsed)
        assert code == 0
        assert len(captured_settings) == 1
        assert captured_settings[0].execution_mode == ExecutionMode.ANALYST


def test_server_reject_not_found(
    running_server_with_manager: tuple[str, HealthStatusState, ProposalManager],
) -> None:
    base_url, _, _ = running_server_with_manager
    req = urllib.request.Request(
        f"{base_url}/api/proposals/non_existent_proposal_id/reject",
        data=json.dumps({"reason": "manual"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
    assert exc_info.value.code == HTTPStatus.NOT_FOUND


def test_server_http_dispatch_executor_exception() -> None:
    manager = ProposalManager()
    setup = make_test_setup("prop_exc")
    intent = make_test_intent(setup)
    prop = manager.create_proposal(setup=setup, intent=intent)

    executor = MagicMock()
    executor.execute_intent = AsyncMock(side_effect=RuntimeError("Broker disconnected"))

    state = HealthStatusState(proposal_manager=manager, order_executor=executor)
    server = create_server(host="127.0.0.1", port=0, state=state)
    actual_port = server.server_address[1]
    base_url = f"http://127.0.0.1:{actual_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        req = urllib.request.Request(
            f"{base_url}/api/proposals/{prop.id}/approve",
            data=b"",
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == HTTPStatus.BAD_GATEWAY
        err_data = json.loads(exc_info.value.read().decode("utf-8"))
        assert err_data["status"] == "dispatch_failed"
        assert "Broker disconnected" in err_data["error"]
        p = manager.get_proposal(prop.id)
        assert p is not None
        assert p.status == ProposalStatus.DISPATCH_FAILED
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


@pytest.mark.asyncio
async def test_advisor_dispatcher_auto_without_executor() -> None:
    manager = ProposalManager()
    alerts = MagicMock()
    dispatcher = AdvisorDispatcher(
        mode=ExecutionMode.AUTO,
        proposal_manager=manager,
        executor=None,
        alert_manager=alerts,
    )
    setup = make_test_setup("auto_no_exec")
    intent = make_test_intent(setup)

    prop = await dispatcher.handle_setup(setup, intent)
    assert prop is not None
    assert prop.status == ProposalStatus.DISPATCH_FAILED
    assert "Missing executor" in (prop.reason or "")
    alerts.notify.assert_called_once()
    assert alerts.notify.call_args[1]["level"] == "ERROR"


def test_server_proposals_filter_and_ordering() -> None:
    manager = ProposalManager()
    t0 = datetime.now(UTC)
    t1 = t0 + timedelta(seconds=1)
    t2 = t0 + timedelta(seconds=2)

    p1 = manager.create_proposal(setup=make_test_setup("p1"), now=t0)
    p2 = manager.create_proposal(setup=make_test_setup("p2"), now=t1)
    p3 = manager.create_proposal(setup=make_test_setup("p3"), now=t2)
    ok, _, _ = manager.approve_proposal(p2.id)
    assert ok is True

    state = HealthStatusState(proposal_manager=manager)
    server = create_server(host="127.0.0.1", port=0, state=state)
    actual_port = server.server_address[1]
    base_url = f"http://127.0.0.1:{actual_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # All proposals should be in reverse chronological order (p3, p2, p1)
        with urllib.request.urlopen(f"{base_url}/api/proposals") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            ids = [p["id"] for p in data["proposals"]]
            assert ids == [p3.id, p2.id, p1.id]

        # Filter by status=approved
        with urllib.request.urlopen(f"{base_url}/api/proposals?status=approved") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            ids = [p["id"] for p in data["proposals"]]
            assert ids == [p2.id]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_proposal_manager_listeners_and_stats() -> None:
    manager = ProposalManager(default_timeout_s=10.0)
    received: list[TradeProposal] = []

    def _good_listener(p: TradeProposal) -> None:
        received.append(p)

    def _bad_listener(_: TradeProposal) -> None:
        raise RuntimeError("boom in listener")

    manager.add_listener(_good_listener)
    manager.add_listener(_bad_listener)

    # 1. create_proposal
    setup = make_test_setup("prop_lst_1")
    p1 = manager.create_proposal(setup=setup, mode=ExecutionMode.ANALYST)
    assert p1.status == ProposalStatus.ADVISORY
    assert len(received) == 1
    assert received[-1].id == p1.id

    # 2. create pending
    p2 = manager.create_proposal(setup=make_test_setup("prop_lst_2"), mode=ExecutionMode.SEMIAUTO)
    assert len(received) == 2

    # 3. approve
    manager.approve_proposal(p2.id)
    assert len(received) == 3
    assert received[2].status == ProposalStatus.APPROVED

    # 4. mark_dispatched
    manager.mark_dispatched(p2.id, order_ticket=999)
    assert len(received) == 4
    assert received[3].status == ProposalStatus.DISPATCHED

    # 5. mark_dispatch_failed
    p3 = manager.create_proposal(setup=make_test_setup("prop_lst_3"), mode=ExecutionMode.SEMIAUTO)
    manager.mark_dispatch_failed(p3.id, reason="Broker reject")
    assert len(received) == 6  # +1 create, +1 dispatch_failed
    assert received[5].status == ProposalStatus.DISPATCH_FAILED

    # 6. reject
    p4 = manager.create_proposal(setup=make_test_setup("prop_lst_4"), mode=ExecutionMode.SEMIAUTO)
    manager.reject_proposal(p4.id, reason="Manual skip")
    assert len(received) == 8  # +1 create, +1 reject
    assert received[7].status == ProposalStatus.REJECTED

    # 7. check_expirations
    t0 = datetime.now(UTC) - timedelta(seconds=20)
    p5 = manager.create_proposal(setup=make_test_setup("prop_lst_5"), now=t0)
    assert len(received) == 9
    exp = manager.check_expirations()
    assert p5 in exp
    assert len(received) == 10
    assert received[9].status == ProposalStatus.EXPIRED

    # 8. cancel_invalidated
    p6 = manager.create_proposal(setup=make_test_setup("prop_lst_6"), mode=ExecutionMode.SEMIAUTO)
    canc = manager.cancel_invalidated(p6.setup_id, reason="poi_mitigated")
    assert p6 in canc
    assert received[11].status == ProposalStatus.CANCELLED

    # 9. cancel_by_price
    p7 = manager.create_proposal(setup=make_test_setup("prop_lst_7"), mode=ExecutionMode.SEMIAUTO)
    canc_price = manager.cancel_by_price(symbol=p7.symbol, current_price=p7.stop_loss - 1.0)
    assert p7 in canc_price
    assert received[13].status == ProposalStatus.CANCELLED

    stats = manager.get_stats()
    assert stats["advisory"] >= 1
    assert stats["dispatch_failed"] >= 1
    assert stats["cancelled"] >= 2
    assert stats["total"] >= 7


@pytest.mark.asyncio
async def test_advisor_dispatcher_telegram_service_and_alert_fallbacks() -> None:
    manager = ProposalManager()
    mock_tel = AsyncMock()
    mock_tel.send_proposal.return_value = True

    # 1. telegram service success async
    dispatcher = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.SEMIAUTO,
        telegram_service=mock_tel,
    )
    s = make_test_setup("setup_tel_1")
    prop = await dispatcher.handle_setup(s)
    mock_tel.send_proposal.assert_awaited_once_with(prop)

    # 2. telegram service sync & raising error
    sync_tel = MagicMock()
    sync_tel.send_proposal.side_effect = RuntimeError("network fail")
    dispatcher_sync = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.SEMIAUTO,
        telegram_service=sync_tel,
    )
    prop2 = await dispatcher_sync.handle_setup(make_test_setup("setup_tel_2"))
    assert prop2 is not None

    # 3. alert_manager with emit() fallback
    emit_alert = MagicMock(spec=["emit"])
    dispatcher_emit = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.ANALYST,
        alert_manager=emit_alert,
    )
    await dispatcher_emit.handle_setup(make_test_setup("setup_emit_1"))
    emit_alert.emit.assert_called_once()

    # 4. alert_manager raising error
    failing_alert = MagicMock()
    failing_alert.notify.side_effect = RuntimeError("alert down")
    dispatcher_alert_err = AdvisorDispatcher(
        proposal_manager=manager,
        mode=ExecutionMode.ANALYST,
        alert_manager=failing_alert,
    )
    prop3 = await dispatcher_alert_err.handle_setup(make_test_setup("setup_alert_err"))
    assert prop3 is not None


@pytest.mark.asyncio
async def test_telegram_service_send_proposal(monkeypatch: pytest.MonkeyPatch) -> None:
    from xau_bot.notify.models import TelegramChat, TelegramMessage
    from xau_bot.notify.telegram import (
        TelegramBotClient,
        TelegramCommandHandler,
        TelegramService,
    )

    client = TelegramBotClient("123456:fake_token", "123456789")
    handler = TelegramCommandHandler(bot_client=client, authorized_chat_id="123456789")
    svc = TelegramService(client, handler)

    fake_msg = TelegramMessage(
        message_id=42,
        date=1000,
        chat=TelegramChat(id=123456789, type="private"),
    )
    mock_send = AsyncMock(return_value=fake_msg)
    monkeypatch.setattr(client, "send_message", mock_send)

    manager = ProposalManager()
    setup = make_test_setup("tel_format_1")
    proposal = manager.create_proposal(setup=setup, mode=ExecutionMode.SEMIAUTO)

    res = await svc.send_proposal(proposal)
    assert res is fake_msg
    mock_send.assert_awaited_once()
    kwargs = mock_send.call_args.kwargs
    assert "PROPOSTA DE TRADE PENDENTE" in kwargs["text"]
    assert kwargs["reply_markup"] is not None
    buttons = kwargs["reply_markup"].inline_keyboard[0]
    assert len(buttons) == 2
    assert "Executar" in buttons[0].text
    assert f"approve:{proposal.id}" == buttons[0].callback_data
    assert "Rejeitar" in buttons[1].text
    assert f"reject:{proposal.id}" == buttons[1].callback_data


@pytest.mark.asyncio
async def test_advisor_dispatcher_telegram_service_integration() -> None:
    pm = ProposalManager()
    mock_service = AsyncMock()
    mock_service.send_proposal = AsyncMock(return_value=None)

    dispatcher = AdvisorDispatcher(
        proposal_manager=pm,
        mode=ExecutionMode.SEMIAUTO,
        telegram_service=mock_service,
    )
    setup = make_test_setup("setup_telegram_int")
    prop = await dispatcher.handle_setup(setup, atr=1.85)

    assert prop is not None
    assert prop.atr == 1.85
    mock_service.send_proposal.assert_awaited_once_with(prop)
