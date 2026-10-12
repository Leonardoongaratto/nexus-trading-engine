import json
import math
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from xau_bot.filters.models import NewsFilterConfig, NewsImpact
from xau_bot.filters.news import (
    NewsFilter,
    load_calendar_from_file,
    load_calendar_payload_from_file,
    parse_calendar_payload,
)
from xau_bot.filters.sessions import TradeBlockReason
from xau_bot.guardian.bridge import GuardianBridge
from xau_bot.guardian.heartbeat import HeartbeatReader, HeartbeatWriter
from xau_bot.guardian.models import (
    GuardianConfig,
    GuardianKillSwitchReason,
    GuardianSignal,
    HeartbeatPayload,
)
from xau_bot.risk.guards import RiskGuard
from xau_bot.risk.models import AccountState, DailyRiskLimits


class TestGuardianModels:
    def test_heartbeat_payload_valid(self) -> None:
        now = datetime(2026, 10, 3, 22, 30, tzinfo=UTC)
        payload = HeartbeatPayload(
            timestamp=now,
            status="ok",
            pid=1234,
            sequence=1,
            trading_halted=False,
        )
        assert payload.status == "ok"
        assert payload.pid == 1234
        assert payload.sequence == 1
        assert not payload.trading_halted

        d = payload.to_dict()
        assert d["pid"] == 1234
        assert d["sequence"] == 1
        assert d["status"] == "ok"
        assert d["trading_halted"] is False
        assert "2026-10-03T22:30:00" in d["timestamp"]

        payload_json = payload.to_json()
        restored = HeartbeatPayload.from_json(payload_json)
        assert restored.pid == payload.pid
        assert restored.sequence == payload.sequence
        assert restored.status == payload.status
        assert restored.timestamp == payload.timestamp
        assert restored.trading_halted == payload.trading_halted

    def test_heartbeat_payload_from_dict_and_json_edge_cases(self) -> None:
        dt_naive = datetime(2026, 10, 3, 12, 0)
        d_with_dt = {
            "timestamp": dt_naive,
            "status": "ok",
            "pid": 100,
            "sequence": 5,
        }
        p = HeartbeatPayload.from_dict(d_with_dt)
        assert p.timestamp.tzinfo == UTC
        assert p.pid == 100

        with pytest.raises(ValueError, match="JSON must decode to a dictionary"):
            HeartbeatPayload.from_json("[]")

    def test_heartbeat_payload_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            HeartbeatPayload(
                timestamp=datetime(2026, 10, 3, 22, 0),
                status="ok",
                pid=123,
                sequence=0,
                trading_halted=False,
            )

        now = datetime.now(UTC)
        with pytest.raises(ValueError, match="sequence must be non-negative"):
            HeartbeatPayload(
                timestamp=now,
                status="ok",
                pid=123,
                sequence=-1,
                trading_halted=False,
            )

        with pytest.raises(ValueError, match="pid must be positive"):
            HeartbeatPayload(
                timestamp=now,
                status="ok",
                pid=0,
                sequence=0,
                trading_halted=False,
            )

        with pytest.raises(ValueError, match="status must not be empty"):
            HeartbeatPayload(
                timestamp=now,
                status="   ",
                pid=10,
                sequence=0,
                trading_halted=False,
            )

    def test_guardian_signal_valid_and_serialization(self) -> None:
        now = datetime(2026, 10, 3, 22, 30, tzinfo=UTC)
        sig = GuardianSignal(
            timestamp=now,
            reason=GuardianKillSwitchReason.DAILY_LOSS_BREACHED,
            current_loss_pct=3.45,
            max_loss_pct=3.0,
            positions_closed=2,
            orders_cancelled=1,
            details="Emergency stop triggered by EA",
        )
        assert sig.reason == GuardianKillSwitchReason.DAILY_LOSS_BREACHED
        assert sig.positions_closed == 2

        d = sig.to_dict()
        assert d["reason"] == "DAILY_LOSS_BREACHED"
        assert d["current_loss_pct"] == 3.45

        sig_json = sig.to_json()
        restored = GuardianSignal.from_json(sig_json)
        assert restored.timestamp == now
        assert restored.reason == GuardianKillSwitchReason.DAILY_LOSS_BREACHED
        assert restored.current_loss_pct == 3.45
        assert restored.positions_closed == 2

    def test_guardian_signal_with_string_reason_and_datetime(self) -> None:
        now = datetime(2026, 10, 3, 22, 30, tzinfo=UTC)
        sig = GuardianSignal(
            timestamp=now,
            reason="CUSTOM_VOLATILITY_HALT",
        )
        d = sig.to_dict()
        assert d["reason"] == "CUSTOM_VOLATILITY_HALT"

        d_naive = {
            "timestamp": datetime(2026, 10, 3, 22, 30),
            "reason": "CUSTOM_ALERT",
        }
        sig_restored = GuardianSignal.from_dict(d_naive)
        assert sig_restored.timestamp.tzinfo == UTC
        assert sig_restored.reason == "CUSTOM_ALERT"

        with pytest.raises(ValueError, match="JSON must decode to a dictionary"):
            GuardianSignal.from_json('"some_string"')

    def test_guardian_signal_validation_errors(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            GuardianSignal(
                timestamp=datetime(2026, 10, 3, 22, 0),
                reason=GuardianKillSwitchReason.DAILY_LOSS_BREACHED,
            )

        now = datetime.now(UTC)
        with pytest.raises(ValueError, match="positions_closed cannot be negative"):
            GuardianSignal(
                timestamp=now,
                reason="CUSTOM",
                positions_closed=-1,
            )

        with pytest.raises(ValueError, match="orders_cancelled cannot be negative"):
            GuardianSignal(
                timestamp=now,
                reason="CUSTOM",
                orders_cancelled=-1,
            )

        with pytest.raises(ValueError, match="current_loss_pct must be finite"):
            GuardianSignal(
                timestamp=now,
                reason="CUSTOM",
                current_loss_pct=math.nan,
            )

        with pytest.raises(ValueError, match="max_loss_pct must be finite"):
            GuardianSignal(
                timestamp=now,
                reason="CUSTOM",
                max_loss_pct=math.inf,
            )

    def test_guardian_config_validation(self, tmp_path: Path) -> None:
        cfg = GuardianConfig(common_dir=tmp_path)
        assert isinstance(cfg.common_dir, Path)
        assert cfg.heartbeat_filename == "xau_heartbeat.json"
        assert cfg.magic_number == 202610

        with pytest.raises(ValueError, match="heartbeat_interval_s must be positive"):
            GuardianConfig(common_dir=tmp_path, heartbeat_interval_s=0.0)

        with pytest.raises(ValueError, match="heartbeat_timeout_s must be positive"):
            GuardianConfig(common_dir=tmp_path, heartbeat_timeout_s=-5.0)

        with pytest.raises(ValueError, match="calendar_max_age_h must be positive"):
            GuardianConfig(common_dir=tmp_path, calendar_max_age_h=math.nan)

        with pytest.raises(ValueError, match="magic_number must be positive"):
            GuardianConfig(common_dir=tmp_path, magic_number=0)


class TestHeartbeatWriterAndReader:
    def test_writer_creates_atomic_file_with_sequence(self, tmp_path: Path) -> None:
        writer = HeartbeatWriter(common_dir=tmp_path, filename="xau_heartbeat.json", pid=9999)
        reader = HeartbeatReader(common_dir=tmp_path, filename="xau_heartbeat.json", timeout_s=10.0)

        hb_path = tmp_path / "xau_heartbeat.json"
        assert not hb_path.exists()

        p1 = writer.write(status="ok", trading_halted=False)
        assert p1.sequence == 1
        assert p1.pid == 9999
        assert hb_path.exists()

        read_p1 = reader.read()
        assert read_p1 is not None
        assert read_p1.sequence == 1
        assert read_p1.status == "ok"
        assert read_p1.pid == 9999
        assert read_p1.trading_halted is False

        p2 = writer.write(status="halted", trading_halted=True)
        assert p2.sequence == 2
        read_p2 = reader.read()
        assert read_p2 is not None
        assert read_p2.sequence == 2
        assert read_p2.status == "halted"
        assert read_p2.trading_halted is True

    def test_writer_naive_datetime_raises_error(self, tmp_path: Path) -> None:
        writer = HeartbeatWriter(common_dir=tmp_path)
        with pytest.raises(ValueError, match="timezone-aware"):
            writer.write(now=datetime(2026, 10, 3, 10, 0))

    def test_writer_retry_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        writer = HeartbeatWriter(common_dir=tmp_path)

        def mock_replace(src: Path, dst: Path) -> None:
            raise PermissionError("Access denied")

        monkeypatch.setattr("os.replace", mock_replace)
        with pytest.raises(PermissionError):
            writer.write()

    def test_reader_is_alive_states(self, tmp_path: Path) -> None:
        reader = HeartbeatReader(common_dir=tmp_path, filename="xau_heartbeat.json", timeout_s=15.0)

        alive, msg = reader.is_alive()
        assert not alive
        assert "HEARTBEAT_MISSING" in msg

        writer = HeartbeatWriter(common_dir=tmp_path, filename="xau_heartbeat.json", pid=100)
        t0 = datetime(2026, 10, 3, 20, 0, 0, tzinfo=UTC)
        writer.write(now=t0)

        t_fresh = datetime(2026, 10, 3, 20, 0, 10, tzinfo=UTC)
        alive, msg = reader.is_alive(now=t_fresh)
        assert alive
        assert msg == "OK"

        t_expired = datetime(2026, 10, 3, 20, 0, 25, tzinfo=UTC)
        alive, msg = reader.is_alive(now=t_expired)
        assert not alive
        assert "HEARTBEAT_TIMEOUT" in msg

        with pytest.raises(ValueError, match="timezone-aware"):
            reader.is_alive(now=datetime(2026, 10, 3, 20, 0, 0))

        hb_file = tmp_path / "xau_heartbeat.json"
        hb_file.write_text("{corrupt json", encoding="utf-8")
        alive, msg = reader.is_alive(now=t_fresh)
        assert not alive
        assert "HEARTBEAT_CORRUPT" in msg
        assert reader.read() is None


class TestGuardianBridge:
    def test_bridge_pulse_and_signal_monitoring(self, tmp_path: Path) -> None:
        cfg = GuardianConfig(
            common_dir=tmp_path,
            heartbeat_interval_s=1.0,
            heartbeat_timeout_s=10.0,
            magic_number=202610,
        )
        limits = DailyRiskLimits(
            max_daily_loss_usd=300.0,
            max_daily_loss_percent=0.03,
            max_daily_drawdown_usd=500.0,
            max_daily_drawdown_percent=0.05,
            max_consecutive_losses=3,
        )
        risk_guard = RiskGuard(limits=limits)
        now_init = datetime(2026, 10, 3, 14, 0, tzinfo=UTC)
        risk_guard.initialize_day(
            now_init,
            AccountState(
                balance=10000.0,
                equity=10000.0,
                margin=0.0,
                free_margin=10000.0,
                leverage=100.0,
            ),
        )
        bridge = GuardianBridge(config=cfg, risk_guard=risk_guard)

        assert not risk_guard.state.halted
        p = bridge.pulse(status="ok", trading_halted=False)
        assert p.sequence == 1

        sig = bridge.check_guardian_signal()
        assert sig is None

        killswitch_file = tmp_path / cfg.killswitch_filename
        killswitch_payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "reason": "DAILY_LOSS_BREACHED",
            "current_loss_pct": 3.2,
            "max_loss_pct": 3.0,
            "positions_closed": 1,
            "orders_cancelled": 0,
            "details": "Triggered by XauGuardian EA",
        }
        killswitch_file.write_text(json.dumps(killswitch_payload), encoding="utf-8")

        detected_sig = bridge.check_guardian_signal()
        assert detected_sig is not None
        assert detected_sig.reason == GuardianKillSwitchReason.DAILY_LOSS_BREACHED
        assert detected_sig.current_loss_pct == 3.2
        assert risk_guard.state.halted

        cleared = bridge.clear_guardian_signal()
        assert cleared
        assert not killswitch_file.exists()

    def test_bridge_corrupt_signal_and_clear_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = GuardianConfig(common_dir=tmp_path)
        bridge = GuardianBridge(config=cfg)

        assert not bridge.clear_guardian_signal()

        killswitch_file = tmp_path / cfg.killswitch_filename
        killswitch_file.write_text("corrupted json content", encoding="utf-8")
        assert bridge.check_guardian_signal() is None

        def mock_unlink(self: Path, missing_ok: bool = False) -> None:
            raise OSError("Permission denied")

        monkeypatch.setattr(Path, "unlink", mock_unlink)
        assert not bridge.clear_guardian_signal()

    def test_bridge_start_and_stop_worker(self, tmp_path: Path) -> None:
        cfg = GuardianConfig(common_dir=tmp_path, heartbeat_interval_s=0.05)
        bridge = GuardianBridge(config=cfg)

        bridge.start()
        bridge.start()
        assert bridge._thread is not None and bridge._thread.is_alive()

        hb_file = tmp_path / cfg.heartbeat_filename
        for _ in range(50):
            if hb_file.exists():
                break

        bridge.stop()
        bridge.stop()
        assert bridge._thread is None

    def test_bridge_worker_with_risk_guard_and_exception(self, tmp_path: Path) -> None:
        import time

        cfg = GuardianConfig(common_dir=tmp_path, heartbeat_interval_s=0.02)
        limits = DailyRiskLimits(max_daily_loss_usd=500.0)
        risk_guard = RiskGuard(limits=limits)
        risk_guard.initialize_day(
            datetime.now(UTC),
            AccountState(
                balance=10000.0,
                equity=10000.0,
                margin=0.0,
                free_margin=10000.0,
                leverage=100.0,
            ),
        )
        bridge = GuardianBridge(config=cfg, risk_guard=risk_guard)
        bridge.start()
        time.sleep(0.05)
        bridge.stop()

        exploding_writer = MagicMock()
        exploding_writer.write.side_effect = RuntimeError("Pulse error")
        bridge_err = GuardianBridge(config=cfg)
        object.__setattr__(bridge_err, "writer", exploding_writer)
        bridge_err.start()
        time.sleep(0.05)
        bridge_err.stop()


class TestCalendarStaleness:
    def test_parse_calendar_payload_with_generated_at(self) -> None:
        payload = {
            "generated_at": "2026-10-03T12:00:00Z",
            "events": [
                {
                    "title": "US Non-Farm Payrolls",
                    "currency": "USD",
                    "impact": "HIGH",
                    "time": "2026-10-03T12:30:00Z",
                    "actual": None,
                    "forecast": 150.0,
                    "previous": 142.0,
                }
            ],
        }
        events, gen_at = parse_calendar_payload(payload)
        assert len(events) == 1
        assert events[0].title == "US Non-Farm Payrolls"
        assert gen_at == datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)

    def test_parse_calendar_payload_invalid_structure(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON array or object with 'events'"):
            parse_calendar_payload('"just a string"')

        with pytest.raises(ValueError, match="Expected JSON array or object with 'events'"):
            parse_calendar_payload({"generated_at": "2026-10-03T12:00:00Z"})

        with pytest.raises(ValueError, match="unsupported generated_at format"):
            parse_calendar_payload({"generated_at": 12345, "events": []})

        with pytest.raises(ValueError, match="generated_at must be timezone-aware"):
            parse_calendar_payload({"generated_at": "2026-10-03T12:00:00", "events": []})

    def test_load_calendar_payload_from_file(self, tmp_path: Path) -> None:
        f = tmp_path / "calendar.json"
        payload = {
            "generated_at": "2026-10-03T12:00:00Z",
            "events": [
                {
                    "title": "FOMC Statement",
                    "currency": "USD",
                    "impact": "HIGH",
                    "time": "2026-10-03T18:00:00Z",
                }
            ],
        }
        f.write_text(json.dumps(payload), encoding="utf-8")

        events, gen_at = load_calendar_payload_from_file(f)
        assert len(events) == 1
        assert gen_at == datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)

        events_compat = load_calendar_from_file(f)
        assert len(events_compat) == 1

        with pytest.raises(FileNotFoundError):
            load_calendar_payload_from_file(tmp_path / "missing.json")

    def test_load_calendar_with_utf8_accents(self, tmp_path: Path) -> None:
        f = tmp_path / "calendar_pt.json"
        payload = {
            "generated_at": "2026-10-03T12:00:00Z",
            "events": [
                {
                    "title": "Índice de Preços ao Consumidor (IPC) & Inflação",
                    "currency": "EUR",
                    "impact": "HIGH",
                    "time": "2026-10-03T14:00:00Z",
                }
            ],
        }
        f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        events, gen_at = load_calendar_payload_from_file(f)
        assert len(events) == 1
        assert "Índice" in events[0].title
        assert "Inflação" in events[0].title

    def test_news_filter_blocks_stale_calendar(self) -> None:
        config = NewsFilterConfig(
            max_age_hours=24.0,
            min_impact=NewsImpact.HIGH,
            currencies=("USD",),
        )
        filter_instance = NewsFilter(config=config)

        t_gen = datetime(2026, 10, 1, 10, 0, 0, tzinfo=UTC)
        filter_instance.set_events([], generated_at=t_gen)

        current_time = datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)
        result = filter_instance.check(current_time)
        assert not result.can_trade
        assert result.reason == TradeBlockReason.NEWS_CALENDAR_STALE
        assert "stale" in result.details.lower()

        fresh_time = datetime(2026, 10, 1, 15, 0, 0, tzinfo=UTC)
        fresh_res = filter_instance.check(fresh_time)
        assert fresh_res.can_trade
        assert fresh_res.reason is None

        with pytest.raises(ValueError, match="timezone-aware"):
            filter_instance.set_events([], generated_at=datetime(2026, 10, 1, 10, 0))

        with pytest.raises(ValueError, match="timezone-aware"):
            filter_instance.check(datetime(2026, 10, 3, 12, 0))

        with pytest.raises(ValueError, match="timezone-aware"):
            filter_instance.is_blackout(datetime(2026, 10, 3, 12, 0))

    def test_news_filter_blocks_fail_closed_when_generated_at_is_missing(self) -> None:
        config = NewsFilterConfig(
            max_age_hours=24.0,
            min_impact=NewsImpact.HIGH,
            currencies=("USD",),
        )
        filter_instance = NewsFilter(config=config)
        filter_instance.set_events([], generated_at=None)

        current_time = datetime(2026, 10, 3, 12, 0, 0, tzinfo=UTC)
        result = filter_instance.check(current_time)
        assert not result.can_trade
        assert result.reason == TradeBlockReason.NEWS_CALENDAR_STALE
        assert "fail-closed" in result.details.lower() or "missing" in result.details.lower()


class TestGuardianRegressionFixes:
    def test_risk_guard_has_state_and_bridge_halt_initialization(self, tmp_path: Path) -> None:
        rg = RiskGuard(limits=DailyRiskLimits())
        assert not rg.has_state
        assert not rg.is_halted
        with pytest.raises(RuntimeError, match="RiskGuard has not been initialized"):
            rg.trigger_manual_halt("EMERGENCY_HALT")

        # Bridge inicializa o dia e aplica halt quando o sinal chega
        cfg = GuardianConfig(common_dir=tmp_path)
        bridge = GuardianBridge(config=cfg, risk_guard=rg)
        sig_file = tmp_path / "xau_killswitch.json"
        sig_data = {
            "timestamp": "2026-10-03T23:00:00Z",
            "reason": "DAILY_LOSS_BREACHED",
            "current_loss_pct": 3.5,
            "max_loss_pct": 3.0,
            "positions_closed": 1,
            "orders_cancelled": 2,
        }
        sig_file.write_text(json.dumps(sig_data), encoding="utf-8")
        signal = bridge.check_guardian_signal()
        assert signal is not None
        assert rg.has_state
        assert rg.is_halted

    def test_heartbeat_reader_is_alive_reusing_read(self, tmp_path: Path) -> None:
        reader = HeartbeatReader(common_dir=tmp_path, timeout_s=10.0)
        alive, reason = reader.is_alive()
        assert not alive
        assert reason == "HEARTBEAT_MISSING"

        # Escreve arquivo corrompido
        (tmp_path / "xau_heartbeat.json").write_text("corrupted json", encoding="utf-8")
        alive, reason = reader.is_alive()
        assert not alive
        assert reason == "HEARTBEAT_CORRUPT"

    def test_heartbeat_writer_retry_transient_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        writer = HeartbeatWriter(common_dir=tmp_path)
        calls = 0
        orig_replace = os.replace

        def flaky_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise PermissionError("Access denied (transient)")
            orig_replace(src, dst)

        monkeypatch.setattr(os, "replace", flaky_replace)
        payload = writer.write(status="ok")
        assert payload.status == "ok"
        assert calls == 2
        assert writer.target_file.is_file()

    def test_runner_with_guardian_bridge_integration(self, tmp_path: Path) -> None:
        import time

        from xau_bot.broker.sim_adapter import SimBrokerAdapter
        from xau_bot.runtime.runner import WatchdogRunner
        from xau_bot.runtime.watchdog import Watchdog
        from xau_bot.server import HealthStatusState

        broker = SimBrokerAdapter()
        watchdog = Watchdog(broker=broker)
        cfg = GuardianConfig(common_dir=tmp_path, heartbeat_interval_s=0.05)
        rg = RiskGuard(limits=DailyRiskLimits())
        bridge = GuardianBridge(config=cfg, risk_guard=rg, watchdog=watchdog)

        state = HealthStatusState()
        runner = WatchdogRunner(
            watchdog=watchdog,
            target=state,
            interval_s=0.05,
            guardian_bridge=bridge,
        )
        runner.start()
        time.sleep(0.15)

        # 1. Verifica que heartbeat foi gravado em disco pelo runner
        hb_file = tmp_path / "xau_heartbeat.json"
        assert hb_file.is_file()
        reader = HeartbeatReader(common_dir=tmp_path)
        alive, reason = reader.is_alive()
        assert alive
        assert reason == "OK"
        assert not state.trading_halted

        # 2. Injeta sinal de kill switch manual no Common Files
        signal_file = tmp_path / "xau_killswitch.json"
        sig_data = {
            "timestamp": "2026-10-03T23:00:00Z",
            "reason": "DAILY_LOSS_BREACHED",
            "current_loss_pct": 3.5,
            "max_loss_pct": 3.0,
            "positions_closed": 1,
            "orders_cancelled": 2,
        }
        signal_file.write_text(json.dumps(sig_data), encoding="utf-8")

        # Aguarda ciclo do runner processar o sinal
        time.sleep(0.15)
        assert state.trading_halted
        assert state.kill_switches_active
        assert rg.is_halted

        runner.stop()

    def test_check_guardian_signal_deduplication_log_and_reapply(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        cfg = GuardianConfig(common_dir=tmp_path)
        rg = RiskGuard(limits=DailyRiskLimits())
        bridge = GuardianBridge(config=cfg, risk_guard=rg)

        sig_file = tmp_path / "xau_killswitch.json"
        sig_data = {
            "timestamp": "2026-10-03T23:30:00Z",
            "reason": "DAILY_LOSS_BREACHED",
            "current_loss_pct": 3.2,
            "max_loss_pct": 3.0,
            "positions_closed": 1,
            "orders_cancelled": 0,
        }
        sig_file.write_text(json.dumps(sig_data), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            # Primeira chamada: deve emitir WARNING e aplicar o halt
            s1 = bridge.check_guardian_signal()
            assert s1 is not None
            assert rg.is_halted

            # Segunda chamada com o mesmo arquivo e timestamp: nao deve emitir novo WARNING
            caplog.clear()
            s2 = bridge.check_guardian_signal()
            assert s2 is not None
            assert not any(
                "Guardian kill switch signal detected" in r.message for r in caplog.records
            )
