import os
from collections.abc import Generator

import pytest
from pydantic import ValidationError

from xau_bot.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    for key in list(os.environ.keys()):
        if key.startswith("XAU_"):
            monkeypatch.delenv(key, raising=False)
    yield


def test_default_settings() -> None:
    settings = Settings()
    assert settings.app_name == "xau-bot"
    assert settings.server_host == "127.0.0.1"
    assert settings.server_port == 8000
    assert settings.symbol == "XAUUSD"
    assert settings.timeframe_entry == "M5"
    assert settings.risk_per_trade_pct == 1.0
    assert settings.max_daily_loss_pct == 3.0
    assert settings.max_drawdown_pct == 10.0
    assert settings.max_consecutive_losses == 3
    assert settings.max_daily_trades == 5
    assert settings.telegram_token is None
    assert settings.mt5_password is None
    assert settings.guardian_common_dir is None
    assert settings.guardian_heartbeat_interval_s == 5.0
    assert settings.guardian_heartbeat_timeout_s == 30.0
    assert settings.guardian_calendar_max_age_h == 24.0
    assert settings.guardian_magic == 202610


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAU_SYMBOL", "GOLD")
    monkeypatch.setenv("XAU_RISK_PER_TRADE_PCT", "1.5")
    monkeypatch.setenv("XAU_SERVER_PORT", "9000")
    monkeypatch.setenv("XAU_TELEGRAM_TOKEN", "secret-token-123")
    monkeypatch.setenv("XAU_MT5_LOGIN", "12345678")
    monkeypatch.setenv("XAU_MT5_PASSWORD", "super-secret-pwd")
    monkeypatch.setenv("XAU_GUARDIAN_COMMON_DIR", "C:\\MT5\\Common\\Files")
    monkeypatch.setenv("XAU_GUARDIAN_HEARTBEAT_INTERVAL_S", "2.5")
    monkeypatch.setenv("XAU_GUARDIAN_MAGIC", "999888")

    settings = Settings()
    assert settings.symbol == "GOLD"
    assert settings.risk_per_trade_pct == 1.5
    assert settings.server_port == 9000
    assert settings.telegram_token is not None
    assert settings.telegram_token.get_secret_value() == "secret-token-123"
    assert settings.mt5_login is not None
    assert settings.mt5_login.get_secret_value() == "12345678"
    assert settings.mt5_password is not None
    assert settings.mt5_password.get_secret_value() == "super-secret-pwd"
    assert settings.guardian_common_dir == "C:\\MT5\\Common\\Files"
    assert settings.guardian_heartbeat_interval_s == 2.5
    assert settings.guardian_magic == 999888

    settings_repr = repr(settings)
    assert "secret-token-123" not in settings_repr
    assert "12345678" not in settings_repr
    assert "super-secret-pwd" not in settings_repr


def test_settings_validation_errors() -> None:
    with pytest.raises(ValidationError):
        Settings(risk_per_trade_pct=0.01)

    with pytest.raises(ValidationError):
        Settings(max_daily_loss_pct=15.0)

    with pytest.raises(ValidationError):
        Settings(max_consecutive_losses=0)


def test_settings_broker_server_tz_valid() -> None:
    settings = Settings(broker_server_tz="Etc/GMT-2")
    assert settings.broker_server_tz == "Etc/GMT-2"


def test_settings_broker_server_tz_invalid_raises() -> None:
    with pytest.raises(ValidationError, match="Invalid timezone"):
        Settings(broker_server_tz="Mars/Base")


def test_settings_get_trigger_timeframes() -> None:
    settings_single = Settings(timeframe_entry="M5")
    assert settings_single.get_trigger_timeframes() == ("M5",)

    settings_multi = Settings(timeframe_entry="m1, M5, h1")
    assert settings_multi.get_trigger_timeframes() == ("M1", "M5", "H1")

    settings_empty = Settings(timeframe_entry="")
    assert settings_empty.get_trigger_timeframes() == ("M5",)
