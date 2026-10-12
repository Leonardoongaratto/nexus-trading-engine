import sys
from zoneinfo import ZoneInfo

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from xau_bot.advisor.modes import ExecutionMode


class Settings(BaseSettings):
    app_name: str = "xau-bot"
    env: str = "development"
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    symbol: str = "XAUUSD"
    crypto_24_7: bool = False
    timeframe_entry: str = "M5"
    risk_per_trade_pct: float = Field(default=1.0, ge=0.1, le=5.0)
    max_daily_loss_pct: float = Field(default=3.0, ge=0.5, le=10.0)
    max_drawdown_pct: float = Field(default=10.0, ge=1.0, le=25.0)
    max_consecutive_losses: int = Field(default=3, ge=1, le=10)
    max_daily_trades: int = Field(default=5, ge=1, le=20)
    remove_tp_after_partial: bool = True
    telegram_token: SecretStr | None = None
    telegram_chat_id: SecretStr | None = None
    mt5_login: SecretStr | None = None
    mt5_server: str | None = None
    mt5_password: SecretStr | None = None
    mt5_path: str | None = None
    mt5_timeout_ms: int = Field(default=60000, ge=1000, le=300000)
    broker_server_tz: str = "UTC"
    guardian_common_dir: str | None = None
    guardian_heartbeat_interval_s: float = Field(default=5.0, ge=0.5, le=60.0)
    guardian_heartbeat_timeout_s: float = Field(default=30.0, ge=5.0, le=300.0)
    guardian_calendar_max_age_h: float = Field(default=24.0, ge=1.0, le=168.0)
    guardian_magic: int = Field(default=202610, ge=1)
    execution_mode: ExecutionMode = ExecutionMode.SEMIAUTO
    proposal_timeout_s: float = Field(default=300.0, ge=1.0, le=86400.0)
    journal_db: str = "journal.db"
    shadow_params_json: str | None = None
    llm_enabled: bool = False
    llm_api_key: SecretStr | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = Field(default=15.0, gt=0.0, le=15.0)
    ml_meta_labeling_enabled: bool = False
    ml_model_path: str | None = None
    ml_min_probability: float = Field(default=0.50, ge=0.0, le=1.0)
    setup_stats_path: str | None = None
    llm_veto_enabled: bool = False
    llm_veto_shadow_mode: bool = True
    llm_veto_model: str = "gpt-4o-mini-2024-07-18"
    llm_veto_timeout_seconds: float = Field(default=5.0, ge=0.5, le=30.0)

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, v: str) -> str:
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"Invalid llm_base_url scheme (must be http:// or https://): {v!r}")
        return v

    @field_validator("broker_server_tz")
    @classmethod
    def validate_broker_server_tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except Exception as e:
            raise ValueError(f"Invalid timezone: {v!r}") from e
        return v

    def get_trigger_timeframes(self) -> tuple[str, ...]:
        parts = [p.strip().upper() for p in self.timeframe_entry.split(",") if p.strip()]
        return tuple(parts) if parts else ("M5",)

    model_config = SettingsConfigDict(
        env_prefix="XAU_",
        env_file=None if "pytest" in sys.modules else ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
