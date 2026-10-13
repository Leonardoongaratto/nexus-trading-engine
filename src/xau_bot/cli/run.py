import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import TextIO

from xau_bot.advisor.modes import AdvisorDispatcher, ExecutionMode, ProposalManager
from xau_bot.advisor.stats import SetupStatsStore
from xau_bot.broker.ports import BrokerPort
from xau_bot.config import Settings
from xau_bot.domain.models import Timeframe
from xau_bot.execution.executor import OrderExecutor
from xau_bot.filters.composite import CompositeTradeFilter
from xau_bot.filters.models import NewsFilterConfig
from xau_bot.filters.news import NewsFilter
from xau_bot.filters.sessions import SessionFilter
from xau_bot.filters.spread import SpreadFilter
from xau_bot.journal import JournalRepository
from xau_bot.llm import HttpLLMClient, LLMVetoService
from xau_bot.notify.service import TelegramRunner, build_telegram_service
from xau_bot.risk.models import PositionRiskConfig
from xau_bot.runtime.live_loop import LiveTradingEngine, LiveTradingLoopRunner
from xau_bot.runtime.ops.alerts import (
    AlertChannel,
    AlertManager,
    ConsoleAlertChannel,
    TelegramAlertChannel,
)
from xau_bot.runtime.ops.logging import JsonLogFormatter, SensitiveDataFilter
from xau_bot.runtime.watchdog import Watchdog
from xau_bot.server import HealthStatusState, run_server

logger = logging.getLogger("xau_bot.cli.run")


def build_run_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--broker",
        choices=["sim", "mt5"],
        default="sim",
        help="Tipo de adaptador de broker (simulador determinístico ou MetaTrader 5)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host de bind do servidor HTTP de observabilidade (ex: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Porta do servidor HTTP de observabilidade (ex: 8000)",
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        help="Ativa formatação estruturada de logs em JSON RFC 8259",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nível de log (DEBUG, INFO, WARNING, ERROR)",
    )
    parser.add_argument(
        "--mode",
        choices=["analyst", "semiauto", "auto"],
        default=None,
        help="Modo de execução do robô (analyst, semiauto, auto)",
    )
    parser.add_argument(
        "--setup-stats",
        type=str,
        default=None,
        help="Caminho do arquivo JSON com estatísticas históricas dos setups (Portão 1)",
    )
    parser.add_argument(
        "--llm-veto",
        action="store_true",
        help="Ativa o serviço de veto qualitativo por LLM para avaliação de propostas",
    )
    parser.add_argument(
        "--llm-veto-active",
        action="store_true",
        help="Executa o veto por LLM em modo ativo (bloqueante) em vez do modo sombra",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Símbolo a operar no broker (ex: XAUUSD, BTCUSD, BTCUSDT)",
    )
    parser.add_argument(
        "--crypto-24-7",
        action="store_true",
        help="Habilita operação contínua 24/7 sem restrições de sessão ou fim de semana",
    )
    parser.add_argument(
        "--timeframe-entry",
        type=str,
        default=None,
        help="Timeframe(s) de gatilho de entrada separados por vírgula (ex: M1, M5, ou M1,M5)",
    )


def configure_logging(
    log_json: bool = False, log_level: str = "INFO", stream: TextIO | None = None
) -> None:
    target_stream = stream or sys.stderr
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    handler = logging.StreamHandler(target_stream)
    handler.setLevel(numeric_level)

    if log_json:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))

    handler.addFilter(SensitiveDataFilter())

    root_logger.handlers = [handler]


def create_broker_adapter(broker_type: str, settings: Settings) -> BrokerPort:
    if broker_type == "sim":
        from xau_bot.broker.sim_adapter import SimBrokerAdapter

        return SimBrokerAdapter()

    if broker_type == "mt5":
        from xau_bot.broker.mt5_adapter import Mt5AdapterConfig, Mt5BrokerAdapter

        try:
            cfg = Mt5AdapterConfig.from_settings(settings)
        except ValueError as e:
            if "mt5_login" in str(e):
                raise ValueError("MT5_LOGIN deve ser um identificador numérico válido") from None
            raise
        return Mt5BrokerAdapter(config=cfg)

    raise ValueError(f"Adaptador de broker não suportado: {broker_type}")


def create_alert_manager(
    settings: Settings,
    stdout: TextIO,
    enable_telegram: bool = True,
) -> AlertManager:
    channels: list[AlertChannel] = [ConsoleAlertChannel(stream=stdout)]

    if enable_telegram and settings.telegram_token and settings.telegram_chat_id:
        channels.append(
            TelegramAlertChannel(
                token=settings.telegram_token.get_secret_value(),
                chat_id=settings.telegram_chat_id.get_secret_value(),
            )
        )

    return AlertManager(channels=channels)


def run_bot_cli(
    args: argparse.Namespace,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr

    configure_logging(
        log_json=getattr(args, "log_json", False),
        log_level=getattr(args, "log_level", "INFO"),
        stream=err,
    )

    settings = Settings()
    if getattr(args, "mode", None) is not None:
        settings.execution_mode = ExecutionMode(args.mode)

    if args.host is not None:
        settings.server_host = args.host
    if args.port is not None:
        settings.server_port = args.port

    shadow_engine = None
    shadow_params = None
    if settings.shadow_params_json:
        try:
            raw_shadow = json.loads(settings.shadow_params_json)
        except json.JSONDecodeError as e:
            logger.error("JSON inválido em shadow_params_json: %s", e)
            err.write(f"Erro: JSON inválido em shadow_params_json: {e}\n")
            return 2

        from xau_bot.cli.params import validate_param_dict

        is_valid, err_msg = validate_param_dict(raw_shadow)
        if not is_valid:
            logger.error("Especificação inválida em shadow_params_json: %s", err_msg)
            err.write(f"Erro: Parâmetros de sombra inválidos: {err_msg}\n")
            return 2

        supported_shadow_keys = {"max_spread_atr_ratio", "min_displacement_multiplier"}
        unsupported = set(raw_shadow.keys()) - supported_shadow_keys
        if unsupported:
            err_msg = f"Chaves não suportadas para execução em sombra: {sorted(unsupported)}"
            logger.error(err_msg)
            err.write(f"Erro: {err_msg}\n")
            return 2

        shadow_params = raw_shadow
        from xau_bot.learning.shadow import ShadowEngine

        shadow_engine = ShadowEngine()

    if settings.server_host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            "Servidor de observabilidade bindado em host não-loopback (%s). "
            "Endpoints de métricas podem estar acessíveis externamente.",
            settings.server_host,
        )

    raw_symbol = getattr(args, "symbol", None) or settings.symbol
    symbols = [s.strip() for s in raw_symbol.split(",") if s.strip()]
    if not symbols:
        symbols = ["XAUUSD"]

    def _is_crypto_symbol(sym: str) -> bool:
        s = sym.upper()
        return s.startswith(("BTC", "ETH", "SOL")) or "BTC" in s or "ETH" in s

    any_is_crypto = any(_is_crypto_symbol(s) for s in symbols)
    any_enable_24_7 = (
        bool(getattr(args, "crypto_24_7", False)) or settings.crypto_24_7 or any_is_crypto
    )

    raw_tf = getattr(args, "timeframe_entry", None) or settings.timeframe_entry
    trigger_tfs: list[Timeframe] = []
    for part in raw_tf.split(","):
        p = part.strip().upper()
        if p:
            try:
                trigger_tfs.append(Timeframe(p))
            except ValueError:
                logger.warning("Timeframe desconhecido: %s. Ignorando.", p)
    if not trigger_tfs:
        trigger_tfs = [Timeframe.M5]

    logger.info(
        "Inicializando xau-bot run com broker=%s, symbols=%s, timeframes=%s (24/7=%s) em %s:%d",
        args.broker,
        symbols,
        [tf.value for tf in trigger_tfs],
        any_enable_24_7,
        settings.server_host,
        settings.server_port,
    )

    broker = create_broker_adapter(args.broker, settings)
    watchdog_session_filter = SessionFilter(
        server_tz=settings.broker_server_tz,
        enable_24_7=any_enable_24_7,
    )
    spread_filter = SpreadFilter()
    news_filter: NewsFilter | None = None
    if settings.guardian_common_dir:
        cal_path = Path(settings.guardian_common_dir) / "calendar.json"
        cfg = NewsFilterConfig(max_age_hours=settings.guardian_calendar_max_age_h)
        news_filter = NewsFilter(config=cfg)
        if cal_path.is_file():
            try:
                news_filter.load_events_from_file(cal_path)
            except Exception as e:
                logger.warning("Falha ao carregar calendário econômico de %s: %s", cal_path, e)
        else:
            logger.warning(
                "Calendário econômico não encontrado em %s (NewsFilter em modo fail-closed)",
                cal_path,
            )
    else:
        if settings.execution_mode == ExecutionMode.AUTO and not any_enable_24_7:
            cfg = NewsFilterConfig(max_age_hours=settings.guardian_calendar_max_age_h)
            news_filter = NewsFilter(config=cfg)
            logger.warning(
                "Aviso: guardian_common_dir não configurado no modo auto. "
                "NewsFilter operando em modo fail-closed por segurança."
            )
        else:
            news_filter = NewsFilter()

    watchdog = Watchdog(broker=broker, session_filter=watchdog_session_filter)
    alert_manager = create_alert_manager(settings, stdout=out, enable_telegram=True)

    try:
        asyncio.run(broker.connect())
    except Exception as conn_err:
        logger.warning(
            "Conexão inicial com broker %s falhou (modo resiliente ativado): %s",
            args.broker,
            conn_err,
        )

    order_executor = OrderExecutor(broker=broker)
    stats_path = getattr(args, "setup_stats", None) or settings.setup_stats_path
    stats_store: SetupStatsStore | None = None
    if stats_path:
        path_obj = Path(stats_path)
        if not path_obj.exists():
            logger.warning("Arquivo de estatísticas de setup não encontrado: %s", stats_path)
        stats_store = SetupStatsStore.load_from_file(path_obj)
        logger.info("SetupStatsStore inicializado a partir de %s", stats_path)

    veto_service: LLMVetoService | None = None
    llm_veto_active = bool(getattr(args, "llm_veto_active", False))
    llm_veto_flag = (
        bool(getattr(args, "llm_veto", False)) or llm_veto_active or settings.llm_veto_enabled
    )
    shadow_mode = False if llm_veto_active else settings.llm_veto_shadow_mode

    if llm_veto_flag:
        if not settings.llm_api_key:
            if not shadow_mode:
                logger.error(
                    "Modo LLM Veto ativo configurado, mas LLM_API_KEY está ausente. "
                    "Abortando por segurança para evitar execução desprotegida."
                )
                print(
                    "Erro: Modo LLM Veto ativo exige LLM_API_KEY configurada.",
                    file=sys.stderr,
                )
                return 2
            logger.warning(
                "LLM Veto habilitado em modo sombra, mas LLM_API_KEY está ausente; "
                "serviço de veto operará desativado."
            )
        else:
            veto_model = settings.llm_veto_model
            llm_client = HttpLLMClient(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=veto_model,
                timeout_seconds=settings.llm_veto_timeout_seconds,
            )
            veto_service = LLMVetoService(
                client=llm_client,
                model=veto_model,
                shadow_mode=shadow_mode,
                timeout_seconds=settings.llm_veto_timeout_seconds,
            )
            logger.info(
                "LLMVetoService inicializado (shadow=%s, model=%s)",
                shadow_mode,
                veto_model,
            )

    proposal_manager = ProposalManager(
        default_timeout_s=settings.proposal_timeout_s,
        stats_store=stats_store,
        veto_service=veto_service,
    )

    health_state = HealthStatusState(
        broker_connected=broker.is_connected() if broker else False,
        trading_halted=False,
        execution_mode=settings.execution_mode.value,
        proposal_manager=proposal_manager,
        order_executor=order_executor,
    )

    def _set_halt(halted: bool, reason: str = "") -> None:
        health_state.set_trading_halted(halted)
        logger.warning(
            "Estado de trading halt alterado para %s (motivo: %s)",
            halted,
            reason,
            extra={"event": "trading_halt_changed", "halted": halted, "reason": reason},
        )

    telegram_service = build_telegram_service(
        settings=settings,
        proposal_manager=proposal_manager,
        broker=broker,
        order_executor=order_executor,
        set_trading_halt_fn=_set_halt,
        is_trading_halted_fn=lambda: bool(health_state.trading_halted),
        get_status_fn=lambda: dict(health_state.get_status()[1]),
    )
    telegram_runner = TelegramRunner(telegram_service) if telegram_service else None
    if telegram_runner is not None:
        telegram_runner.start()

    advisor_dispatcher = AdvisorDispatcher(
        proposal_manager=proposal_manager,
        mode=settings.execution_mode,
        alert_manager=alert_manager,
        executor=order_executor,
        telegram_service=telegram_service,
        is_trading_halted_fn=lambda: bool(health_state.trading_halted),
    )

    journal_repo = JournalRepository(db_path=settings.journal_db)

    active_params = {"max_spread_atr_ratio": spread_filter.config.max_spread_atr_ratio}

    engines: list[LiveTradingEngine] = []
    for idx, sym in enumerate(symbols):
        sym_is_crypto = _is_crypto_symbol(sym)
        sym_enable_24_7 = (
            bool(getattr(args, "crypto_24_7", False)) or settings.crypto_24_7 or sym_is_crypto
        )
        sym_session_filter = SessionFilter(
            server_tz=settings.broker_server_tz,
            enable_24_7=sym_enable_24_7,
        )
        sym_composite_filter = CompositeTradeFilter(
            session_filter=sym_session_filter,
            news_filter=news_filter,
            spread_filter=spread_filter,
        )
        sym_magic = 123456 if len(symbols) == 1 else 123450 + idx * 100

        engine = LiveTradingEngine(
            broker=broker,
            proposal_manager=proposal_manager,
            advisor_dispatcher=advisor_dispatcher,
            composite_filter=sym_composite_filter,
            is_trading_halted_fn=lambda: bool(health_state.trading_halted),
            telegram_service=telegram_service,
            journal_repo=journal_repo,
            shadow_engine=shadow_engine,
            shadow_params=shadow_params,
            active_params=active_params,
            symbol=sym,
            trigger_timeframe=tuple(trigger_tfs),
            magic=sym_magic,
            risk_config=PositionRiskConfig(
                remove_tp_after_partial=settings.remove_tp_after_partial
            ),
            require_premium_discount=settings.smc_require_premium_discount,
            require_sweep_guard=settings.smc_require_sweep_guard,
            sweep_guard_max_rejection_ratio=settings.smc_sweep_guard_max_rejection_ratio,
            htf_alignment_required=settings.smc_htf_alignment_required,
        )
        engines.append(engine)

    live_runner = LiveTradingLoopRunner(
        engine=engines[0],
        engines=engines,
        watchdog=watchdog,
        poll_interval_s=2.0,
    )
    live_runner.start()

    try:
        run_server(
            settings=settings,
            watchdog=watchdog,
            alert_manager=alert_manager,
            proposal_manager=proposal_manager,
            order_executor=order_executor,
            state=health_state,
        )
    except KeyboardInterrupt:
        logger.info("Encerramento solicitado pelo usuário.")
    finally:
        live_runner.stop()
        if telegram_runner is not None:
            telegram_runner.stop()
        journal_repo.close()

    return 0
