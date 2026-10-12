"""Tests for the LLM Narrator subpackage, client, prompts, and CLI integration (T-A9)."""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr, ValidationError

from xau_bot.advisor.modes import ProposalStatus
from xau_bot.cli.journal import run_journal_cli
from xau_bot.cli.learn import run_learn_cli
from xau_bot.cli.main import build_parser, main
from xau_bot.config import Settings
from xau_bot.domain.models import Direction
from xau_bot.journal.models import (
    JournalEntry,
    LossCategory,
    LossRootCause,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)
from xau_bot.journal.reporter import JournalReporter
from xau_bot.journal.repository import JournalRepository
from xau_bot.learning.models import (
    CohortMetrics,
    FeatureStats,
    HumanDiscretionReport,
    HumanDiscretionVerdict,
    HumanSegmentStats,
    Hypothesis,
    HypothesisStatus,
    LearningReviewReport,
)
from xau_bot.learning.reporter import LearningReporter
from xau_bot.llm.client import HttpLLMClient, LLMClientError
from xau_bot.llm.models import NarrativeReport, NarratorMode
from xau_bot.llm.narrator import LLMNarrator
from xau_bot.llm.prompts import (
    SYSTEM_PROMPT_NARRATOR,
    format_human_discretion_prompt,
    format_learning_review_prompt,
    format_trade_post_mortem_prompt,
)


@pytest.fixture(autouse=True)
def isolate_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensures test suite never inherits external LLM keys or network configuration."""
    monkeypatch.delenv("XAU_LLM_ENABLED", raising=False)
    monkeypatch.delenv("XAU_LLM_API_KEY", raising=False)
    monkeypatch.delenv("XAU_LLM_BASE_URL", raising=False)


def _create_sample_entry(
    entry_id: str = "prop_1",
    outcome: TradeOutcome = TradeOutcome.LOSS,
    realized_r: float = -1.0,
) -> JournalEntry:
    now = datetime(2026, 10, 9, 10, 0, tzinfo=UTC)
    snap = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BULLISH",
        swing_high=2650.0,
        swing_low=2620.0,
        poi_type="OrderBlock",
        poi_level=2630.0,
        swept_liquidity=True,
        displacement_atr=1.5,
        fvg_size_atr=0.8,
        choch_level=2625.0,
        spread_price=0.2,
        atr_m5=2.0,
        news_distance_min=45.0,
        confluences=("FVG", "OrderBlock"),
        user_decision="APPROVED",
    )
    outcome_data = OutcomeData(
        executed=True,
        entry_price=2630.0,
        stop_loss=2625.0,
        take_profit=2640.0,
        exit_price=2625.0,
        realized_r=realized_r,
        pnl_usd=-100.0,
        mfe_r=0.4,
        mae_r=1.0,
        duration_seconds=120.0,
        exit_reason="SL_HIT",
    )
    return JournalEntry(
        id=entry_id,
        proposal_id=entry_id,
        snapshot=snap,
        direction=Direction.BULLISH,
        status=ProposalStatus.APPROVED,
        outcome=outcome,
        loss_category=LossCategory.BAD_DECISION,
        loss_cause=LossRootCause.COUNTER_BIAS,
        outcome_data=outcome_data,
        created_at=now,
    )


def _create_sample_learning_report() -> LearningReviewReport:
    now = datetime(2026, 10, 9, 10, 0, tzinfo=UTC)
    hyp = Hypothesis(
        id="hyp_01",
        created_at=now,
        parameter_name="min_displacement_atr",
        current_value="1.0",
        proposed_value="1.5",
        sample_size=35,
        condition_loss_rate=0.714,
        baseline_loss_rate=0.450,
        p_value=0.005,
        z_score=2.81,
        observed_expectancy_r=-0.65,
        status=HypothesisStatus.PROPOSED,
        description="Aumento do filtro de deslocamento",
        rationale="Deslocamento baixo concentra perdas",
    )
    obs = FeatureStats(
        feature_name="spread_max",
        condition_key="spread > 0.4",
        sample_size=12,
        loss_count=10,
        win_count=2,
        breakeven_count=0,
        loss_rate=0.833,
        baseline_loss_rate=0.500,
        z_score=1.45,
        p_value=0.15,
        mean_realized_r=-0.80,
        is_statistically_significant=False,
        description="Aguardando amostra",
    )
    feat = FeatureStats(
        feature_name="confluences",
        condition_key="single_confluence",
        sample_size=32,
        loss_count=20,
        win_count=12,
        breakeven_count=0,
        loss_rate=0.625,
        baseline_loss_rate=0.400,
        z_score=2.10,
        p_value=0.03,
        mean_realized_r=-0.25,
        is_statistically_significant=True,
        description="Confluência única",
    )
    return LearningReviewReport(
        reviewed_at=now,
        total_trades_analyzed=50,
        total_wins=20,
        total_losses=30,
        total_breakevens=0,
        win_rate=0.40,
        loss_rate=0.60,
        global_expectancy_r=-0.10,
        min_samples=30,
        alpha=0.05,
        loss_cause_distribution={"COUNTER_BIAS": 15, "INSUFFICIENT_DISPLACEMENT": 15},
        feature_stats=(feat,),
        hypotheses=(hyp,),
        accumulating_observations=(obs,),
    )


def _create_sample_human_report() -> HumanDiscretionReport:
    now = datetime(2026, 10, 9, 10, 0, tzinfo=UTC)
    cohort_app = CohortMetrics(
        cohort_name="APPROVED",
        total_proposals=25,
        evaluated_trades=25,
        win_count=18,
        loss_count=7,
        breakeven_count=0,
        win_rate=0.72,
        loss_rate=0.28,
        expectancy_r=1.20,
        std_dev_r=0.90,
        total_realized_r=30.0,
        profit_factor=2.8,
        realized_r_values=(2.0, -1.0),
    )
    cohort_ign = CohortMetrics(
        cohort_name="IGNORED",
        total_proposals=25,
        evaluated_trades=25,
        win_count=8,
        loss_count=17,
        breakeven_count=0,
        win_rate=0.32,
        loss_rate=0.68,
        expectancy_r=-0.35,
        std_dev_r=1.10,
        total_realized_r=-8.75,
        profit_factor=0.6,
        realized_r_values=(2.0, -1.0),
    )
    segment = HumanSegmentStats(
        dimension="session",
        segment_value="LONDON",
        total_count=30,
        approved_count=20,
        ignored_count=10,
        approval_rate=0.667,
        approved_expectancy_r=1.40,
        ignored_expectancy_r=-0.50,
        delta_expectancy_r=1.90,
        approved_win_rate=0.75,
        ignored_win_rate=0.30,
        delta_win_rate=0.45,
    )
    return HumanDiscretionReport(
        analyzed_at=now,
        min_samples=30,
        alpha=0.05,
        total_proposals=50,
        approved_metrics=cohort_app,
        ignored_metrics=cohort_ign,
        combined_metrics=cohort_app,
        acceptance_rate=0.50,
        delta_expectancy_r=1.55,
        delta_win_rate=0.40,
        p_value_expectancy=0.001,
        p_value_win_rate=0.002,
        verdict=HumanDiscretionVerdict.POSITIVE_EDGE,
        verdict_summary="Discricionariedade humana adiciona valor significativo.",
        segments=(segment,),
    )


def test_narrative_models_serialization() -> None:
    report = NarrativeReport(
        summary="Síntese do post-mortem do trade.",
        key_insights=("Entrada em Londres com confluência.", "Drawdown de 1.0R."),
        suggested_action="Refinar critério de ChoCH.",
        model="gpt-4o-mini",
        elapsed_seconds=1.234,
        mode=NarratorMode.POST_MORTEM,
        is_fallback=False,
        error_message=None,
    )
    data = report.to_dict()
    assert data["summary"] == "Síntese do post-mortem do trade."
    assert len(data["key_insights"]) == 2
    assert data["mode"] == "POST_MORTEM"
    assert data["is_fallback"] is False

    reconstructed = NarrativeReport.from_dict(data)
    assert reconstructed.summary == report.summary
    assert reconstructed.key_insights == report.key_insights
    assert reconstructed.suggested_action == report.suggested_action
    assert reconstructed.model == report.model
    assert reconstructed.mode == report.mode
    assert reconstructed.is_fallback == report.is_fallback


def test_http_llm_client_repr_masks_api_key() -> None:
    client = HttpLLMClient(
        api_key=SecretStr("super_secret_token_12345"),
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )
    client_repr = repr(client)
    assert "super_secret_token_12345" not in client_repr
    assert "gpt-4o-mini" in client_repr


def test_http_llm_client_success() -> None:
    client = HttpLLMClient(
        api_key=SecretStr("sk-test"),
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )
    response_payload = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": (
                        "RESUMO EXECUTIVO:\nTrade executado.\n"
                        "INSIGHTS CHAVE:\n- Ponto 1\n"
                        "AÇÃO SUGERIDA:\nMonitorar."
                    ),
                }
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = json.dumps(response_payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        result = client.complete("System prompt", "User prompt")
        assert "RESUMO EXECUTIVO" in result
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk-test"
        assert req.get_header("Content-type") == "application/json"


def test_http_llm_client_http_error() -> None:
    client = HttpLLMClient(api_key=SecretStr("sk-test"))
    error_body = json.dumps({"error": {"message": "Invalid API Key"}}).encode("utf-8")
    http_error = urllib.error.HTTPError(
        url="https://api.openai.com/v1/chat/completions",
        code=401,
        msg="Unauthorized",
        hdrs=MagicMock(),
        fp=io.BytesIO(error_body),
    )

    with (
        patch("urllib.request.urlopen", side_effect=http_error),
        pytest.raises(LLMClientError, match="HTTP 401 error from LLM endpoint"),
    ):
        client.complete("Sys", "User")


def test_http_llm_client_connection_error() -> None:
    client = HttpLLMClient(api_key=SecretStr("sk-test"))
    url_error = urllib.error.URLError("Connection refused")

    with (
        patch("urllib.request.urlopen", side_effect=url_error),
        pytest.raises(LLMClientError, match="Connection failure to LLM endpoint"),
    ):
        client.complete("Sys", "User")


def test_http_llm_client_timeout() -> None:
    client = HttpLLMClient(api_key=SecretStr("sk-test"), timeout_seconds=5.0)

    with (
        patch("urllib.request.urlopen", side_effect=TimeoutError("Timed out")),
        pytest.raises(LLMClientError, match="LLM request timed out after 5.0 seconds"),
    ):
        client.complete("Sys", "User")


def test_http_llm_client_malformed_json_response() -> None:
    client = HttpLLMClient(api_key=SecretStr("sk-test"))
    mock_resp = MagicMock()
    mock_resp.getcode.return_value = 200
    mock_resp.read.return_value = b"Not JSON"
    mock_resp.__enter__.return_value = mock_resp

    with (
        patch("urllib.request.urlopen", return_value=mock_resp),
        pytest.raises(LLMClientError, match="Failed to parse LLM response payload"),
    ):
        client.complete("Sys", "User")


def test_prompt_templates_and_no_invention_constraint_ac_narrator_03() -> None:
    assert "DO NOT invent numbers or statistics" in SYSTEM_PROMPT_NARRATOR
    assert "NÃO CRIE OU INVENTE NENHUM NÚMERO OU ESTATÍSTICA" in SYSTEM_PROMPT_NARRATOR

    entry = _create_sample_entry()
    prompt_trade = format_trade_post_mortem_prompt(entry)
    assert "IMPORTANTE: NÃO invente números" in prompt_trade
    assert entry.proposal_id in prompt_trade
    assert "- Desfecho: LOSS" in prompt_trade
    assert "COUNTER_BIAS" in prompt_trade

    learn_rep = _create_sample_learning_report()
    prompt_learn = format_learning_review_prompt(learn_rep)
    assert "IMPORTANTE: NÃO invente números" in prompt_learn
    assert "min_displacement_atr" in prompt_learn
    assert "Distribuição de Causas Raiz de Perda:" in prompt_learn

    human_rep = _create_sample_human_report()
    prompt_human = format_human_discretion_prompt(human_rep)
    assert "IMPORTANTE: NÃO invente números" in prompt_human
    assert "POSITIVE_EDGE" in prompt_human
    assert "Comparativo de Coortes:" in prompt_human


def test_narrator_trade_post_mortem_success_ac_narrator_01() -> None:
    mock_client = MagicMock()
    mock_client.complete.return_value = (
        "RESUMO EXECUTIVO:\n"
        "O trade sofreu stop loss devido a choque com viés macro na sessão de Londres.\n"
        "INSIGHTS CHAVE:\n"
        "- Confluência de OrderBlock não foi suficiente contra momentum HTF.\n"
        "- Drawdown rápido atingiu -1.0R em 120 segundos.\n"
        "AÇÃO SUGERIDA:\n"
        "Evitar compras quando viés de tempo gráfico maior divergir da estrutura de M5."
    )
    narrator = LLMNarrator(client=mock_client, settings=Settings(llm_enabled=True))
    entry = _create_sample_entry()

    report = narrator.narrate_trade_post_mortem(entry)
    assert report.is_fallback is False
    assert report.mode == NarratorMode.POST_MORTEM
    assert "O trade sofreu stop loss" in report.summary
    assert len(report.key_insights) == 2
    assert "Evitar compras" in (report.suggested_action or "")
    mock_client.complete.assert_called_once()


def test_narrator_learning_review_and_human_discretion_ac_narrator_02() -> None:
    mock_client = MagicMock()
    mock_client.complete.return_value = (
        "RESUMO EXECUTIVO:\n"
        "Identificada dominância de perdas por conflito HTF e edge positivo na seleção humana.\n"
        "INSIGHTS CHAVE:\n"
        "- Hipótese de deslocamento mínimo de ATR reduz perdas com significância.\n"
        "- Sessão de Londres apresenta maior consistência de aprovação.\n"
        "AÇÃO SUGERIDA:\n"
        "Promover parâmetro min_displacement_atr para validação em sombra."
    )
    narrator = LLMNarrator(client=mock_client, settings=Settings(llm_enabled=True))

    learn_rep = _create_sample_learning_report()
    res_learn = narrator.narrate_learning_review(learn_rep)
    assert res_learn.is_fallback is False
    assert res_learn.mode == NarratorMode.LEARNING_REVIEW

    human_rep = _create_sample_human_report()
    res_human = narrator.narrate_human_discretion(human_rep)
    assert res_human.is_fallback is False
    assert res_human.mode == NarratorMode.HUMAN_DISCRETION


def test_narrator_fallback_when_disabled_or_no_key_ac_narrator_04() -> None:
    narrator = LLMNarrator(settings=Settings(llm_enabled=False))
    assert narrator.is_available is False

    entry = _create_sample_entry()
    res_trade = narrator.narrate_trade_post_mortem(entry)
    assert res_trade.is_fallback is True
    assert res_trade.model == "deterministic-fallback"
    assert "Parecer qualitativo LLM não executado" in res_trade.summary

    learn_rep = _create_sample_learning_report()
    res_learn = narrator.narrate_learning_review(learn_rep)
    assert res_learn.is_fallback is True
    assert "Revisão periódica de aprendizado concluída" in res_learn.summary

    human_rep = _create_sample_human_report()
    res_human = narrator.narrate_human_discretion(human_rep)
    assert res_human.is_fallback is True
    assert "Análise de discricionariedade humana resultou no veredito" in res_human.summary


def test_narrator_fallback_on_client_error_ac_narrator_05() -> None:
    mock_client = MagicMock()
    mock_client.complete.side_effect = LLMClientError("Network timeout during synthesis")
    narrator = LLMNarrator(client=mock_client, settings=Settings(llm_enabled=True))

    entry = _create_sample_entry()
    report = narrator.narrate_trade_post_mortem(entry)

    assert report.is_fallback is True
    assert report.model == "deterministic-fallback"
    assert report.error_message == "Network timeout during synthesis"
    assert "Operação BULLISH em LONDON finalizada" in report.summary


def test_narrator_fallback_on_generic_exception() -> None:
    mock_client = MagicMock()
    mock_client.complete.side_effect = RuntimeError("Crash inesperado")
    narrator = LLMNarrator(client=mock_client, settings=Settings(llm_enabled=True))

    entry = _create_sample_entry()
    report = narrator.narrate_trade_post_mortem(entry)

    assert report.is_fallback is True
    assert report.model == "deterministic-fallback"
    assert "Crash inesperado" in (report.error_message or "")


def test_reporters_render_narrative() -> None:
    entry = _create_sample_entry()
    narrative = NarrativeReport(
        summary="Síntese analítica do trade.",
        key_insights=("Insight técnico 1", "Insight técnico 2"),
        suggested_action="Refinar gatilho.",
        model="gpt-4o-mini",
        elapsed_seconds=0.45,
        mode=NarratorMode.POST_MORTEM,
        is_fallback=False,
    )
    md_trade = JournalReporter.generate_markdown_report(entry, narrative=narrative)
    assert "## 4. Parecer Qualitativo do Narrador LLM" in md_trade
    assert "Síntese analítica do trade." in md_trade
    assert "Insight técnico 1" in md_trade
    assert "Refinar gatilho." in md_trade

    learn_rep = _create_sample_learning_report()
    md_learn = LearningReporter.generate_markdown_report(learn_rep, narrative=narrative)
    assert "## 6. Parecer Qualitativo do Narrador LLM" in md_learn

    human_rep = _create_sample_human_report()
    md_human = LearningReporter.generate_human_discretion_markdown(human_rep, narrative=narrative)
    assert "## 5. Parecer Qualitativo do Narrador LLM" in md_human


def test_cli_learn_with_narrate(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["learn", "--db", ":memory:", "--narrate"])
    out = io.StringIO()
    err = io.StringIO()

    exit_code = run_learn_cli(args, stdout=out, stderr=err)
    assert exit_code == 0
    output_text = out.getvalue()
    assert "Parecer Qualitativo do Narrador LLM" in output_text

    out_json = io.StringIO()
    err_json = io.StringIO()
    args_json = parser.parse_args(["learn", "--db", ":memory:", "--narrate", "--json"])
    exit_json = run_learn_cli(args_json, stdout=out_json, stderr=err_json)
    assert exit_json == 0
    data = json.loads(out_json.getvalue())
    assert "narrative" in data
    assert data["narrative"]["mode"] == "LEARNING_REVIEW"


def test_cli_journal_with_narrate(tmp_path: Path) -> None:
    db_file = tmp_path / "journal_test.db"
    repo = JournalRepository(db_path=str(db_file))
    entry = _create_sample_entry(entry_id="prop_narrate_1")
    repo.save_entry(entry)
    repo.close()

    parser = build_parser()
    args = parser.parse_args(
        ["journal", "--db", str(db_file), "--report", "prop_narrate_1", "--narrate"]
    )
    out = io.StringIO()
    err = io.StringIO()

    exit_code = run_journal_cli(args, stdout=out, stderr=err)
    assert exit_code == 0
    output = out.getvalue()
    assert "Trade Post-Mortem Report" in output
    assert "## 4. Parecer Qualitativo do Narrador LLM" in output


def test_main_cli_dispatch_narrate_flags() -> None:
    out = io.StringIO()
    err = io.StringIO()
    exit_code = main(["learn", "--db", ":memory:", "--narrate"], stdout=out, stderr=err)
    assert exit_code == 0
    assert "Parecer Qualitativo do Narrador LLM" in out.getvalue()


def test_settings_llm_timeout_max_constraint_ac_narrator_06() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=16.0)


def test_parse_completion_preserves_numbers_without_bullet() -> None:
    text = (
        "RESUMO EXECUTIVO:\nResumo.\n"
        "INSIGHTS CHAVE:\n"
        "1.5x ATR insuficiente\n"
        "-1.0R de drawdown\n"
        "- Marcador normal\n"
        "AÇÃO SUGERIDA:\nMonitorar."
    )
    _, insights, _ = LLMNarrator._parse_completion(text)
    assert "1.5x ATR insuficiente" in insights
    assert "-1.0R de drawdown" in insights
    assert "Marcador normal" in insights


def test_narrator_enabled_without_key_falls_back_ac_narrator_04() -> None:
    settings = Settings(llm_enabled=True, llm_api_key=None)
    with patch("urllib.request.urlopen") as mock_urlopen:
        narrator = LLMNarrator(settings=settings)
        assert narrator.is_available is False
        entry = _create_sample_entry()
        report = narrator.narrate_trade_post_mortem(entry)
        assert report.is_fallback is True
        mock_urlopen.assert_not_called()


def test_narrator_creates_http_client_when_key_present() -> None:
    settings = Settings(llm_enabled=True, llm_api_key=SecretStr("sk-sample"))
    narrator = LLMNarrator(settings=settings)
    assert narrator.is_available is True
    assert isinstance(narrator.client, HttpLLMClient)


def test_cli_journal_post_mortem_subcommand(tmp_path: Path) -> None:
    db_file = tmp_path / "journal_test.db"
    repo = JournalRepository(db_path=str(db_file))
    entry = _create_sample_entry(entry_id="prop_pm_1")
    repo.save_entry(entry)
    repo.close()

    out = io.StringIO()
    err = io.StringIO()
    exit_code = main(
        ["journal", "post-mortem", "prop_pm_1", "--db", str(db_file)],
        stdout=out,
        stderr=err,
    )
    assert exit_code == 0
    assert "Trade Post-Mortem Report" in out.getvalue()
    assert "## 4. Parecer Qualitativo do Narrador LLM" in out.getvalue()


def test_cli_learn_human_with_narrate() -> None:
    out = io.StringIO()
    err = io.StringIO()
    exit_code = main(["learn", "human", "--db", ":memory:", "--narrate"], stdout=out, stderr=err)
    assert exit_code == 0
    assert "Parecer Qualitativo do Narrador LLM" in out.getvalue()

    out_json = io.StringIO()
    err_json = io.StringIO()
    exit_json = main(
        ["learn", "human", "--db", ":memory:", "--narrate", "--json"],
        stdout=out_json,
        stderr=err_json,
    )
    assert exit_json == 0
    data = json.loads(out_json.getvalue())
    assert "narrative" in data
    assert data["narrative"]["mode"] == "HUMAN_DISCRETION"


def test_network_isolation_fixture_prevents_accidental_network() -> None:
    with patch("urllib.request.urlopen") as mock_urlopen:
        out = io.StringIO()
        err = io.StringIO()
        exit_code = main(
            ["learn", "--db", ":memory:", "--narrate"],
            stdout=out,
            stderr=err,
        )
        assert exit_code == 0
        mock_urlopen.assert_not_called()
