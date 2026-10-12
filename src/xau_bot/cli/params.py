"""CLI subcommands for managing parameter versions, promote actions and audit rollbacks."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from xau_bot.learning.registry import ParameterRegistry


@dataclass(frozen=True, slots=True)
class ParamSpec:
    param_type: type
    min_value: float
    max_value: float
    description: str


ALLOWED_PARAM_SPECS: dict[str, ParamSpec] = {
    "min_displacement_multiplier": ParamSpec(
        float, 0.0, 10.0, "Multiplicador de deslocamento mínimo em ATR"
    ),
    "max_spread_atr_ratio": ParamSpec(
        float, 0.001, 1.0, "Razão máxima de spread em relação ao ATR"
    ),
    "risk_per_trade_pct": ParamSpec(float, 0.1, 5.0, "Percentual de risco por operação (Settings)"),
    "max_daily_loss_pct": ParamSpec(
        float, 0.5, 10.0, "Limite percentual de perda diária (Settings)"
    ),
    "max_drawdown_pct": ParamSpec(float, 1.0, 25.0, "Limite percentual de drawdown (Settings)"),
    "max_consecutive_losses": ParamSpec(int, 1, 10, "Limite de perdas consecutivas (Settings)"),
    "max_daily_trades": ParamSpec(int, 1, 20, "Limite máximo de operações por dia (Settings)"),
    "be_trigger_r": ParamSpec(float, 0.1, 10.0, "Gatilho de breakeven em R (PositionRiskConfig)"),
    "trailing_atr_multiplier": ParamSpec(
        float, 0.1, 10.0, "Multiplicador de ATR para trailing stop (PositionRiskConfig)"
    ),
}


def build_params_parser(parser: argparse.ArgumentParser) -> None:
    action_sub = parser.add_subparsers(dest="params_action")

    list_p = action_sub.add_parser("list", help="Lista todas as versões de parâmetros.")
    list_p.add_argument(
        "--db",
        default="parameters.db",
        help="Caminho para o banco de dados SQLite de parâmetros (padrão: parameters.db).",
    )
    list_p.add_argument(
        "--json",
        action="store_true",
        help="Emite saída em formato JSON.",
    )

    show_p = action_sub.add_parser("show", help="Exibe a versão ativa de parâmetros.")
    show_p.add_argument(
        "--db",
        default="parameters.db",
        help="Caminho para o banco de dados SQLite de parâmetros.",
    )
    show_p.add_argument(
        "--json",
        action="store_true",
        help="Emite saída em formato JSON.",
    )

    create_p = action_sub.add_parser("create", help="Registra uma nova versão de parâmetros.")
    create_p.add_argument(
        "--db",
        default="parameters.db",
        help="Caminho para o banco de dados SQLite de parâmetros.",
    )
    create_p.add_argument(
        "--param",
        action="append",
        dest="params_kv",
        help="Parâmetro no formato CHAVE=VALOR (pode ser repetido).",
    )
    create_p.add_argument(
        "--hypothesis",
        dest="hypothesis_id",
        help="Identificador opcional da hipótese que originou a versão.",
    )
    create_p.add_argument(
        "--by",
        dest="applied_by",
        default="operator",
        help="Identificação do operador responsável pela criação da versão.",
    )

    promote_p = action_sub.add_parser(
        "promote",
        help="Promove uma hipótese validada criando nova versão ativa.",
    )
    promote_p.add_argument(
        "--hypothesis-id",
        required=True,
        help="Identificador da hipótese validada a ser promovida.",
    )
    promote_p.add_argument(
        "--db",
        default="parameters.db",
        help="Caminho para o banco de dados SQLite onde a versão ativa será registrada.",
    )
    promote_p.add_argument(
        "--journal-db",
        default="journal.db",
        help="Caminho para o banco de dados SQLite do journal onde as validações residem.",
    )
    promote_p.add_argument(
        "--by",
        dest="applied_by",
        default="operator",
        help="Identificação do operador responsável pela promoção.",
    )

    rollback_p = action_sub.add_parser(
        "rollback", help="Executa rollback para uma versão anterior."
    )
    rollback_p.add_argument(
        "--version",
        dest="target_version",
        help="Identificador da versão de parâmetros para a qual reverter.",
    )
    rollback_p.add_argument(
        "target_version_pos",
        nargs="?",
        help="Identificador da versão de parâmetros posicional.",
    )
    rollback_p.add_argument(
        "--db",
        default="parameters.db",
        help="Caminho para o banco de dados SQLite de parâmetros.",
    )
    rollback_p.add_argument(
        "--by",
        dest="applied_by",
        default="operator",
        help="Identificação do operador responsável pelo rollback.",
    )


def validate_param_dict(params: Any) -> tuple[bool, str]:
    if not isinstance(params, dict) or not params:
        return False, "Parâmetros devem ser um dicionário não vazio."

    for k, v in params.items():
        if not isinstance(k, str) or k not in ALLOWED_PARAM_SPECS:
            allowed_keys = ", ".join(sorted(ALLOWED_PARAM_SPECS.keys()))
            return False, f"Parâmetro desconhecido: '{k}'. Parâmetros permitidos: {allowed_keys}"

        spec = ALLOWED_PARAM_SPECS[k]
        if isinstance(v, bool):
            return False, f"Valor booleano inválido para parâmetro '{k}'."

        if spec.param_type is int:
            if not isinstance(v, (int, float)):
                return False, f"Valor inválido '{v}' para parâmetro '{k}' (esperado int)."
            if isinstance(v, float) and not v.is_integer():
                return False, f"Valor decimal '{v}' inválido para parâmetro inteiro '{k}'."
            val_int = int(v)
            if not (spec.min_value <= val_int <= spec.max_value):
                return (
                    False,
                    f"Valor {val_int} para '{k}' fora da faixa permitida "
                    f"[{spec.min_value}, {spec.max_value}].",
                )
        else:
            if not isinstance(v, (int, float)):
                return False, f"Valor inválido '{v}' para parâmetro '{k}' (esperado float)."
            val_flt = float(v)
            if math.isnan(val_flt) or math.isinf(val_flt):
                return False, f"Valor numérico inválido (NaN/Inf) para parâmetro '{k}'."
            if not (spec.min_value <= val_flt <= spec.max_value):
                return (
                    False,
                    f"Valor {val_flt} para '{k}' fora da faixa permitida "
                    f"[{spec.min_value}, {spec.max_value}].",
                )

    return True, ""


def _parse_and_validate_params(
    raw_list: list[str],
    stderr: TextIO,
) -> dict[str, Any] | None:
    temp_dict: dict[str, Any] = {}
    for item in raw_list:
        if "=" not in item:
            stderr.write(f"Formato inválido de parâmetro: '{item}'. Use CHAVE=VALOR.\n")
            return None
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in ALLOWED_PARAM_SPECS:
            allowed_keys = ", ".join(sorted(ALLOWED_PARAM_SPECS.keys()))
            stderr.write(f"Parâmetro desconhecido: '{k}'. Parâmetros permitidos: {allowed_keys}\n")
            return None

        spec = ALLOWED_PARAM_SPECS[k]
        try:
            if spec.param_type is int:
                temp_dict[k] = int(v)
            else:
                temp_dict[k] = float(v)
        except (ValueError, TypeError):
            stderr.write(
                f"Valor inválido '{v}' para parâmetro '{k}' "
                f"(esperado tipo {spec.param_type.__name__}).\n"
            )
            return None

    ok, err_msg = validate_param_dict(temp_dict)
    if not ok:
        stderr.write(f"{err_msg}\n")
        return None

    return temp_dict


def run_params_cli(
    args: argparse.Namespace,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    action = getattr(args, "params_action", None)
    if not action:
        action = "list"

    db_str = getattr(args, "db", "parameters.db")

    if (
        action in ("list", "show", "rollback")
        and db_str != ":memory:"
        and not Path(db_str).exists()
    ):
        stderr.write(f"Banco de dados de parâmetros não encontrado: {db_str}\n")
        return 1

    as_json = getattr(args, "json", False)

    try:
        if action == "promote":
            hyp_id = str(getattr(args, "hypothesis_id", ""))
            applied_by = getattr(args, "applied_by", "operator")
            journal_db_str = str(getattr(args, "journal_db", "journal.db"))

            if journal_db_str != ":memory:" and not Path(journal_db_str).exists():
                stderr.write(f"Banco de dados do journal não encontrado: {journal_db_str}\n")
                return 1

            conn = sqlite3.connect(journal_db_str)
            try:
                table_check = conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='validation_results';"
                ).fetchone()
                if not table_check:
                    stderr.write(
                        f"Erro: Nenhuma validação registrada no banco '{journal_db_str}'. "
                        "Execute 'xau-bot validate' previamente.\n"
                    )
                    return 1

                row = conn.execute(
                    """
                    SELECT parameter_name, proposed_value, verdict, rejection_reason
                    FROM validation_results
                    WHERE hypothesis_id = ?;
                    """,
                    (hyp_id,),
                ).fetchone()

                if not row:
                    stderr.write(
                        f"Erro: Hipótese '{hyp_id}' não encontrada no histórico de validações.\n"
                    )
                    return 1

                param_name, proposed_value, verdict, rejection_reason = row
                if verdict != "VALIDATED":
                    reason_msg = rejection_reason or "Critérios não atendidos"
                    stderr.write(
                        f"Erro: Hipótese '{hyp_id}' não pode ser promovida "
                        f"pois o veredito é '{verdict}'. Justificativa: {reason_msg}.\n"
                    )
                    return 1
            finally:
                conn.close()

            parsed_params = _parse_and_validate_params([f"{param_name}={proposed_value}"], stderr)
            if parsed_params is None:
                return 2

            with ParameterRegistry(db_str) as registry:
                active = registry.get_active_version()
                new_params = dict(active.parameters) if active else {}
                new_params.update(parsed_params)

                new_version = registry.create_version(
                    parameters=new_params,
                    source_hypothesis_id=hyp_id,
                    applied_by=applied_by,
                    activate=True,
                )
                stdout.write(
                    f"Nova versão de parâmetros {new_version.version_id} promovida com sucesso "
                    f"a partir da hipótese {hyp_id} ({param_name}={parsed_params[param_name]}).\n"
                )
                return 0

        with ParameterRegistry(db_str) as registry:
            if action == "list":
                versions = registry.list_versions()
                if as_json:
                    stdout.write(json.dumps([v.to_dict() for v in versions], indent=2))
                    stdout.write("\n")
                else:
                    if not versions:
                        stdout.write("Nenhuma versão de parâmetros registrada no banco.\n")
                        return 0
                    stdout.write("# Versões de Parâmetros\n\n")
                    stdout.write(
                        "| Versão | Criado em | Ativo | Responsável | Hipótese Origem |"
                        " Rollback Alvo | Parâmetros |\n"
                    )
                    stdout.write("| :--- | :--- | :---: | :--- | :--- | :--- | :--- |\n")
                    for v in versions:
                        status_str = "SIM" if v.is_active else "NÃO"
                        hyp_str = v.source_hypothesis_id or "-"
                        rb_str = v.rollback_target_id or "-"
                        p_summary = ", ".join(f"`{k}={val}`" for k, val in v.parameters.items())
                        stdout.write(
                            f"| `{v.version_id}` | {v.created_at.isoformat()} | "
                            f"{status_str} | {v.applied_by} | {hyp_str} | {rb_str} | "
                            f"{p_summary} |\n"
                        )
                return 0

            if action == "show":
                active = registry.get_active_version()
                if as_json:
                    if active is None:
                        stdout.write("{}\n")
                    else:
                        stdout.write(json.dumps(active.to_dict(), indent=2))
                        stdout.write("\n")
                else:
                    if active is None:
                        stdout.write("Nenhuma versão ativa de parâmetros encontrada.\n")
                        return 0
                    stdout.write(f"# Versão Ativa: {active.version_id}\n\n")
                    stdout.write(f"- **Criado em:** {active.created_at.isoformat()}\n")
                    stdout.write(f"- **Aplicado por:** {active.applied_by}\n")
                    stdout.write(
                        f"- **Hipótese Origem:** {active.source_hypothesis_id or 'Inicialização'}\n"
                    )
                    stdout.write("## Parâmetros:\n")
                    for k, val in active.parameters.items():
                        stdout.write(f"- `{k}`: `{val}`\n")
                return 0

            if action == "create":
                raw_params = getattr(args, "params_kv", None)
                if not raw_params:
                    stderr.write("Nenhum parâmetro fornecido via --param CHAVE=VALOR.\n")
                    return 2

                parsed_params = _parse_and_validate_params(raw_params, stderr)
                if parsed_params is None:
                    return 2

                source_hyp_id: str | None = getattr(args, "hypothesis_id", None)
                applied_by = getattr(args, "applied_by", "operator")

                new_version = registry.create_version(
                    parameters=parsed_params,
                    source_hypothesis_id=source_hyp_id,
                    applied_by=applied_by,
                    activate=True,
                )
                msg = f"Nova versão de parâmetros {new_version.version_id} registrada com sucesso"
                if source_hyp_id:
                    msg += f" (Hipótese: {source_hyp_id})"
                stdout.write(f"{msg}.\n")
                return 0

            if action == "rollback":
                target_id = getattr(args, "target_version", None) or getattr(
                    args, "target_version_pos", None
                )
                if not target_id:
                    stderr.write("Informe o ID da versão para rollback via argumento.\n")
                    return 2

                applied_by = getattr(args, "applied_by", "operator")
                try:
                    reverted_version = registry.rollback(
                        target_version_id=target_id,
                        applied_by=applied_by,
                    )
                    stdout.write(
                        f"Rollback executado com sucesso para a versão {target_id}. "
                        f"Nova versão ativa criada: {reverted_version.version_id}.\n"
                    )
                    return 0
                except ValueError as exc:
                    stderr.write(f"Erro no rollback: {exc}\n")
                    return 1

    except (sqlite3.Error, OSError) as exc:
        stderr.write(f"Erro no banco de dados de parâmetros: {exc}\n")
        return 1

    return 0
