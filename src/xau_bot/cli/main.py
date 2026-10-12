import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from xau_bot.cli.backtest import build_backtest_parser, run_backtest_cli
from xau_bot.cli.journal import build_journal_parser, run_journal_cli
from xau_bot.cli.learn import build_learn_parser, run_learn_cli
from xau_bot.cli.ml import build_ml_parser, run_ml_cli
from xau_bot.cli.params import build_params_parser, run_params_cli
from xau_bot.cli.run import build_run_parser, run_bot_cli
from xau_bot.cli.validate import build_validate_parser, run_validate_cli


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xau-bot",
        description="XAU-BOT: Robô de Trading Institucional SMC para Ouro (XAU/USD)",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        title="Subcomandos disponíveis",
        required=True,
    )

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Executa simulação de backtest com análise estatística e validação do Portão 1.",
    )
    build_backtest_parser(backtest_parser)
    backtest_parser.set_defaults(handler=run_backtest_cli)

    run_parser = subparsers.add_parser(
        "run",
        help="Executa o robô em tempo real ou simulado com servidor de observabilidade e alertas.",
    )
    build_run_parser(run_parser)
    run_parser.set_defaults(handler=run_bot_cli)

    journal_parser = subparsers.add_parser(
        "journal",
        help="Auditoria e relatórios do diário de trading com desfechos e taxonomia de perdas.",
    )
    build_journal_parser(journal_parser)
    journal_parser.set_defaults(handler=run_journal_cli)

    learn_parser = subparsers.add_parser(
        "learn",
        help="Ciclo de aprendizado e revisão periódica com hipóteses de ajuste.",
    )
    build_learn_parser(learn_parser)
    learn_parser.set_defaults(handler=run_learn_cli)

    params_parser = subparsers.add_parser(
        "params",
        help="Gerencia versões de parâmetros e histórico de rollback auditável.",
    )
    build_params_parser(params_parser)
    params_parser.set_defaults(handler=run_params_cli)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Valida hipóteses geradas pelo ciclo de aprendizado contra critérios empíricos.",
    )
    build_validate_parser(validate_parser)
    validate_parser.set_defaults(handler=run_validate_cli)

    ml_parser = subparsers.add_parser(
        "ml",
        help="Meta-labeling com modelo supervisionado sobre features do setup de trading.",
    )
    build_ml_parser(ml_parser)
    ml_parser.set_defaults(handler=run_ml_cli)

    return parser


def main(
    argv: Sequence[str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    if stdout is None:
        stdout = sys.stdout
    if stderr is None:
        stderr = sys.stderr

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2

    handler = args.handler
    return int(handler(args, stdout=stdout, stderr=stderr))


if __name__ == "__main__":
    sys.exit(main())
