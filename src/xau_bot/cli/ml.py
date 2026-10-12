"""CLI subcommands for Machine Learning Meta-Labeling."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import TextIO

from xau_bot.config import Settings
from xau_bot.journal.repository import JournalRepository
from xau_bot.ml.classifier import LogisticMetaClassifier
from xau_bot.ml.evaluator import calculate_metrics
from xau_bot.ml.features import FeatureExtractor
from xau_bot.ml.models import ModelMetrics

DEFAULT_MODEL_PATH = "data/models/meta_classifier.json"


def build_ml_parser(parser: argparse.ArgumentParser) -> None:
    """Configures CLI arguments and subcommands for ML meta-labeling."""
    settings = Settings()
    default_model = settings.ml_model_path or DEFAULT_MODEL_PATH
    subparsers = parser.add_subparsers(
        dest="action",
        title="ML Ações",
        required=True,
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Treina o classificador de meta-labeling a partir do diário de trading.",
    )
    train_parser.add_argument(
        "--db",
        type=str,
        default=settings.journal_db,
        help=f"Caminho do SQLite do journal (padrão: {settings.journal_db}).",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default=default_model,
        help=f"Caminho de saída para o modelo JSON (padrão: {default_model}).",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="Número de épocas de treinamento SGD (padrão: 250).",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Taxa de aprendizado inicial (padrão: 0.1).",
    )
    train_parser.add_argument(
        "--l2",
        type=float,
        default=0.01,
        help="Fator de regularização L2 (padrão: 0.01).",
    )
    train_parser.add_argument(
        "--threshold",
        type=float,
        default=settings.ml_min_probability,
        help=f"Limiar de corte para decisão ALLOW (padrão: {settings.ml_min_probability}).",
    )
    train_parser.add_argument(
        "--json",
        action="store_true",
        help="Emite saída em JSON estruturado.",
    )

    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Avalia o modelo de meta-labeling contra o diário de trading.",
    )
    eval_parser.add_argument(
        "--db",
        type=str,
        default=settings.journal_db,
        help=f"Caminho do SQLite do journal (padrão: {settings.journal_db}).",
    )
    eval_parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help=f"Caminho do arquivo de modelo JSON (padrão: {default_model}).",
    )
    eval_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Limiar de corte alternativo para avaliação (opcional).",
    )
    eval_parser.add_argument(
        "--all",
        action="store_true",
        help="Avalia o diário completo incluindo o período usado no treino (in-sample).",
    )
    eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Emite saída em JSON estruturado.",
    )


def _format_markdown_report(
    title: str,
    metrics: ModelMetrics,
    extra_info: dict[str, str],
    test_metrics: ModelMetrics | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "## Informações do Treinamento / Modelo",
    ]
    for k, v in extra_info.items():
        lines.append(f"- **{k}**: {v}")

    if test_metrics is not None:
        lines.extend(
            [
                "",
                "## Métricas de Desempenho (Split Temporal Treino / Teste)",
                "| Métrica | Treino (In-Sample) | Teste (Out-of-Sample) |",
                "| :--- | :--- | :--- |",
                f"| Amostras Totais | {metrics.sample_size} | {test_metrics.sample_size} |",
                (
                    f"| Taxa de Vencedores (Positivos) | {metrics.positive_rate * 100:.1f}% |"
                    f" {test_metrics.positive_rate * 100:.1f}% |"
                ),
                f"| Acurácia | {metrics.accuracy:.4f} | {test_metrics.accuracy:.4f} |",
                (
                    f"| Precisão (Precision) | {metrics.precision:.4f} |"
                    f" {test_metrics.precision:.4f} |"
                ),
                f"| Revocação (Recall) | {metrics.recall:.4f} | {test_metrics.recall:.4f} |",
                f"| ROC-AUC | {metrics.roc_auc:.4f} | {test_metrics.roc_auc:.4f} |",
                f"| Brier Score | {metrics.brier_score:.4f} | {test_metrics.brier_score:.4f} |",
                (
                    f"| Ganho de Expectativa (ΔR) | {metrics.delta_r:+.4f} R |"
                    f" {test_metrics.delta_r:+.4f} R |"
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Métricas de Desempenho",
                "| Métrica | Valor |",
                "| :--- | :--- |",
                f"| Amostras Totais | {metrics.sample_size} |",
                f"| Taxa de Vencedores (Positivos) | {metrics.positive_rate * 100:.1f}% |",
                f"| Acurácia | {metrics.accuracy:.4f} |",
                f"| Precisão (Precision) | {metrics.precision:.4f} |",
                f"| Revocação (Recall) | {metrics.recall:.4f} |",
                f"| ROC-AUC | {metrics.roc_auc:.4f} |",
                f"| Brier Score | {metrics.brier_score:.4f} |",
                f"| Ganho de Expectativa (ΔR) | {metrics.delta_r:+.4f} R |",
                "",
            ]
        )
    return "\n".join(lines)


def run_ml_cli(
    args: argparse.Namespace,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Executes the ML CLI commands."""
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    settings = Settings()
    default_model = settings.ml_model_path or DEFAULT_MODEL_PATH

    action = getattr(args, "action", None)
    if action not in ("train", "evaluate"):
        err.write(f"Ação desconhecida: {action}\n")
        return 1

    db_path = str(getattr(args, "db", None) or settings.journal_db)

    if action == "evaluate":
        model_path = Path(str(getattr(args, "model", None) or default_model))
        if not model_path.exists():
            err.write(f"Arquivo de modelo não encontrado: {model_path}\n")
            return 1
        if db_path != ":memory:" and not Path(db_path).exists():
            err.write(f"Arquivo de banco de dados não encontrado: {db_path}\n")
            return 1
        clf = LogisticMetaClassifier.load_from_file(model_path)
    else:
        output_path = Path(str(getattr(args, "output", None) or default_model))
        if db_path != ":memory:" and not Path(db_path).exists():
            msg = (
                "Dados insuficientes no diário: requer ao menos 1 amostra WIN e 1 amostra LOSS"
                " com snapshot para treinar/avaliar."
            )
            if getattr(args, "json", False):
                out.write(
                    json.dumps(
                        {"status": "insufficient_data", "samples": 0, "message": msg}, indent=2
                    )
                    + "\n"
                )
            else:
                out.write(f"AVISO: {msg}\n")
            return 0
        clf = None

    try:
        repo = JournalRepository(db_path=db_path)
        try:
            entries = repo.list_entries(limit=5000)
        finally:
            repo.close()
    except sqlite3.Error as e:
        err.write(f"Erro ao acessar banco de dados {db_path}: {e}\n")
        return 1

    entries.sort(key=lambda item: item.created_at)

    if (
        action == "evaluate"
        and clf is not None
        and not getattr(args, "all", False)
        and clf.trained_at is not None
    ):
        entries = [e for e in entries if e.created_at > clf.trained_at]

    X, y, r_vals = FeatureExtractor.extract_dataset(entries)
    if not X or not any(label == 1 for label in y) or not any(label == 0 for label in y):
        msg = (
            "Dados insuficientes no diário: requer ao menos 1 amostra WIN e 1 amostra LOSS"
            " com snapshot para treinar/avaliar."
        )
        if action == "evaluate" and not getattr(args, "all", False) and not X:
            msg = (
                "Nenhum dado out-of-sample posterior ao treinamento encontrado."
                " Use a flag --all para avaliar o histórico completo (in-sample)."
            )
        if getattr(args, "json", False):
            out.write(
                json.dumps(
                    {"status": "insufficient_data", "samples": len(X), "message": msg}, indent=2
                )
                + "\n"
            )
        else:
            out.write(f"AVISO: {msg}\n")
        return 0

    if action == "train":
        epochs = int(getattr(args, "epochs", 250))
        lr = float(getattr(args, "lr", 0.1))
        l2 = float(getattr(args, "l2", 0.01))
        threshold = float(getattr(args, "threshold", 0.50))

        n_samples = len(X)
        if n_samples >= 10:
            split_idx = int(n_samples * 0.70)
            X_train, y_train, r_train = X[:split_idx], y[:split_idx], r_vals[:split_idx]
            X_test, y_test, r_test = X[split_idx:], y[split_idx:], r_vals[split_idx:]
            has_split = True
        else:
            X_train, y_train, r_train = X, y, r_vals
            X_test, y_test, r_test = X, y, r_vals
            has_split = False

        trained_clf = LogisticMetaClassifier(threshold=threshold)
        train_metrics = trained_clf.fit(
            X=X_train,
            y=y_train,
            r_values=r_train,
            epochs=epochs,
            lr=lr,
            l2=l2,
            threshold=threshold,
        )

        test_metrics: ModelMetrics | None = None
        if has_split:
            test_probs = [trained_clf.predict_proba(row) for row in X_test]
            test_metrics = calculate_metrics(
                y_true=y_test,
                y_probs=test_probs,
                r_values=r_test,
                threshold=threshold,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        trained_clf.save_to_file(output_path)

        if getattr(args, "json", False):
            payload = {
                "status": "trained",
                "split": "temporal_70_30" if has_split else "none",
                "model_path": str(output_path),
                "metrics": test_metrics.to_dict() if test_metrics else train_metrics.to_dict(),
                "train_metrics": train_metrics.to_dict(),
            }
            if test_metrics:
                payload["test_metrics"] = test_metrics.to_dict()
            out.write(json.dumps(payload, indent=2) + "\n")
        else:
            extra = {
                "Modelo Salvo em": str(output_path),
                "Épocas": str(epochs),
                "Taxa de Aprendizado (lr)": str(lr),
                "Penalidade L2": str(l2),
                "Limiar": str(threshold),
            }
            report = _format_markdown_report(
                "Relatório de Treinamento: Meta-Labeling",
                train_metrics,
                extra,
                test_metrics=test_metrics,
            )
            out.write(report + "\n")
        return 0

    assert clf is not None

    threshold_arg = getattr(args, "threshold", None)
    threshold = float(threshold_arg) if threshold_arg is not None else clf.threshold

    probabilities = [clf.predict_proba(row) for row in X]
    metrics = calculate_metrics(
        y_true=y,
        y_probs=probabilities,
        r_values=r_vals,
        threshold=threshold,
    )

    if getattr(args, "json", False):
        payload = {
            "status": "evaluated",
            "model_path": str(model_path),
            "model_version": clf.version,
            "metrics": metrics.to_dict(),
            "eval_scope": "all" if getattr(args, "all", False) else "out_of_sample",
        }
        out.write(json.dumps(payload, indent=2) + "\n")
    else:
        extra = {
            "Modelo Avaliado": str(model_path),
            "Versão do Modelo": clf.version,
            "Treinado em": clf.trained_at.isoformat() if clf.trained_at else "N/A",
            "Limiar Avaliado": str(threshold),
            "Escopo de Avaliação": (
                "Completo (in-sample)" if getattr(args, "all", False) else "Out-of-sample"
            ),
        }
        report = _format_markdown_report("Relatório de Avaliação: Meta-Labeling", metrics, extra)
        out.write(report + "\n")
    return 0
