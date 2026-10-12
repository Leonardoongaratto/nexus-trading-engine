"""Tests for the Machine Learning Meta-Labeling subpackage (T-21)."""

from __future__ import annotations

import io
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from xau_bot.advisor.modes import ProposalStatus
from xau_bot.cli.main import main
from xau_bot.config import Settings
from xau_bot.domain.models import Direction
from xau_bot.journal.models import (
    JournalEntry,
    OutcomeData,
    SnapshotData,
    TradeOutcome,
)
from xau_bot.journal.repository import JournalRepository
from xau_bot.ml.classifier import LogisticMetaClassifier
from xau_bot.ml.evaluator import calculate_metrics, calculate_roc_auc, compute_log_loss
from xau_bot.ml.features import FeatureExtractor
from xau_bot.ml.filter import MetaLabelFilter
from xau_bot.ml.models import (
    MetaLabelDecision,
    MetaPrediction,
    ModelMetrics,
    StandardScalerParams,
    TradeFeatures,
)


def _create_sample_entry(
    entry_id: str,
    outcome: TradeOutcome,
    realized_r: float,
    displacement_atr: float = 2.0,
    fvg_size_atr: float = 1.0,
    confluences: tuple[str, ...] = ("H4_OB", "M15_CHOCH"),
) -> JournalEntry:
    now = datetime(2026, 10, 9, 10, 0, tzinfo=UTC)
    snap = SnapshotData(
        timestamp=now,
        session="LONDON",
        htf_bias="BULLISH",
        swing_high=2660.0,
        swing_low=2640.0,
        poi_type="ORDER_BLOCK",
        poi_level=2650.0,
        swept_liquidity=True,
        displacement_atr=displacement_atr,
        fvg_size_atr=fvg_size_atr,
        choch_level=2648.0,
        spread_price=0.25,
        atr_m5=1.5,
        news_distance_min=90.0,
        confluences=confluences,
        engine_version="0.1.0",
        params_hash="hash123",
        user_decision="APPROVED",
        user_reason="Valid setup",
    )
    out = OutcomeData(
        executed=True,
        entry_price=2650.0,
        stop_loss=2645.0,
        take_profit=2660.0,
        fill_price=2650.5,
        slippage_points=0.5,
        mae_r=-0.5,
        mfe_r=2.5,
        max_r=3.0,
        realized_r=realized_r,
        duration_seconds=300.0,
        exit_reason="Take Profit" if outcome == TradeOutcome.WIN else "Stop Loss",
    )
    return JournalEntry(
        id=f"j_{entry_id}",
        proposal_id=entry_id,
        created_at=now,
        direction=Direction.BULLISH,
        status=ProposalStatus.APPROVED,
        outcome=outcome,
        snapshot=snap,
        outcome_data=out,
    )


def test_trade_features_extraction_and_vectorization() -> None:
    entry = _create_sample_entry("prop_1", TradeOutcome.WIN, 2.0, displacement_atr=2.5)
    features = FeatureExtractor.extract_from_entry(entry)

    assert isinstance(features, TradeFeatures)
    assert features.displacement_atr == 2.5
    assert features.fvg_size_atr == 1.0
    assert pytest.approx(features.spread_atr_ratio) == 0.25 / 1.5
    assert features.session_code == 1.0
    assert features.news_distance_min == 90.0
    assert features.confluences_count == 2.0
    assert features.direction_code == 1.0

    vec = features.to_vector()
    assert len(vec) == 7
    assert vec[0] == 2.5


def test_standard_scaler_params_normalization() -> None:
    scaler = StandardScalerParams(means=[2.0, 10.0], stds=[1.0, 2.0])
    norm = scaler.transform([3.0, 14.0])
    assert pytest.approx(norm[0]) == 1.0
    assert pytest.approx(norm[1]) == 2.0


def test_logistic_classifier_training_and_convergence() -> None:
    X_train = [
        [3.0, 2.0, 0.1, 1.0, 120.0, 3.0, 1.0],
        [2.8, 1.9, 0.12, 1.0, 90.0, 2.0, 1.0],
        [3.5, 2.2, 0.08, 1.0, 150.0, 3.0, 1.0],
        [0.8, 0.4, 0.45, 0.0, 5.0, 0.0, -1.0],
        [0.9, 0.3, 0.50, 0.0, 10.0, 1.0, -1.0],
        [1.1, 0.5, 0.40, 0.0, 8.0, 0.0, -1.0],
    ]
    y_train = [1, 1, 1, 0, 0, 0]
    r_train = [2.5, 1.8, 3.2, -1.0, -1.0, -0.9]

    clf = LogisticMetaClassifier()
    metrics = clf.fit(X_train, y_train, r_values=r_train, epochs=300, lr=0.1, l2=0.01)

    assert isinstance(metrics, ModelMetrics)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.delta_r > 0.0

    p_high = clf.predict_proba([3.2, 2.1, 0.09, 1.0, 100.0, 3.0, 1.0])
    p_low = clf.predict_proba([0.7, 0.2, 0.60, 0.0, 4.0, 0.0, -1.0])
    assert p_high > 0.70
    assert p_low < 0.30

    dec_high = clf.decide([3.2, 2.1, 0.09, 1.0, 100.0, 3.0, 1.0], threshold=0.50)
    dec_low = clf.decide([0.7, 0.2, 0.60, 0.0, 4.0, 0.0, -1.0], threshold=0.50)
    assert dec_high.decision == MetaLabelDecision.ALLOW
    assert dec_low.decision == MetaLabelDecision.VETO


def test_roc_auc_calculation_properties() -> None:
    y_true = [1, 1, 0, 0]
    y_probs_perfect = [0.9, 0.8, 0.2, 0.1]
    assert calculate_roc_auc(y_true, y_probs_perfect) == 1.0

    y_probs_inverted = [0.1, 0.2, 0.8, 0.9]
    assert calculate_roc_auc(y_true, y_probs_inverted) == 0.0

    y_probs_tied = [0.5, 0.5, 0.5, 0.5]
    assert calculate_roc_auc(y_true, y_probs_tied) == 0.5


def test_model_serialization_roundtrip(tmp_path: Path) -> None:
    clf = LogisticMetaClassifier(feature_names=("feat_1", "feat_2"))
    X = [[1.0, 2.0], [2.0, 3.0], [0.1, 0.2], [0.2, 0.1]]
    y = [1, 1, 0, 0]
    r = [1.5, 2.0, -1.0, -1.0]
    clf.fit(X, y, r_values=r, epochs=100, lr=0.05)

    model_file = tmp_path / "meta_model.json"
    clf.save_to_file(model_file)

    loaded = LogisticMetaClassifier.load_from_file(model_file)
    assert loaded.to_dict() == clf.to_dict()

    orig_p = clf.predict_proba([1.5, 2.5])
    loaded_p = loaded.predict_proba([1.5, 2.5])
    assert pytest.approx(orig_p) == loaded_p


def test_meta_label_filter_runtime_decision_and_fallback() -> None:
    filter_disabled = MetaLabelFilter(settings=Settings(ml_meta_labeling_enabled=False))
    entry = _create_sample_entry("prop_1", TradeOutcome.WIN, 1.5)
    pred_disabled = filter_disabled.evaluate_entry(entry)
    assert pred_disabled.decision == MetaLabelDecision.ALLOW
    assert pred_disabled.is_fallback is True

    clf = LogisticMetaClassifier()
    good_entry = _create_sample_entry(
        "prop_good", TradeOutcome.WIN, 2.0, displacement_atr=3.5, fvg_size_atr=2.0
    )
    bad_entry = _create_sample_entry(
        "prop_bad",
        TradeOutcome.LOSS,
        -1.0,
        displacement_atr=0.4,
        fvg_size_atr=0.2,
        confluences=(),
    )
    X, y, _ = FeatureExtractor.extract_dataset([good_entry, bad_entry])
    clf.fit(X, y, epochs=200, lr=0.2)

    filter_active = MetaLabelFilter(
        classifier=clf,
        settings=Settings(ml_meta_labeling_enabled=True, ml_min_probability=0.55),
    )
    pred_good = filter_active.evaluate_entry(good_entry)
    assert pred_good.is_fallback is False
    assert pred_good.decision == MetaLabelDecision.ALLOW

    pred_bad = filter_active.evaluate_entry(bad_entry)
    assert pred_bad.decision == MetaLabelDecision.VETO


def test_meta_label_filter_from_settings_and_file_not_found(tmp_path: Path) -> None:
    non_existent = tmp_path / "model_missing.json"
    settings = Settings(ml_meta_labeling_enabled=True, ml_model_path=str(non_existent))
    filter_instance = MetaLabelFilter.from_settings(settings)
    assert filter_instance.classifier is None

    entry = _create_sample_entry("prop_fallback", TradeOutcome.WIN, 1.0)
    pred = filter_instance.evaluate_entry(entry)
    assert pred.decision == MetaLabelDecision.ALLOW
    assert pred.is_fallback is True

    valid_model_path = tmp_path / "valid_model.json"
    clf = LogisticMetaClassifier(threshold=0.6)
    good_entry = _create_sample_entry("g1", TradeOutcome.WIN, 2.0)
    bad_entry = _create_sample_entry("b1", TradeOutcome.LOSS, -1.0)
    X, y, _ = FeatureExtractor.extract_dataset([good_entry, bad_entry])
    clf.fit(X, y, epochs=10)
    clf.save_to_file(valid_model_path)

    settings_loaded = Settings(ml_meta_labeling_enabled=True, ml_model_path=str(valid_model_path))
    filter_loaded = MetaLabelFilter.from_settings(settings_loaded)
    assert filter_loaded.classifier is not None


def test_ml_cli_insufficient_data(tmp_path: Path) -> None:
    empty_db = tmp_path / "empty.db"
    out = io.StringIO()
    err = io.StringIO()

    exit_code = main(["ml", "train", "--db", str(empty_db)], stdout=out, stderr=err)
    assert exit_code == 0
    assert "AVISO: Dados insuficientes" in out.getvalue()

    out_json = io.StringIO()
    exit_code_json = main(
        ["ml", "train", "--db", str(empty_db), "--json"],
        stdout=out_json,
        stderr=err,
    )
    assert exit_code_json == 0
    data = json.loads(out_json.getvalue())
    assert data["status"] == "insufficient_data"


def test_ml_cli_train_and_evaluate_workflow(tmp_path: Path) -> None:
    db_path = tmp_path / "test_journal.db"
    repo = JournalRepository(db_path=str(db_path))

    for i in range(15):
        w = _create_sample_entry(f"win_{i}", TradeOutcome.WIN, 2.0, displacement_atr=3.0)
        repo.save_entry(w)
    for i in range(15):
        loss_entry = _create_sample_entry(
            f"loss_{i}",
            TradeOutcome.LOSS,
            -1.0,
            displacement_atr=0.3,
            fvg_size_atr=0.1,
            confluences=(),
        )
        repo.save_entry(loss_entry)

    model_path = tmp_path / "models" / "meta_model.json"
    out_train_json = io.StringIO()
    err = io.StringIO()

    exit_code = main(
        [
            "ml",
            "train",
            "--db",
            str(db_path),
            "--output",
            str(model_path),
            "--epochs",
            "50",
            "--threshold",
            "0.50",
            "--json",
        ],
        stdout=out_train_json,
        stderr=err,
    )
    assert exit_code == 0
    train_res = json.loads(out_train_json.getvalue())
    assert train_res["status"] == "trained"
    assert model_path.exists()
    assert "accuracy" in train_res["metrics"]

    out_train_md = io.StringIO()
    exit_code_md = main(
        [
            "ml",
            "train",
            "--db",
            str(db_path),
            "--output",
            str(model_path),
            "--epochs",
            "20",
        ],
        stdout=out_train_md,
        stderr=err,
    )
    assert exit_code_md == 0
    assert "# Relatório de Treinamento: Meta-Labeling" in out_train_md.getvalue()

    out_eval_json = io.StringIO()
    exit_code_eval = main(
        [
            "ml",
            "evaluate",
            "--db",
            str(db_path),
            "--model",
            str(model_path),
            "--all",
            "--json",
        ],
        stdout=out_eval_json,
        stderr=err,
    )
    assert exit_code_eval == 0
    eval_res = json.loads(out_eval_json.getvalue())
    assert eval_res["status"] == "evaluated"
    assert eval_res["metrics"]["sample_size"] == 30

    out_eval_md = io.StringIO()
    exit_code_eval_md = main(
        [
            "ml",
            "evaluate",
            "--db",
            str(db_path),
            "--model",
            str(model_path),
            "--all",
        ],
        stdout=out_eval_md,
        stderr=err,
    )
    assert exit_code_eval_md == 0
    assert "# Relatório de Avaliação: Meta-Labeling" in out_eval_md.getvalue()


def test_ml_cli_train_temporal_split_and_out_of_sample_metrics(tmp_path: Path) -> None:
    db_path = tmp_path / "temporal_journal.db"
    repo = JournalRepository(db_path=str(db_path))

    base_time = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    for i in range(21):
        dt = base_time.replace(day=1 + i)
        outcome = TradeOutcome.WIN if i % 2 == 0 else TradeOutcome.LOSS
        entry = _create_sample_entry(
            f"train_entry_{i}", outcome, 2.0 if outcome == TradeOutcome.WIN else -1.0
        )
        entry = JournalEntry(
            id=entry.id,
            proposal_id=entry.proposal_id,
            created_at=dt,
            direction=entry.direction,
            status=entry.status,
            outcome=entry.outcome,
            snapshot=entry.snapshot,
            outcome_data=entry.outcome_data,
        )
        repo.save_entry(entry)

    for i in range(9):
        dt = base_time.replace(day=22 + i)
        entry = _create_sample_entry(f"test_entry_{i}", TradeOutcome.LOSS, -1.0)
        entry = JournalEntry(
            id=entry.id,
            proposal_id=entry.proposal_id,
            created_at=dt,
            direction=entry.direction,
            status=entry.status,
            outcome=entry.outcome,
            snapshot=entry.snapshot,
            outcome_data=entry.outcome_data,
        )
        repo.save_entry(entry)

    model_path = tmp_path / "models" / "temporal_model.json"
    out_json = io.StringIO()
    err = io.StringIO()

    exit_code = main(
        [
            "ml",
            "train",
            "--db",
            str(db_path),
            "--output",
            str(model_path),
            "--epochs",
            "50",
            "--json",
        ],
        stdout=out_json,
        stderr=err,
    )
    assert exit_code == 0
    res = json.loads(out_json.getvalue())
    assert res["status"] == "trained"
    assert res["split"] == "temporal_70_30"
    assert res["train_metrics"]["sample_size"] == 21
    assert res["test_metrics"]["sample_size"] == 9
    assert res["test_metrics"]["positive_rate"] == 0.0


def test_ml_cli_evaluate_model_not_found(tmp_path: Path) -> None:
    db_path = tmp_path / "journal.db"
    repo = JournalRepository(db_path=str(db_path))
    repo.save_entry(_create_sample_entry("w1", TradeOutcome.WIN, 2.0))
    repo.save_entry(_create_sample_entry("l1", TradeOutcome.LOSS, -1.0))

    missing_model = tmp_path / "not_found.json"
    out = io.StringIO()
    err = io.StringIO()
    exit_code = main(
        ["ml", "evaluate", "--db", str(db_path), "--model", str(missing_model)],
        stdout=out,
        stderr=err,
    )
    assert exit_code == 1
    assert "Arquivo de modelo não encontrado" in err.getvalue()


def test_feature_extractor_sessions_and_dataset_filtering() -> None:
    assert FeatureExtractor._map_session_code("ASIA_SESSION") == 0.0
    assert FeatureExtractor._map_session_code("NEW_YORK_AM") == 2.0
    assert FeatureExtractor._map_session_code("LONDON_CLOSE") == 3.0
    assert FeatureExtractor._map_session_code("UNKNOWN_TZ") == 4.0

    valid_entry = _create_sample_entry("valid", TradeOutcome.WIN, 1.5)
    unexecuted_entry = _create_sample_entry("unexec", TradeOutcome.UNEXECUTED, 0.0)
    no_outcome_entry = JournalEntry(
        id="no_out",
        proposal_id="no_out",
        snapshot=valid_entry.snapshot,
        outcome=TradeOutcome.WIN,
        outcome_data=None,
    )

    X, y, r_vals = FeatureExtractor.extract_dataset(
        [valid_entry, unexecuted_entry, no_outcome_entry]
    )
    assert len(X) == 1
    assert y == [1]
    assert r_vals == [1.5]


def test_logistic_meta_classifier_log_loss_convergence_and_l2_shrinkage() -> None:
    clf = LogisticMetaClassifier(feature_names=("f1", "f2"))
    X = [[2.0, 1.5], [1.8, 1.2], [-1.5, -2.0], [-1.2, -1.8]]
    y = [1, 1, 0, 0]

    clf.fit(X, y, epochs=50, lr=0.1, l2=0.01)
    assert len(clf.loss_history) > 1
    assert clf.loss_history[-1] < clf.loss_history[0]

    clf_unreg = LogisticMetaClassifier(feature_names=("f1", "f2"))
    clf_unreg.fit(X, y, epochs=100, lr=0.1, l2=0.0)
    norm_unreg = math.sqrt(sum(w**2 for w in clf_unreg.weights))

    clf_reg = LogisticMetaClassifier(feature_names=("f1", "f2"))
    clf_reg.fit(X, y, epochs=100, lr=0.1, l2=0.5)
    norm_reg = math.sqrt(sum(w**2 for w in clf_reg.weights))

    assert norm_reg < norm_unreg


def test_calculate_metrics_exact_deterministic_values() -> None:
    y_true = [1, 1, 0, 0]
    y_probs = [0.9, 0.4, 0.6, 0.1]
    r_vals = [2.0, 1.0, -1.0, -1.0]

    metrics = calculate_metrics(y_true=y_true, y_probs=y_probs, r_values=r_vals, threshold=0.5)
    assert metrics.sample_size == 4
    assert metrics.positive_rate == 0.5
    assert metrics.accuracy == 0.5
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert pytest.approx(0.185) == metrics.brier_score
    assert pytest.approx(0.25) == metrics.delta_r

    loss = compute_log_loss(y_true, y_probs)
    expected_loss = (
        -(math.log(0.9) + math.log(0.4) + math.log(1.0 - 0.6) + math.log(1.0 - 0.1)) / 4.0
    )
    assert pytest.approx(expected_loss) == loss


def test_logistic_meta_classifier_dimension_validation() -> None:
    invalid_data = {
        "version": "1.0.0",
        "threshold": 0.5,
        "weights": [0.1, 0.2],
        "bias": 0.0,
        "scaler": {"means": [0.0, 0.0, 0.0], "stds": [1.0, 1.0, 1.0]},
        "feature_names": ["f1", "f2"],
    }
    with pytest.raises(ValueError, match="Model dimension mismatch"):
        LogisticMetaClassifier.from_dict(invalid_data)


def test_ml_cli_evaluate_missing_db(tmp_path: Path) -> None:
    valid_model_file = tmp_path / "model.json"
    clf = LogisticMetaClassifier(threshold=0.5)
    clf.save_to_file(valid_model_file)

    missing_db = tmp_path / "not_there.db"
    out = io.StringIO()
    err = io.StringIO()
    exit_code = main(
        ["ml", "evaluate", "--db", str(missing_db), "--model", str(valid_model_file)],
        stdout=out,
        stderr=err,
    )
    assert exit_code == 1
    assert "Arquivo de banco de dados não encontrado" in err.getvalue()


def test_meta_label_filter_incompatible_feature_names_fail_open() -> None:
    incompatible_clf = LogisticMetaClassifier(feature_names=("feat_a", "feat_b"))
    filter_instance = MetaLabelFilter(classifier=incompatible_clf)
    assert filter_instance.classifier is None
    assert filter_instance.is_active is False

    entry = _create_sample_entry("prop_fallback", TradeOutcome.WIN, 1.0)
    pred = filter_instance.evaluate_entry(entry)
    assert pred.decision == MetaLabelDecision.ALLOW
    assert pred.is_fallback is True


def test_compute_log_loss_length_mismatch_raises_error() -> None:
    with pytest.raises(ValueError, match="Lengths of y_true and y_probs must match"):
        compute_log_loss([1, 0], [0.5])


def test_meta_label_filter_exception_fail_open(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(ml_meta_labeling_enabled=True)
    clf = LogisticMetaClassifier(threshold=0.5)
    filter_instance = MetaLabelFilter(classifier=clf, settings=settings)
    assert filter_instance.is_active is True

    def _raising_decide(*args: object, **kwargs: object) -> MetaPrediction:
        raise RuntimeError("Falha simulada de inferência")

    monkeypatch.setattr(clf, "decide", _raising_decide)

    entry = _create_sample_entry("prop_exc", TradeOutcome.WIN, 1.0)
    pred = filter_instance.evaluate_entry(entry)
    assert pred.decision == MetaLabelDecision.ALLOW
    assert pred.is_fallback is True
    assert pred.probability == 1.0


def test_meta_label_filter_corrupted_model_file(tmp_path: Path) -> None:
    bad_model = tmp_path / "bad_model.json"
    bad_model.write_text("invalid json content")
    settings = Settings(ml_model_path=str(bad_model), ml_meta_labeling_enabled=True)
    filter_instance = MetaLabelFilter(settings=settings)
    assert filter_instance.classifier is None
    assert filter_instance.is_active is False


def test_ml_cli_evaluate_out_of_sample_lookahead_prevention(tmp_path: Path) -> None:
    db_path = tmp_path / "lookahead_journal.db"
    model_path = tmp_path / "model_lookahead.json"
    repo = JournalRepository(db_path=str(db_path))

    past_base = datetime(2026, 8, 1, 10, 0, tzinfo=UTC)
    for i in range(12):
        dt = past_base.replace(day=1 + i)
        outcome = TradeOutcome.WIN if i % 2 == 0 else TradeOutcome.LOSS
        entry = _create_sample_entry(
            f"hist_{i}", outcome, 1.5 if outcome == TradeOutcome.WIN else -1.0
        )
        entry = JournalEntry(
            id=entry.id,
            proposal_id=entry.proposal_id,
            created_at=dt,
            direction=entry.direction,
            status=entry.status,
            outcome=entry.outcome,
            snapshot=entry.snapshot,
            outcome_data=entry.outcome_data,
        )
        repo.save_entry(entry)
    repo.close()

    err = io.StringIO()
    exit_train = main(
        ["ml", "train", "--db", str(db_path), "--output", str(model_path), "--epochs", "10"],
        stdout=io.StringIO(),
        stderr=err,
    )
    assert exit_train == 0
    assert model_path.exists()

    clf = LogisticMetaClassifier.load_from_file(model_path)
    assert clf.trained_at is not None

    out_eval_empty_json = io.StringIO()
    exit_eval_empty = main(
        ["ml", "evaluate", "--db", str(db_path), "--model", str(model_path), "--json"],
        stdout=out_eval_empty_json,
        stderr=err,
    )
    assert exit_eval_empty == 0
    res_empty = json.loads(out_eval_empty_json.getvalue())
    assert res_empty["status"] == "insufficient_data"
    assert "Nenhum dado out-of-sample" in res_empty["message"]

    out_eval_empty_md = io.StringIO()
    exit_eval_empty_md = main(
        ["ml", "evaluate", "--db", str(db_path), "--model", str(model_path)],
        stdout=out_eval_empty_md,
        stderr=err,
    )
    assert exit_eval_empty_md == 0
    assert "Nenhum dado out-of-sample posterior ao treinamento" in out_eval_empty_md.getvalue()

    repo2 = JournalRepository(db_path=str(db_path))
    for i in range(6):
        future_dt = clf.trained_at + timedelta(minutes=10 * (i + 1))
        outcome = TradeOutcome.WIN if i % 2 == 0 else TradeOutcome.LOSS
        entry = _create_sample_entry(
            f"future_{i}", outcome, 2.0 if outcome == TradeOutcome.WIN else -1.0
        )
        entry = JournalEntry(
            id=entry.id,
            proposal_id=entry.proposal_id,
            created_at=future_dt,
            direction=entry.direction,
            status=entry.status,
            outcome=entry.outcome,
            snapshot=entry.snapshot,
            outcome_data=entry.outcome_data,
        )
        repo2.save_entry(entry)
    repo2.close()

    out_eval_oos = io.StringIO()
    exit_eval_oos = main(
        ["ml", "evaluate", "--db", str(db_path), "--model", str(model_path), "--json"],
        stdout=out_eval_oos,
        stderr=err,
    )
    assert exit_eval_oos == 0
    res_oos = json.loads(out_eval_oos.getvalue())
    assert res_oos["status"] == "evaluated"
    assert res_oos["eval_scope"] == "out_of_sample"
    assert res_oos["metrics"]["sample_size"] == 6

    out_eval_all = io.StringIO()
    exit_eval_all = main(
        ["ml", "evaluate", "--db", str(db_path), "--model", str(model_path), "--all", "--json"],
        stdout=out_eval_all,
        stderr=err,
    )
    assert exit_eval_all == 0
    res_all = json.loads(out_eval_all.getvalue())
    assert res_all["status"] == "evaluated"
    assert res_all["eval_scope"] == "all"
    assert res_all["metrics"]["sample_size"] == 18
