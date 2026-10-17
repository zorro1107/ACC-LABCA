import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

import predictor as predictor_module
from feature_pipeline import (
    chronological_train_val_test_split,
    engineer_features,
)
from forecast_config import (
    MIN_HISTORY_SAMPLES,
    MODEL_PATH,
    PREDICTION_HORIZON_SAMPLES,
    PREDICTION_HORIZON_SECONDS,
    SAMPLE_INTERVAL_SECONDS,
)


def _artifact(path, prediction=42.125):
    history = list(np.linspace(20.0, 30.0, MIN_HISTORY_SAMPLES))
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2000-01-01",
                periods=MIN_HISTORY_SAMPLES,
                freq=f"{SAMPLE_INTERVAL_SECONDS}s",
            ),
            "cpu_usage": history,
            "memory_usage": [50.0] * MIN_HISTORY_SAMPLES,
            "active_processes": [100] * MIN_HISTORY_SAMPLES,
            "disk_io_mb_s": [1.0] * MIN_HISTORY_SAMPLES,
        }
    )
    _, feature_cols, _ = engineer_features(raw, dropna=False)
    model = DummyRegressor(strategy="constant", constant=prediction)
    model.fit(pd.DataFrame([[0.0] * len(feature_cols)], columns=feature_cols), [prediction])
    joblib.dump(
        {
            "model": model,
            "model_name": "test model",
            "feature_cols": feature_cols,
            "test_metrics": {},
            "sampling_interval_seconds": SAMPLE_INTERVAL_SECONDS,
            "prediction_horizon_seconds": PREDICTION_HORIZON_SECONDS,
            "prediction_horizon_samples": PREDICTION_HORIZON_SAMPLES,
        },
        path,
    )


@pytest.fixture(autouse=True)
def clear_default_predictor_cache():
    predictor_module._default_predictor.cache_clear()
    yield
    predictor_module._default_predictor.cache_clear()


def test_public_prediction_returns_float(monkeypatch, tmp_path):
    artifact_path = tmp_path / "model.joblib"
    _artifact(artifact_path)
    monkeypatch.setattr(predictor_module, "MODEL_PATH", artifact_path)

    prediction = predictor_module.predict_workload([25.0] * MIN_HISTORY_SAMPLES)

    assert type(prediction) is float
    assert prediction == 42.12


def test_feature_pipeline_targets_exactly_five_minutes_ahead():
    assert SAMPLE_INTERVAL_SECONDS == 5
    assert PREDICTION_HORIZON_SECONDS == 300
    assert PREDICTION_HORIZON_SAMPLES == 60

    raw = pd.DataFrame({"cpu_usage": np.arange(130, dtype=float)})
    featured, _, target_col = engineer_features(raw, dropna=False)
    assert target_col == "target_cpu_future_60"
    assert featured.loc[10, target_col] == 70.0


def test_chronological_split_purges_horizon_between_partitions():
    row_ids = np.arange(1000)
    data = pd.DataFrame({"row_id": row_ids, "target": row_ids + PREDICTION_HORIZON_SAMPLES})

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_train_val_test_split(
        data,
        feature_cols=["row_id"],
        target_col="target",
        gap_steps=PREDICTION_HORIZON_SAMPLES,
    )

    assert y_train.iloc[-1] < X_val.iloc[0]["row_id"]
    assert y_val.iloc[-1] < X_test.iloc[0]["row_id"]
    assert X_train["row_id"].is_monotonic_increasing
    assert X_val["row_id"].is_monotonic_increasing
    assert X_test["row_id"].is_monotonic_increasing


@pytest.mark.parametrize(
    "history",
    [
        [],
        [25.0] * (MIN_HISTORY_SAMPLES - 1),
        [25.0] * (MIN_HISTORY_SAMPLES - 1) + ["not numeric"],
        [25.0] * (MIN_HISTORY_SAMPLES - 1) + [float("nan")],
        [25.0] * (MIN_HISTORY_SAMPLES - 1) + [float("inf")],
        [25.0] * (MIN_HISTORY_SAMPLES - 1) + [-0.1],
        [25.0] * (MIN_HISTORY_SAMPLES - 1) + [100.1],
    ],
)
def test_public_prediction_rejects_invalid_history(history):
    with pytest.raises(ValueError):
        predictor_module.predict_workload(history)


def test_public_prediction_rejects_non_sequence():
    with pytest.raises(TypeError, match="sequence"):
        predictor_module.predict_workload(25.0)


def test_missing_model_has_actionable_error(monkeypatch, tmp_path):
    missing_path = tmp_path / "missing.joblib"
    monkeypatch.setattr(predictor_module, "MODEL_PATH", missing_path)

    with pytest.raises(FileNotFoundError, match="Train it with"):
        predictor_module.predict_workload([25.0] * MIN_HISTORY_SAMPLES)


def test_corrupt_model_has_clear_error(monkeypatch, tmp_path):
    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"not a joblib artifact")
    monkeypatch.setattr(predictor_module, "MODEL_PATH", corrupt_path)

    with pytest.raises(RuntimeError, match="corrupt or incompatible"):
        predictor_module.predict_workload([25.0] * MIN_HISTORY_SAMPLES)


def test_incompatible_model_metadata_is_rejected(monkeypatch, tmp_path):
    incompatible_path = tmp_path / "incompatible.joblib"
    joblib.dump({"model": object()}, incompatible_path)
    monkeypatch.setattr(predictor_module, "MODEL_PATH", incompatible_path)

    with pytest.raises(ValueError, match="missing metadata"):
        predictor_module.predict_workload([25.0] * MIN_HISTORY_SAMPLES)


def test_model_path_is_project_relative():
    assert MODEL_PATH.name == "model.joblib"
    assert MODEL_PATH.parent.name == "ml"
