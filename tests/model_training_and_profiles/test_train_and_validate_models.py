import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from llm_behavior_xai.model_training_and_profiles.train_and_validate_models import (
    build_model_candidates,
    encode_target,
    fit_with_params,
    metric_row,
    normalize_label,
    prediction_context,
    safe_filename_part,
    train_and_validate_models,
)


@pytest.fixture
def mock_splits():
    np.random.seed(42)
    train_size, val_size, test_size = 20, 10, 10

    feature_splits = {
        "train": pd.DataFrame({"f1": np.random.rand(train_size), "f2": np.random.rand(train_size)}),
        "val": pd.DataFrame({"f1": np.random.rand(val_size), "f2": np.random.rand(val_size)}),
        "test": pd.DataFrame({"f1": np.random.rand(test_size), "f2": np.random.rand(test_size)}),
    }

    final_splits = {
        "train": feature_splits["train"].copy(),
        "val": feature_splits["val"].copy(),
        "test": feature_splits["test"].copy(),
    }

    final_splits["train"]["target_col"] = np.random.choice(["Class_A", "Class_B"], train_size)
    final_splits["val"]["target_col"] = np.random.choice(["Class_A", "Class_B"], val_size)
    final_splits["test"]["target_col"] = np.random.choice(["Class_A", "Class_B"], test_size)

    final_splits["train"].loc[final_splits["train"].index[:2], "target_col"] = ["Class_A", "Class_B"]
    final_splits["val"].loc[final_splits["val"].index[:2], "target_col"] = ["Class_A", "Class_B"]
    final_splits["test"].loc[final_splits["test"].index[:2], "target_col"] = ["Class_A", "Class_B"]

    return feature_splits, final_splits


class TestDataHelpers:
    def test_normalize_label(self):
        assert normalize_label(np.nan) == "missing"
        assert normalize_label(pd.NA) == "missing"
        assert normalize_label(True) == "True"
        assert normalize_label(False) == "False"
        assert normalize_label(np.bool_(True)) == "True"
        assert normalize_label("Valid Label") == "Valid Label"
        assert normalize_label(123) == "123"

    def test_prediction_context(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
        result = prediction_context(df, ("A", "C", "D"))

        # 'D' should be ignored because it's not in the dataframe
        assert list(result.columns) == ["A", "C"]
        # Ensure it's a copy
        assert result is not df[list(result.columns)]

    def test_safe_filename_part(self):
        assert safe_filename_part("Hello World!") == "hello_world"
        assert safe_filename_part("Valid_Name-123") == "valid_name_123"
        assert safe_filename_part("!@#$") == "class"
        assert safe_filename_part("") == "class"


class TestEncodingAndMetrics:
    """Tests for label encoding and metric row generation."""

    def test_encode_target(self, mock_splits):
        _, final_splits = mock_splits

        encoder, encoded_targets = encode_target("target_col", final_splits)

        assert isinstance(encoder, LabelEncoder)
        # assert list(encoder.classes_) == ["Class_A", "Class_B"]
        assert "train" in encoded_targets
        assert "val" in encoded_targets
        assert "test" in encoded_targets
        assert isinstance(encoded_targets["train"], np.ndarray)
        assert set(encoded_targets["train"]).issubset({0, 1})

    def test_metric_row(self):
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = pd.Series([0, 1, 0, 0])
        class_names = ["Class_A", "Class_B"]
        params = {"C": 1.0}

        row = metric_row("my_target", "val", "log_reg", params, y_true, y_pred, class_names)

        assert row["target"] == "my_target"
        assert row["split"] == "val"
        assert row["model_name"] == "log_reg"
        assert row["params"] == json.dumps(params, sort_keys=True)
        assert row["accuracy"] == 0.75
        assert isinstance(row["macro_f1"], float)
        assert isinstance(row["balanced_accuracy"], float)
        assert "Class_A" in row["classification_report"]


class TestModelTraining:
    """Tests for model candidate generation, parameter fitting, and the main pipeline."""

    def test_build_model_candidates(self):
        candidates = build_model_candidates(random_state=42)

        assert "logistic_regression" in candidates
        assert "random_forest" in candidates

        lr_estimator, lr_params = candidates["logistic_regression"]
        assert hasattr(lr_estimator, "fit")
        assert "model__C" in lr_params

    def test_fit_with_params(self):
        base_estimator = LogisticRegression(max_iter=100)
        params = {"C": 5.0, "max_iter": 200}

        model = fit_with_params(base_estimator, params)

        # Ensure it's cloned properly and params are updated
        assert model is not base_estimator
        assert model.C == 5.0
        assert model.max_iter == 200
        # Ensure original is unchanged
        assert base_estimator.C == 1.0

    def test_train_and_validate_models(self, mock_splits, tmp_path: Path):
        feature_splits, final_splits = mock_splits

        best_by_target, validation_metrics = train_and_validate_models(
            random_state=42,
            selection_metric="macro_f1",
            target_columns=("target_col",),
            xai_dir=tmp_path,
            feature_splits=feature_splits,
            final_splits=final_splits,
        )

        # 1. Check returned dictionary
        assert "target_col" in best_by_target
        best_info = best_by_target["target_col"]
        assert isinstance(best_info["encoder"], LabelEncoder)
        assert "model_name" in best_info
        assert hasattr(best_info["model"], "predict")

        # 2. Check validation metrics dataframe
        assert isinstance(validation_metrics, pd.DataFrame)
        assert "target" in validation_metrics.columns
        assert "macro_f1" in validation_metrics.columns

        # 3. Check if CSV file was saved
        saved_csv_path = tmp_path / "validation_tuning_metrics.csv"
        assert saved_csv_path.exists()

        # Read back to ensure format is valid
        saved_df = pd.read_csv(saved_csv_path)
        assert len(saved_df) == len(validation_metrics)
