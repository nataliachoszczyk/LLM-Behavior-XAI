import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from llm_behavior_xai.model_training_and_profiles.metrics import (
    calculate_final_metrics,
    save_confusion_matrix,
    save_predictions,
)


@pytest.fixture
def sample_prediction_data():
    return {
        "source_df": pd.DataFrame(
            {
                "text": ["a", "b"],
                "id": [1, 2],
            }
        ),
        "y_true": pd.Series([0, 1]),
        "y_pred": pd.Series([0, 0]),
        "probabilities": np.array([[0.9, 0.1], [0.8, 0.2]]),
        "class_names": ["negative", "positive"],
    }


class TestSavePredictions:
    def test_save_predictions_creates_csv(self, tmp_path, sample_prediction_data):
        predictions_dir = tmp_path / "predictions"
        predictions_dir.mkdir()

        encoder = MagicMock()
        encoder.inverse_transform.side_effect = lambda values: [
            "negative" if value == 0 else "positive" for value in values
        ]

        best_by_target = {
            "sentiment": {
                "encoder": encoder,
            }
        }

        with patch("llm_behavior_xai.model_training_and_profiles.metrics.prediction_context") as mock_prediction_context:
            mock_prediction_context.return_value = sample_prediction_data["source_df"].copy()

            save_predictions(
                target="sentiment",
                split="test",
                source_df=sample_prediction_data["source_df"],
                y_true=sample_prediction_data["y_true"],
                y_pred=sample_prediction_data["y_pred"],
                probabilities=sample_prediction_data["probabilities"],
                class_names=sample_prediction_data["class_names"],
                best_by_target=best_by_target,
                xai_dir=tmp_path,
                prediction_context_columns=("text",),
            )

        output_file = predictions_dir / "sentiment_test_predictions.csv"

        assert output_file.exists()

        result_df = pd.read_csv(output_file)

        assert "target" in result_df.columns
        assert "split" in result_df.columns
        assert "y_true" in result_df.columns
        assert "y_pred" in result_df.columns
        assert "correct" in result_df.columns
        assert "prediction_confidence" in result_df.columns

        assert result_df.loc[0, "correct"] is np.True_
        assert result_df.loc[1, "correct"] is np.False_

        assert "probability_negative" in result_df.columns
        assert "probability_positive" in result_df.columns


class TestSaveConfusionMatrix:
    def test_save_confusion_matrix_creates_files(self, tmp_path):
        confusion_dir = tmp_path / "confusion_matrices"
        confusion_dir.mkdir()

        y_true = pd.Series([0, 1, 1, 0])
        y_pred = pd.Series([0, 1, 0, 0])
        class_names = ["negative", "positive"]

        save_confusion_matrix(
            target="sentiment",
            split="test",
            y_true=y_true,
            y_pred=y_pred,
            class_names=class_names,
            xai_dir=tmp_path,
        )

        csv_file = confusion_dir / "sentiment_test_confusion_matrix.csv"
        png_file = confusion_dir / "sentiment_test_confusion_matrix.png"

        assert csv_file.exists()
        assert png_file.exists()

        matrix_df = pd.read_csv(csv_file, index_col=0)

        assert matrix_df.shape == (2, 2)
        assert matrix_df.loc["negative", "negative"] == 2
        assert matrix_df.loc["positive", "negative"] == 1


class TestCalculateFinalMetrics:
    @patch("llm_behavior_xai.model_training_and_profiles.metrics.metric_row")
    @patch("llm_behavior_xai.model_training_and_profiles.metrics.save_predictions")
    @patch("llm_behavior_xai.model_training_and_profiles.metrics.save_confusion_matrix")
    @patch("llm_behavior_xai.model_training_and_profiles.metrics.joblib.dump")
    def test_calculate_final_metrics(
        self,
        mock_joblib_dump,
        mock_save_confusion_matrix,
        mock_save_predictions,
        mock_metric_row,
        tmp_path,
    ):
        models_dir = tmp_path / "models"
        xai_dir = tmp_path / "xai"

        models_dir.mkdir()
        xai_dir.mkdir()

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([0, 1])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

        mock_metric_row.return_value = {
            "target": "sentiment",
            "split": "train",
            "accuracy": 0.95,
        }

        best_by_target = {
            "sentiment": {
                "model": mock_model,
                "class_names": ["negative", "positive"],
                "encoded_targets": {
                    "train": pd.Series([0, 1]),
                    "val": pd.Series([1, 0]),
                    "test": pd.Series([0, 0]),
                },
                "model_name": "RandomForest",
                "params": {"n_estimators": 10},
                "encoder": MagicMock(),
            }
        }

        feature_splits = {
            "train": pd.DataFrame({"x": [1, 2]}),
            "val": pd.DataFrame({"x": [3, 4]}),
            "test": pd.DataFrame({"x": [5, 6]}),
        }

        final_splits = {
            "train": pd.DataFrame({"text": ["a", "b"]}),
            "val": pd.DataFrame({"text": ["c", "d"]}),
            "test": pd.DataFrame({"text": ["e", "f"]}),
        }

        result_df = calculate_final_metrics(
            models_dir=models_dir,
            prediction_context_columns=("text",),
            selection_metric="accuracy",
            xai_dir=xai_dir,
            best_by_target=best_by_target,
            feature_columns=["x"],
            feature_splits=feature_splits,
            final_splits=final_splits,
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3

        assert mock_metric_row.call_count == 3
        assert mock_save_predictions.call_count == 3
        assert mock_save_confusion_matrix.call_count == 3

        mock_joblib_dump.assert_called_once()

        metadata_file = models_dir / "sentiment_metadata.json"
        assert metadata_file.exists()

        with metadata_file.open(encoding="utf-8") as file:
            metadata = json.load(file)

        assert metadata["target"] == "sentiment"
        assert metadata["model_name"] == "RandomForest"

        metrics_csv = xai_dir / "all_metrics.csv"
        assert metrics_csv.exists()
