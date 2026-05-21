import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

from llm_behavior_xai.model_training_and_profiles.surrogate_decision_tree import (
    train_surrogate_tree,
    calculate_surrogate_decision_tree_metrics,
)


@pytest.fixture
def mock_setup(tmp_path: Path):
    """Fixture to provide mock directories, data, and models for testing."""
    # Setup directories
    models_dir = tmp_path / "models"
    xai_dir = tmp_path / "xai"
    models_dir.mkdir()
    xai_dir.mkdir()
    (xai_dir / "surrogate_trees").mkdir()

    # Generate dummy features
    np.random.seed(42)
    feature_columns = ["feature_1", "feature_2"]

    # Needs enough samples to satisfy min_samples_leaf=30 in the default args
    X_train = pd.DataFrame(np.random.rand(100, 2), columns=feature_columns)
    X_val = pd.DataFrame(np.random.rand(30, 2), columns=feature_columns)
    X_test = pd.DataFrame(np.random.rand(30, 2), columns=feature_columns)

    feature_splits = {"train": X_train, "val": X_val, "test": X_test}

    # Generate dummy targets
    y_train = np.random.randint(0, 2, 100)
    y_val = np.random.randint(0, 2, 30)
    y_test = np.random.randint(0, 2, 30)
    encoded_targets = {"train": y_train, "val": y_val, "test": y_test}

    # Setup a mock "main model"
    main_model = DecisionTreeClassifier(max_depth=2, random_state=42)
    main_model.fit(X_train, y_train)

    best_by_target = {
        "target_A": {
            "model": main_model,
            "class_names": ["Class_0", "Class_1"],
            "encoded_targets": encoded_targets,
        },
        "target_B": {
            "model": main_model,
            "class_names": ["Category_X", "Category_Y"],
            "encoded_targets": encoded_targets,
        },
    }

    return {
        "models_dir": models_dir,
        "xai_dir": xai_dir,
        "feature_splits": feature_splits,
        "feature_columns": feature_columns,
        "best_by_target": best_by_target,
        "random_state": 42,
    }


class TestTrainSurrogateTree:
    def test_train_surrogate_tree_outputs(self, mock_setup):
        """Test if the training function outputs correct DataFrames and saves artifacts."""
        target = "target_A"

        result_df = train_surrogate_tree(
            target=target,
            best_by_target=mock_setup["best_by_target"],
            feature_splits=mock_setup["feature_splits"],
            feature_columns=mock_setup["feature_columns"],
            random_state=mock_setup["random_state"],
            models_dir=mock_setup["models_dir"],
            xai_dir=mock_setup["xai_dir"],
            max_depth=2,
            min_samples_leaf=10,
        )

        # 1. Check returned DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3  # One row for train, val, and test
        assert list(result_df["split"]) == ["train", "val", "test"]
        assert all(result_df["target"] == target)

        expected_columns = [
            "target",
            "split",
            "surrogate_max_depth",
            "surrogate_min_samples_leaf",
            "fidelity_accuracy",
            "fidelity_macro_f1",
            "surrogate_task_accuracy",
            "surrogate_task_macro_f1",
        ]
        for col in expected_columns:
            assert col in result_df.columns

        # 2. Check saved artifacts (Joblib model, text rules, png plot)
        models_dir = mock_setup["models_dir"]
        surrogate_trees_dir = mock_setup["xai_dir"] / "surrogate_trees"

        assert (models_dir / f"{target}_surrogate_tree.joblib").exists()
        assert (surrogate_trees_dir / f"{target}_surrogate_tree_rules.txt").exists()
        assert (surrogate_trees_dir / f"{target}_surrogate_tree.png").exists()

        # Check content of rules text
        rules_text = (surrogate_trees_dir / f"{target}_surrogate_tree_rules.txt").read_text()
        assert "feature_1" in rules_text or "feature_2" in rules_text


class TestCalculateSurrogateDecisionTreeMetrics:
    def test_calculate_surrogate_decision_tree_metrics_batch(self, mock_setup):
        """Test the batch calculation wrapper across multiple targets."""
        target_columns = ("target_A", "target_B")

        result_df = calculate_surrogate_decision_tree_metrics(
            models_dir=mock_setup["models_dir"],
            random_state=mock_setup["random_state"],
            target_columns=target_columns,
            xai_dir=mock_setup["xai_dir"],
            best_by_target=mock_setup["best_by_target"],
            feature_columns=mock_setup["feature_columns"],
            feature_splits=mock_setup["feature_splits"],
        )

        # 1. Check returned concatenated DataFrame
        assert isinstance(result_df, pd.DataFrame)
        # 3 splits * 2 targets = 6 rows total
        assert len(result_df) == 6
        assert set(result_df["target"].unique()) == set(target_columns)

        # 2. Check if the summary CSV is correctly saved
        csv_path = mock_setup["xai_dir"] / "surrogate_trees" / "surrogate_tree_metrics.csv"
        assert csv_path.exists()

        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == 6
        assert list(saved_df.columns) == list(result_df.columns)
