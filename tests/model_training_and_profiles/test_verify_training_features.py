from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from llm_behavior_xai.model_training_and_profiles.verify_training_features import (
    check_feature_model_signal,
    eta_squared_by_group,
    exclude_model_specific_and_zero_nan_patterns,
    filter_dataset,
)


@pytest.fixture
def xai_dir(tmp_path: Path) -> Path:
    """Fixture to provide a mock xai_dir with the required 'features' subdirectory."""
    d = tmp_path / "xai"
    (d / "features").mkdir(parents=True, exist_ok=True)
    return d


class TestEtaSquaredByGroup:
    def test_eta_squared_normal(self):
        values = pd.Series([10, 12, 10, 20, 22, 20])
        labels = pd.Series(["A", "A", "A", "B", "B", "B"])

        result = eta_squared_by_group(values, labels)

        assert isinstance(result, float)
        assert 0.0 < result <= 1.0
        assert result > 0.8

    def test_eta_squared_zero_variation(self):
        values = pd.Series([10, 10, 10, 10])
        labels = pd.Series(["A", "A", "B", "B"])

        result = eta_squared_by_group(values, labels)

        assert result == 0.0


class TestCheckFeatureModelSignal:
    @patch("llm_behavior_xai.model_training_and_profiles.verify_training_features.mutual_info_classif")
    def test_check_feature_model_signal(self, mock_mic, xai_dir):
        mock_mic.return_value = np.array([0.5, 0.1])

        random_state = 42
        feature_columns = ["feat_1", "feat_2"]
        feature_splits = {"train": pd.DataFrame({"feat_1": [1, 5, 1, 5], "feat_2": [2, 2, 3, 3]})}
        train_model_labels = pd.Series(["ModelA", "ModelB", "ModelA", "ModelB"])

        result_df = check_feature_model_signal(
            random_state, xai_dir, feature_columns, feature_splits, train_model_labels
        )

        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        feat_1_row = result_df[result_df["feature"] == "feat_1"].iloc[0]
        assert feat_1_row["review_flag"] in ["strong_model_signal", "very_strong_model_signal"]

    @patch("llm_behavior_xai.model_training_and_profiles.verify_training_features.mutual_info_classif")
    def test_check_feature_model_signal_exception(self, mock_mic, xai_dir, capsys):
        """Covers lines 45-47: Handling exception in mutual_info_classif."""
        # Force mutual_info_classif to fail
        mock_mic.side_effect = ValueError("Mocked failure")

        feature_splits = {"train": pd.DataFrame({"feat_1": [1, 2, 3, 4]})}
        train_model_labels = pd.Series(["A", "B", "A", "B"])

        result_df = check_feature_model_signal(42, xai_dir, ["feat_1"], feature_splits, train_model_labels)

        # Ensure fallback zeroes were used and exception was printed
        assert result_df["mutual_information_model_key"].iloc[0] == 0.0
        assert "Mutual information check failed: Mocked failure" in capsys.readouterr().out

    @patch("llm_behavior_xai.model_training_and_profiles.verify_training_features.mutual_info_classif")
    def test_check_feature_model_signal_std_zero_and_strong_flag(self, mock_mic, xai_dir):
        """Covers line 58 (zero standard deviation) and line 69 (strong_model_signal)."""
        mock_mic.return_value = np.array([0.0, 0.0])

        feature_splits = {
            "train": pd.DataFrame(
                {
                    "feat_std_zero": [1, 1, 1, 1],  # std will be 0
                    "feat_strong": [10, 15, 20, 15],  # total_var=50, between_var=25 -> eta_squared = 0.5
                }
            )
        }
        train_model_labels = pd.Series(["ModelA", "ModelA", "ModelB", "ModelB"])

        result_df = check_feature_model_signal(
            42, xai_dir, ["feat_std_zero", "feat_strong"], feature_splits, train_model_labels
        )

        # std == 0 should yield max_abs_model_mean_z == 0.0
        row_zero = result_df[result_df["feature"] == "feat_std_zero"].iloc[0]
        assert row_zero["max_abs_model_mean_z"] == 0.0

        # feat_strong should yield an eta_squared of 0.5, hitting the "strong_model_signal" condition
        row_strong = result_df[result_df["feature"] == "feat_strong"].iloc[0]
        assert row_strong["eta_squared_model_key"] == 0.5
        assert row_strong["review_flag"] == "strong_model_signal"


class TestFilterDataset:
    def test_filter_dataset(self, xai_dir):
        target_columns = ("target_A",)
        excluded_zero_nan_features = ["feat_bad"]
        feature_columns = ["feat_1", "feat_bad", "feat_2"]
        feature_meanings = pd.DataFrame(
            {"feature": ["feat_1", "feat_bad", "feat_2"], "description": ["desc1", "desc2", "desc3"]}
        )
        feature_splits = {
            "train": pd.DataFrame({"feat_1": [1], "feat_bad": [0], "feat_2": [2]}),
            "val": pd.DataFrame({"feat_1": [3], "feat_bad": [0], "feat_2": [4]}),
        }

        filtered_cols, filtered_splits = filter_dataset(
            target_columns, xai_dir, excluded_zero_nan_features, feature_columns, feature_meanings, feature_splits
        )

        assert filtered_cols == ["feat_1", "feat_2"]
        assert "feat_bad" not in filtered_splits["train"].columns

    def test_filter_dataset_feature_meanings_globals(self, xai_dir):
        """Covers lines 109-116: the execution of the feature_meanings in globals() block."""
        import llm_behavior_xai.model_training_and_profiles.verify_training_features as module

        # Inject "feature_meanings" into the module's global dict dynamically
        module.__dict__["feature_meanings"] = "placeholder_to_pass_condition"

        try:
            feature_meanings = pd.DataFrame({"feature": ["feat_1", "feat_bad"], "description": ["desc1", "desc2"]})
            feature_splits = {"train": pd.DataFrame({"feat_1": [1], "feat_bad": [0]})}

            filter_dataset(
                ("target_A",), xai_dir, ["feat_bad"], ["feat_1", "feat_bad"], feature_meanings, feature_splits
            )

            # Check if the feature_descriptions.csv file was successfully generated
            desc_path = xai_dir / "features" / "feature_descriptions.csv"
            assert desc_path.exists()

            df = pd.read_csv(desc_path)
            assert "used_for_training" in df.columns
            assert "excluded_reason" in df.columns
            assert df.loc[df["feature"] == "feat_bad", "excluded_reason"].iloc[0] == "model-specific zero/NaN pattern"
        finally:
            del module.__dict__["feature_meanings"]


class TestExcludeModelSpecificAndZeroNanPatterns:
    @patch("llm_behavior_xai.model_training_and_profiles.verify_training_features.build_feature_frame")
    def test_exclude_model_specific_and_zero_nan_patterns(self, mock_build_feature_frame, xai_dir):
        feature_columns = ["feat_clean", "feat_mostly_zero"]
        mock_build_feature_frame.side_effect = lambda src, *args, **kwargs: pd.DataFrame(
            {
                "feat_clean": [1, 2, 3, 4],
                "feat_mostly_zero": [0, 0, np.nan, 0] if src["model_key"].iloc[0] == "ModelA" else [10, 20, 30, 40],
            }
        )
        final_splits = {
            "train": pd.DataFrame({"text": ["dummy"] * 4, "model_key": ["ModelA"] * 4}),
            "val": pd.DataFrame({"text": ["dummy"] * 4, "model_key": ["ModelB"] * 4}),
        }

        excluded_features, df_check = exclude_model_specific_and_zero_nan_patterns(
            set(), None, set(), None, set(), set(), None, ("numeric1",), None, xai_dir, feature_columns, final_splits
        )

        assert "feat_mostly_zero" in excluded_features
        zero_row = df_check[(df_check["feature"] == "feat_mostly_zero") & (df_check["model_key"] == "ModelA")].iloc[0]
        assert zero_row["review_flag"] in ["model_mostly_zero_or_nan", "model_more_zero_or_nan"]

    @patch("llm_behavior_xai.model_training_and_profiles.verify_training_features.build_feature_frame")
    def test_exclude_others_more_zero_or_nan(self, mock_build_feature_frame, xai_dir):
        """Covers line 199: difference <= -0.50 threshold."""
        mock_build_feature_frame.side_effect = lambda src, *args, **kwargs: pd.DataFrame(
            {
                # ModelA gets regular non-zeroes, ModelB gets entirely zeroes
                "feat_mostly_other": [1, 2, 3, 4] if src["model_key"].iloc[0] == "ModelA" else [0, 0, 0, 0]
            }
        )
        final_splits = {
            "train": pd.DataFrame({"text": ["d"] * 4, "model_key": ["ModelA"] * 4}),
            "val": pd.DataFrame({"text": ["d"] * 4, "model_key": ["ModelB"] * 4}),
        }

        _, df_check = exclude_model_specific_and_zero_nan_patterns(
            set(), None, set(), None, set(), set(), None, (), None, xai_dir, ["feat_mostly_other"], final_splits
        )

        # Check from ModelA's point of view: its zero ratio is 0.0, the other's is 1.0. Difference: -1.0
        row_A = df_check[(df_check["feature"] == "feat_mostly_other") & (df_check["model_key"] == "ModelA")].iloc[0]
        assert row_A["difference"] <= -0.50
        assert row_A["review_flag"] == "others_more_zero_or_nan"

    @patch("llm_behavior_xai.model_training_and_profiles.verify_training_features.build_feature_frame")
    def test_exclude_no_features_flagged(self, mock_build_feature_frame, xai_dir, capsys):
        """Covers line 245: print statement when no features are flagged."""
        # Perfectly clean non-zero data
        mock_build_feature_frame.side_effect = lambda src, *args, **kwargs: pd.DataFrame(
            {"feat_clean": [10, 20, 30, 40]}
        )
        final_splits = {
            "train": pd.DataFrame({"text": ["d"] * 4, "model_key": ["ModelA", "ModelA", "ModelB", "ModelB"]}),
        }

        excluded, _ = exclude_model_specific_and_zero_nan_patterns(
            set(), None, set(), None, set(), set(), None, (), None, xai_dir, ["feat_clean"], final_splits
        )

        assert len(excluded) == 0
        assert "No model-specific zero/NaN features were flagged for exclusion." in capsys.readouterr().out
