import json
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

        # Mean of A = 10.66, Mean of B = 20.66, Overall Mean = 15.66
        # A significant portion of variance is explained by the group
        result = eta_squared_by_group(values, labels)

        assert isinstance(result, float)
        assert 0.0 < result <= 1.0
        assert result > 0.8  # Groups are well separated

    def test_eta_squared_zero_variation(self):
        values = pd.Series([10, 10, 10, 10])
        labels = pd.Series(["A", "A", "B", "B"])

        result = eta_squared_by_group(values, labels)

        assert result == 0.0


class TestCheckFeatureModelSignal:
    @patch("llm_behavior_xai.model_training_and_profiles.verify_training_features.mutual_info_classif")
    def test_check_feature_model_signal(self, mock_mic, xai_dir):
        # Mock mutual_info_classif to return a fixed array
        mock_mic.return_value = np.array([0.5, 0.1])

        random_state = 42
        feature_columns = ["feat_1", "feat_2"]
        feature_splits = {"train": pd.DataFrame({"feat_1": [1, 5, 1, 5], "feat_2": [2, 2, 3, 3]})}
        train_model_labels = pd.Series(["ModelA", "ModelB", "ModelA", "ModelB"])

        result_df = check_feature_model_signal(
            random_state, xai_dir, feature_columns, feature_splits, train_model_labels
        )

        # 1. Assert return DataFrame
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 2
        assert "feature" in result_df.columns
        assert "review_flag" in result_df.columns

        # feat_1 separates Models perfectly, expect a strong flag
        feat_1_row = result_df[result_df["feature"] == "feat_1"].iloc[0]
        assert feat_1_row["review_flag"] in ["strong_model_signal", "very_strong_model_signal"]

        # 2. Assert file was written
        csv_path = xai_dir / "features" / "feature_model_signal_check.csv"
        assert csv_path.exists()
        saved_df = pd.read_csv(csv_path)
        assert len(saved_df) == 2


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

        # 1. Assert outputs
        assert filtered_cols == ["feat_1", "feat_2"]
        assert "feat_bad" not in filtered_splits["train"].columns
        assert "feat_bad" not in filtered_splits["val"].columns

        # 2. Assert written files
        assert (xai_dir / "features" / "feature_list.csv").exists()

        metadata_path = xai_dir / "features" / "feature_metadata.json"
        assert metadata_path.exists()
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            assert metadata["feature_columns"] == ["feat_1", "feat_2"]
            assert metadata["excluded_zero_nan_features"] == ["feat_bad"]


class TestExcludeModelSpecificAndZeroNanPatterns:
    @patch("llm_behavior_xai.model_training_and_profiles.verify_training_features.build_feature_frame")
    def test_exclude_model_specific_and_zero_nan_patterns(self, mock_build_feature_frame, xai_dir):
        feature_columns = ["feat_clean", "feat_mostly_zero"]

        # Mock build_feature_frame to return predictable data
        # Let's make feat_mostly_zero mostly zero/NaN for 'ModelA' and not for others
        mock_build_feature_frame.side_effect = lambda src, *args, **kwargs: pd.DataFrame(
            {
                "feat_clean": [1, 2, 3, 4],
                # If ModelA, lots of zeros, else normal values
                "feat_mostly_zero": [0, 0, np.nan, 0] if src["model_key"].iloc[0] == "ModelA" else [10, 20, 30, 40],
            }
        )

        final_splits = {
            "train": pd.DataFrame({"text": ["dummy"] * 4, "model_key": ["ModelA"] * 4}),
            "val": pd.DataFrame({"text": ["dummy"] * 4, "model_key": ["ModelB"] * 4}),
        }

        # Dummy arguments for regexes/sets
        excluded_features, df_check = exclude_model_specific_and_zero_nan_patterns(
            first_person_pronouns=set(),
            heading_re=None,
            hedge_words=set(),
            list_marker_re=None,
            negation_words=set(),
            second_person_pronouns=set(),
            sentence_re=None,
            source_numeric_columns=("numeric1",),
            word_re=None,
            xai_dir=xai_dir,
            feature_columns=feature_columns,
            final_splits=final_splits,
        )

        # 1. Assert correct items flagged and excluded
        assert isinstance(excluded_features, list)
        assert "feat_mostly_zero" in excluded_features
        assert "feat_clean" not in excluded_features

        # 2. Assert DataFrame contents
        assert isinstance(df_check, pd.DataFrame)
        zero_row = df_check[(df_check["feature"] == "feat_mostly_zero") & (df_check["model_key"] == "ModelA")].iloc[0]

        assert zero_row["review_flag"] in ["model_mostly_zero_or_nan", "model_more_zero_or_nan"]

        # 3. Assert saved files
        assert (xai_dir / "features" / "feature_zero_nan_by_model_check.csv").exists()
        assert (xai_dir / "features" / "excluded_zero_nan_features.csv").exists()
