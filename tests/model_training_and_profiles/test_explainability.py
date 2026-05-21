from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys
from unittest.mock import patch

from llm_behavior_xai.model_training_and_profiles.explainability import (
    built_in_importance,
    estimator_from_model,
    fallback_shap_frame,
    feature_group,
    normalize_group_importance,
    plot_top_importance,
    shap_values_to_importance,
    permutation_importance_frame,
    explain_with_shap,
    calculate_outputs_importances,
    calculate_feature_group_importance,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "text_word_count",
    "text_char_count",
    "text_sentence_count",
    "text_entropy",
    "source_perplexity",
]
N_FEATURES = len(FEATURE_COLUMNS)
N_SAMPLES = 40
TARGET = "quality"


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=3,
        n_redundant=1,
        random_state=0,
    )
    return X, y


@pytest.fixture
def trained_rf(classification_data):
    X, y = classification_data
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    clf.fit(X, y)
    return clf


@pytest.fixture
def trained_lr(classification_data):
    X, y = classification_data
    clf = LogisticRegression(max_iter=200, random_state=0)
    clf.fit(X, y)
    return clf


@pytest.fixture
def trained_pipeline(classification_data):
    X, y = classification_data
    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=5, random_state=0))])
    pipe.fit(X, y)
    return pipe


@pytest.fixture
def feature_splits(classification_data):
    X, y = classification_data
    df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    mid = N_SAMPLES // 2
    return {
        "train": df.iloc[:mid].reset_index(drop=True),
        "val": df.iloc[mid:].reset_index(drop=True),
        "y_train": y[:mid],
        "y_val": y[mid:],
    }


@pytest.fixture
def best_by_target(trained_rf, feature_splits):
    """Minimal best_by_target bundle used by permutation_importance_frame / explain_with_shap."""
    return {
        TARGET: {
            "model": trained_rf,
            "class_names": ["class_0", "class_1"],
            "encoded_targets": {
                "val": feature_splits["y_val"],
                "train": feature_splits["y_train"],
            },
        }
    }


# ---------------------------------------------------------------------------
# estimator_from_model
# ---------------------------------------------------------------------------


class TestEstimatorFromModel:
    def test_plain_estimator_returned_as_is(self, trained_rf):
        assert estimator_from_model(trained_rf) is trained_rf

    def test_pipeline_returns_model_step(self, trained_pipeline):
        result = estimator_from_model(trained_pipeline)
        assert isinstance(result, RandomForestClassifier)

    def test_non_pipeline_without_named_steps(self, trained_lr):
        result = estimator_from_model(trained_lr)
        assert result is trained_lr


# ---------------------------------------------------------------------------
# built_in_importance
# ---------------------------------------------------------------------------


class TestBuiltInImportance:
    def test_rf_returns_feature_importances(self, trained_rf):
        df = built_in_importance(TARGET, trained_rf, FEATURE_COLUMNS)
        assert df["importance_kind"].iloc[0] == "feature_importances"

    def test_lr_returns_absolute_coefficients(self, trained_lr):
        df = built_in_importance(TARGET, trained_lr, FEATURE_COLUMNS)
        assert df["importance_kind"].iloc[0] == "absolute_coefficients"

    def test_no_attr_returns_zeros(self):
        mock_model = MagicMock(spec=[])  # no feature_importances_ or coef_
        df = built_in_importance(TARGET, mock_model, FEATURE_COLUMNS)
        assert df["importance_kind"].iloc[0] == "not_available"
        assert (df["importance"] == 0.0).all()

    def test_returns_dataframe(self, trained_rf):
        df = built_in_importance(TARGET, trained_rf, FEATURE_COLUMNS)
        assert isinstance(df, pd.DataFrame)

    def test_row_count_matches_features(self, trained_rf):
        df = built_in_importance(TARGET, trained_rf, FEATURE_COLUMNS)
        assert len(df) == N_FEATURES

    def test_required_columns_present(self, trained_rf):
        df = built_in_importance(TARGET, trained_rf, FEATURE_COLUMNS)
        for col in ("target", "feature", "importance", "importance_kind"):
            assert col in df.columns

    def test_target_column_correct(self, trained_rf):
        df = built_in_importance(TARGET, trained_rf, FEATURE_COLUMNS)
        assert (df["target"] == TARGET).all()

    def test_sorted_descending(self, trained_rf):
        df = built_in_importance(TARGET, trained_rf, FEATURE_COLUMNS)
        assert df["importance"].is_monotonic_decreasing

    def test_pipeline_model(self, trained_pipeline):
        df = built_in_importance(TARGET, trained_pipeline, FEATURE_COLUMNS)
        assert df["importance_kind"].iloc[0] == "feature_importances"

    def test_feature_values_non_negative_for_rf(self, trained_rf):
        df = built_in_importance(TARGET, trained_rf, FEATURE_COLUMNS)
        assert (df["importance"] >= 0).all()

    def test_coef_multiclass_averages_rows(self, classification_data):
        """LogisticRegression with 3 classes has coef_ shape (3, n_features); should still produce one row per feature."""
        X, _ = classification_data
        y_multi = np.array([i % 3 for i in range(N_SAMPLES)])
        clf = LogisticRegression(max_iter=500, random_state=0)
        clf.fit(X, y_multi)
        df = built_in_importance(TARGET, clf, FEATURE_COLUMNS)
        assert len(df) == N_FEATURES


# ---------------------------------------------------------------------------
# shap_values_to_importance
# ---------------------------------------------------------------------------


class TestShapValuesToImportance:
    def _make_shap(self, n_samples=20, n_features=5, n_classes=2):
        return np.random.default_rng(0).random((n_samples, n_features, n_classes))

    def test_returns_dataframe(self):
        values = self._make_shap()
        df = shap_values_to_importance(TARGET, values, ["c0", "c1"], FEATURE_COLUMNS)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        values = self._make_shap()
        df = shap_values_to_importance(TARGET, values, ["c0", "c1"], FEATURE_COLUMNS)
        for col in ("target", "class_name", "feature", "mean_abs_shap", "rank", "importance_kind"):
            assert col in df.columns

    def test_overall_row_present(self):
        values = self._make_shap()
        df = shap_values_to_importance(TARGET, values, ["c0", "c1"], FEATURE_COLUMNS)
        assert "__overall__" in df["class_name"].values

    def test_class_rows_present(self):
        values = self._make_shap()
        df = shap_values_to_importance(TARGET, values, ["c0", "c1"], FEATURE_COLUMNS)
        assert "c0" in df["class_name"].values
        assert "c1" in df["class_name"].values

    def test_importance_kind_is_shap(self):
        values = self._make_shap()
        df = shap_values_to_importance(TARGET, values, ["c0", "c1"], FEATURE_COLUMNS)
        assert (df["importance_kind"] == "shap").all()

    def test_mean_abs_shap_non_negative(self):
        values = self._make_shap()
        df = shap_values_to_importance(TARGET, values, ["c0", "c1"], FEATURE_COLUMNS)
        assert (df["mean_abs_shap"] >= 0).all()

    def test_2d_input_promoted_to_3d(self):
        # 2D values should be handled without error
        values_2d = np.random.default_rng(0).random((20, N_FEATURES))
        df = shap_values_to_importance(TARGET, values_2d, ["c0"], FEATURE_COLUMNS)
        assert isinstance(df, pd.DataFrame)
        assert "__overall__" in df["class_name"].values

    def test_rank_starts_at_one(self):
        values = self._make_shap()
        df = shap_values_to_importance(TARGET, values, ["c0", "c1"], FEATURE_COLUMNS)
        for class_name in df["class_name"].unique():
            group = df[df["class_name"] == class_name]
            assert group["rank"].min() == 1

    def test_each_feature_appears_per_class(self):
        values = self._make_shap()
        df = shap_values_to_importance(TARGET, values, ["c0"], FEATURE_COLUMNS)
        class_df = df[df["class_name"] == "c0"]
        assert set(class_df["feature"]) == set(FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# fallback_shap_frame
# ---------------------------------------------------------------------------


class TestFallbackShapFrame:
    def test_returns_dataframe(self, trained_rf):
        df = fallback_shap_frame(TARGET, trained_rf, "some error", FEATURE_COLUMNS)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, trained_rf):
        df = fallback_shap_frame(TARGET, trained_rf, "err", FEATURE_COLUMNS)
        for col in ("target", "class_name", "feature", "mean_abs_shap", "rank", "importance_kind", "note"):
            assert col in df.columns

    def test_class_name_is_overall(self, trained_rf):
        df = fallback_shap_frame(TARGET, trained_rf, "err", FEATURE_COLUMNS)
        assert (df["class_name"] == "__overall__").all()

    def test_importance_kind_is_fallback(self, trained_rf):
        df = fallback_shap_frame(TARGET, trained_rf, "err", FEATURE_COLUMNS)
        assert (df["importance_kind"] == "fallback_importance").all()

    def test_note_contains_reason(self, trained_rf):
        df = fallback_shap_frame(TARGET, trained_rf, "my reason", FEATURE_COLUMNS)
        assert (df["note"] == "my reason").all()

    def test_rank_sequential(self, trained_rf):
        df = fallback_shap_frame(TARGET, trained_rf, "err", FEATURE_COLUMNS)
        assert list(df["rank"]) == list(range(1, len(df) + 1))

    def test_row_count(self, trained_rf):
        df = fallback_shap_frame(TARGET, trained_rf, "err", FEATURE_COLUMNS)
        assert len(df) == N_FEATURES


# ---------------------------------------------------------------------------
# plot_top_importance
# ---------------------------------------------------------------------------


class TestPlotTopImportance:
    def _make_df(self):
        rng = np.random.default_rng(1)
        return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": rng.random(N_FEATURES)})

    def test_file_created(self, tmp_path):
        df = self._make_df()
        out = tmp_path / "test_plot.png"
        plot_top_importance(df, "importance", "Test Title", out)
        assert out.exists()

    def test_top_n_respected(self, tmp_path):
        """Ensure function doesn't crash when top_n exceeds available rows."""
        df = self._make_df()
        out = tmp_path / "top2.png"
        plot_top_importance(df, "importance", "Title", out, top_n=2)
        assert out.exists()

    def test_all_rows_when_top_n_larger(self, tmp_path):
        df = self._make_df()
        out = tmp_path / "all.png"
        plot_top_importance(df, "importance", "Title", out, top_n=100)
        assert out.exists()


# ---------------------------------------------------------------------------
# feature_group
# ---------------------------------------------------------------------------


class TestFeatureGroup:
    @pytest.mark.parametrize("feat", ["source_avg_logprob", "source_sum_logprob", "source_perplexity"])
    def test_generation_confidence(self, feat):
        assert feature_group(feat) == "generation_confidence"

    @pytest.mark.parametrize(
        "feat",
        [
            "source_response_length",
            "text_char_count",
            "text_word_count",
            "text_sentence_count",
            "text_paragraph_count",
            "text_avg_sentence_words",
        ],
    )
    def test_length_and_structure(self, feat):
        assert feature_group(feat) == "length_and_structure"

    @pytest.mark.parametrize(
        "feat",
        [
            "text_unique_word_count",
            "text_type_token_ratio",
            "text_hapax_ratio",
            "text_entropy",
            "text_repetition_rate",
            "text_repeated_bigram_ratio",
        ],
    )
    def test_lexical_diversity(self, feat):
        assert feature_group(feat) == "lexical_diversity_and_repetition"

    @pytest.mark.parametrize(
        "feat",
        [
            "text_punctuation_count",
            "text_comma_density",
            "text_colon_density",
            "text_list_marker_count",
            "text_markdown_bold_count",
            "text_code_fence_count",
        ],
    )
    def test_formatting_and_punctuation(self, feat):
        assert feature_group(feat) == "formatting_and_punctuation"

    @pytest.mark.parametrize(
        "feat",
        [
            "text_first_person_pronoun_count",
            "text_hedge_word_count",
            "text_negation_word_count",
            "text_second_person_pronoun_density",
        ],
    )
    def test_stance_pronouns(self, feat):
        assert feature_group(feat) == "stance_pronouns_and_uncertainty"

    def test_unknown_feature_is_other(self):
        assert feature_group("totally_unknown_feature_xyz") == "other"

    def test_returns_string(self):
        assert isinstance(feature_group("text_entropy"), str)


# ---------------------------------------------------------------------------
# normalize_group_importance
# ---------------------------------------------------------------------------


class TestNormalizeGroupImportance:
    def _make_group_df(self):
        return pd.DataFrame(
            {
                "target": ["t1", "t1", "t2", "t2"],
                "method": ["m", "m", "m", "m"],
                "feature_group": ["g1", "g2", "g1", "g2"],
                "importance": [3.0, 1.0, 0.0, 5.0],
            }
        )

    def test_shares_sum_to_one_per_group(self):
        df = normalize_group_importance(self._make_group_df(), "importance")
        totals = df.groupby(["target", "method"])["importance_share"].sum()
        for val in totals:
            assert val == pytest.approx(1.0)

    def test_importance_share_column_added(self):
        df = normalize_group_importance(self._make_group_df(), "importance")
        assert "importance_share" in df.columns

    def test_negative_values_clipped_to_zero(self):
        df = pd.DataFrame(
            {
                "target": ["t1", "t1"],
                "method": ["m", "m"],
                "feature_group": ["g1", "g2"],
                "importance": [-2.0, 4.0],
            }
        )
        result = normalize_group_importance(df, "importance")
        # g1 clipped to 0 → share = 0; g2 = 1.0
        g1_share = result.loc[result["feature_group"] == "g1", "importance_share"].iloc[0]
        assert g1_share == pytest.approx(0.0)

    def test_all_zero_importance_gives_zero_share(self):
        df = pd.DataFrame(
            {
                "target": ["t1", "t1"],
                "method": ["m", "m"],
                "feature_group": ["g1", "g2"],
                "importance": [0.0, 0.0],
            }
        )
        result = normalize_group_importance(df, "importance")
        assert (result["importance_share"] == 0.0).all()

    def test_does_not_mutate_input(self):
        df = self._make_group_df()
        original_values = df["importance"].copy()
        normalize_group_importance(df, "importance")
        pd.testing.assert_series_equal(df["importance"], original_values)

    def test_returns_dataframe(self):
        df = self._make_group_df()
        result = normalize_group_importance(df, "importance")
        assert isinstance(result, pd.DataFrame)

    def test_shares_are_between_zero_and_one(self):
        df = self._make_group_df()
        result = normalize_group_importance(df, "importance")
        assert (result["importance_share"] >= 0).all()
        assert (result["importance_share"] <= 1.0 + 1e-9).all()


class TestPermutationImportanceFrame:
    def test_returns_correct_dataframe(self, trained_rf, best_by_target, feature_splits):
        df = permutation_importance_frame(TARGET, trained_rf, best_by_target, feature_splits, FEATURE_COLUMNS, 42)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["target", "feature", "importance_mean", "importance_std"]
        assert len(df) == N_FEATURES
        assert (df["target"] == TARGET).all()
        # Ensure it's sorted by importance_mean descending
        assert df["importance_mean"].is_monotonic_decreasing


# ---------------------------------------------------------------------------
# explain_with_shap (Lines 126-138)
# ---------------------------------------------------------------------------


class TestExplainWithShap:
    @patch.dict("sys.modules", {"shap": MagicMock()})
    def test_success_path_returns_shap_importance(self, trained_rf, best_by_target, feature_splits):
        # Mock SHAP behavior to avoid requiring the heavy library during tests
        mock_shap = sys.modules["shap"]
        mock_explainer = MagicMock()
        mock_shap.Explainer.return_value = mock_explainer

        # Create dummy SHAP values array: shape (n_samples, n_features, n_classes)
        mock_shap_values = MagicMock()
        val_samples = len(feature_splits["val"])
        n_classes = len(best_by_target[TARGET]["class_names"])
        mock_shap_values.values = np.random.rand(val_samples, N_FEATURES, n_classes)
        mock_explainer.return_value = mock_shap_values

        df = explain_with_shap(TARGET, trained_rf, best_by_target, feature_splits, FEATURE_COLUMNS, 42)

        assert isinstance(df, pd.DataFrame)
        assert "importance_kind" in df.columns
        assert (df["importance_kind"] == "shap").all()

    def test_exception_triggers_fallback_frame(self, trained_rf, best_by_target, feature_splits):
        # By patching the 'shap' module as None, an ImportError is raised, triggering fallback
        with patch.dict("sys.modules", {"shap": None}):
            df = explain_with_shap(TARGET, trained_rf, best_by_target, feature_splits, FEATURE_COLUMNS, 42)

            assert isinstance(df, pd.DataFrame)
            assert (df["importance_kind"] == "fallback_importance").all()
            assert "No module named" in df["note"].iloc[0] or "None" in df["note"].iloc[0]


# ---------------------------------------------------------------------------
# calculate_outputs_importances (Lines 160-199)
# ---------------------------------------------------------------------------


class TestCalculateOutputsImportances:
    def test_generates_all_outputs_and_files(self, tmp_path, trained_rf, best_by_target, feature_splits):
        xai_dir = tmp_path
        (xai_dir / "importance").mkdir()
        (xai_dir / "shap").mkdir()

        # To speed up the test and isolate logic, force the fallback SHAP path
        with patch.dict("sys.modules", {"shap": None}):
            imp_out, perm_out, shap_out = calculate_outputs_importances(
                42, xai_dir, best_by_target, FEATURE_COLUMNS, feature_splits
            )

        # 1. Assert memory outputs
        assert len(imp_out) == 1
        assert len(perm_out) == 1
        assert len(shap_out) == 1

        assert isinstance(imp_out[0], pd.DataFrame)
        assert isinstance(perm_out[0], pd.DataFrame)
        assert isinstance(shap_out[0], pd.DataFrame)

        # 2. Assert file outputs (.csv and .png)
        importance_dir = xai_dir / "importance"
        shap_dir = xai_dir / "shap"

        assert (importance_dir / f"{TARGET}_built_in_importance.csv").exists()
        assert (importance_dir / f"{TARGET}_built_in_importance.png").exists()

        assert (importance_dir / f"{TARGET}_permutation_importance.csv").exists()
        assert (importance_dir / f"{TARGET}_permutation_importance.png").exists()

        assert (shap_dir / f"{TARGET}_shap_importance.csv").exists()
        assert (shap_dir / f"{TARGET}_shap_importance.png").exists()


# ---------------------------------------------------------------------------
# calculate_feature_group_importance (Lines 210-252)
# ---------------------------------------------------------------------------


class TestCalculateFeatureGroupImportance:
    def test_calculates_and_plots_feature_groups(self, tmp_path):
        xai_dir = tmp_path
        (xai_dir / "importance").mkdir()

        # Dummy inputs derived from the expected outputs of previous steps
        # "text_char_count" groups to "length_and_structure"
        imp_out = [
            pd.DataFrame(
                {"target": [TARGET], "method": ["built_in"], "feature": ["text_char_count"], "importance": [0.5]}
            )
        ]
        perm_out = [
            pd.DataFrame(
                {
                    "target": [TARGET],
                    "method": ["permutation"],
                    "feature": ["text_char_count"],
                    "importance_mean": [0.3],
                }
            )
        ]
        shap_out = [
            pd.DataFrame(
                {
                    "target": [TARGET],
                    "method": ["shap"],
                    "class_name": ["__overall__"],
                    "feature": ["text_char_count"],
                    "mean_abs_shap": [0.4],
                }
            )
        ]

        df = calculate_feature_group_importance((TARGET,), xai_dir, imp_out, perm_out, shap_out)

        # 1. Assert structure of returned DataFrame
        assert isinstance(df, pd.DataFrame)
        expected_cols = ["target", "method", "feature_group", "importance", "importance_share"]
        for col in expected_cols:
            assert col in df.columns

        assert "length_and_structure" in df["feature_group"].values

        # 2. Assert files were written
        assert (xai_dir / "importance" / "feature_group_importance.csv").exists()
        assert (xai_dir / "importance" / f"{TARGET}_feature_group_importance.png").exists()
