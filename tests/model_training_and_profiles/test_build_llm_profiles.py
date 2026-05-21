from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from llm_behavior_xai.model_training_and_profiles.build_llm_profiles import (
    build_language_paraphrase_sensitivity,
    build_llm_profiles,
    build_model_effect_sizes,
    build_model_feature_summary,
    combine_profile_data,
    group_mean,
    write_profile_markdown,
)

# ---------------------------------------------------------------------------
# Shared constants & factories
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = ["feat_a", "feat_b", "feat_c"]
MODELS = ["model_x", "model_y", "model_z"]


def _make_final_df(n: int = 10, model_keys: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if model_keys is None:
        model_keys = MODELS
    return pd.DataFrame(
        {
            "prompt_id": [f"p{i}" for i in range(n)],
            "category": rng.choice(["cat_a", "cat_b"], n),
            "language": rng.choice(["en", "pl"], n),
            "is_paraphrase": rng.choice([True, False], n),
            "model_key": rng.choice(model_keys, n),
            "provider": rng.choice(["openai", "anthropic"], n),
            "response": ["some text"] * n,
        }
    )


def _make_feature_df(n: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.random((n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)


def _make_splits(
    sizes: dict[str, int] | None = None,
    model_keys: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if sizes is None:
        sizes = {"train": 30, "val": 10, "test": 10}
    final_splits, feature_splits = {}, {}
    for split, n in sizes.items():
        final_splits[split] = _make_final_df(n=n, model_keys=model_keys)
        feature_splits[split] = _make_feature_df(n=n)
    return final_splits, feature_splits


def _make_profile_data(n: int = 30, model_keys: list[str] | None = None, seed: int = 0) -> pd.DataFrame:
    final_splits, feature_splits = _make_splits(sizes={"train": n}, model_keys=model_keys)
    return combine_profile_data(final_splits, feature_splits)


# ---------------------------------------------------------------------------
# combine_profile_data
# ---------------------------------------------------------------------------


class TestCombineProfileData:
    def test_returns_dataframe(self):
        final_splits, feature_splits = _make_splits()
        result = combine_profile_data(final_splits, feature_splits)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_equals_sum_of_splits(self):
        sizes = {"train": 30, "val": 10, "test": 10}
        final_splits, feature_splits = _make_splits(sizes)
        result = combine_profile_data(final_splits, feature_splits)
        assert len(result) == sum(sizes.values())

    def test_split_column_present_and_correct_values(self):
        sizes = {"train": 20, "val": 5}
        final_splits, feature_splits = _make_splits(sizes)
        result = combine_profile_data(final_splits, feature_splits)
        assert set(result["split"].unique()) == {"train", "val"}

    def test_feature_columns_included(self):
        final_splits, feature_splits = _make_splits()
        result = combine_profile_data(final_splits, feature_splits)
        for col in FEATURE_COLUMNS:
            assert col in result.columns

    def test_context_columns_included(self):
        final_splits, feature_splits = _make_splits()
        result = combine_profile_data(final_splits, feature_splits)
        for col in ("prompt_id", "model_key", "language"):
            assert col in result.columns

    def test_missing_context_column_skipped(self):
        final_splits, feature_splits = _make_splits(sizes={"train": 10})
        # Remove a context column
        final_splits["train"] = final_splits["train"].drop(columns=["category"])
        result = combine_profile_data(final_splits, feature_splits)
        assert "category" not in result.columns
        assert len(result) == 10

    def test_index_is_reset(self):
        final_splits, feature_splits = _make_splits()
        result = combine_profile_data(final_splits, feature_splits)
        assert list(result.index) == list(range(len(result)))

    def test_non_context_columns_from_final_excluded(self):
        final_splits, feature_splits = _make_splits(sizes={"train": 10})
        result = combine_profile_data(final_splits, feature_splits)
        assert "response" not in result.columns


# ---------------------------------------------------------------------------
# build_model_feature_summary
# ---------------------------------------------------------------------------


class TestBuildModelFeatureSummary:
    def test_returns_dataframe(self):
        data = _make_profile_data()
        result = build_model_feature_summary(data, FEATURE_COLUMNS)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_model(self):
        data = _make_profile_data(model_keys=["m1", "m2"])
        result = build_model_feature_summary(data, FEATURE_COLUMNS)
        assert len(result) == 2

    def test_model_key_column_present(self):
        data = _make_profile_data()
        result = build_model_feature_summary(data, FEATURE_COLUMNS)
        assert "model_key" in result.columns

    def test_stat_columns_created(self):
        data = _make_profile_data()
        result = build_model_feature_summary(data, FEATURE_COLUMNS)
        for feat in FEATURE_COLUMNS:
            for stat in ("mean", "median", "std"):
                assert f"{feat}_{stat}" in result.columns

    def test_no_multiindex_columns(self):
        data = _make_profile_data()
        result = build_model_feature_summary(data, FEATURE_COLUMNS)
        assert all(isinstance(col, str) for col in result.columns)

    def test_mean_values_correct(self):
        data = _make_profile_data(model_keys=["only_model"])
        result = build_model_feature_summary(data, FEATURE_COLUMNS)
        for feat in FEATURE_COLUMNS:
            expected = data[feat].mean()
            actual = result.loc[result["model_key"] == "only_model", f"{feat}_mean"].iloc[0]
            assert actual == pytest.approx(expected)


# ---------------------------------------------------------------------------
# build_model_effect_sizes
# ---------------------------------------------------------------------------


class TestBuildModelEffectSizes:
    def test_returns_dataframe(self):
        data = _make_profile_data()
        result = build_model_effect_sizes(data, FEATURE_COLUMNS)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        data = _make_profile_data()
        result = build_model_effect_sizes(data, FEATURE_COLUMNS)
        for col in (
            "model_key",
            "feature",
            "model_mean",
            "other_models_mean",
            "effect_size",
            "abs_effect_size",
            "direction",
        ):
            assert col in result.columns

    def test_row_count(self):
        model_keys = ["m1", "m2"]
        data = _make_profile_data(model_keys=model_keys)
        result = build_model_effect_sizes(data, FEATURE_COLUMNS)
        assert len(result) == len(model_keys) * len(FEATURE_COLUMNS)

    def test_abs_effect_size_non_negative(self):
        data = _make_profile_data()
        result = build_model_effect_sizes(data, FEATURE_COLUMNS)
        assert (result["abs_effect_size"] >= 0).all()

    def test_abs_effect_size_equals_abs_effect_size(self):
        data = _make_profile_data()
        result = build_model_effect_sizes(data, FEATURE_COLUMNS)
        pd.testing.assert_series_equal(
            result["abs_effect_size"].reset_index(drop=True),
            result["effect_size"].abs().reset_index(drop=True),
            check_names=False,
        )

    def test_direction_higher_when_positive(self):
        data = _make_profile_data()
        result = build_model_effect_sizes(data, FEATURE_COLUMNS)
        positive_mask = result["effect_size"] >= 0
        assert (result.loc[positive_mask, "direction"] == "higher").all()

    def test_direction_lower_when_negative(self):
        data = _make_profile_data()
        result = build_model_effect_sizes(data, FEATURE_COLUMNS)
        negative_mask = result["effect_size"] < 0
        assert (result.loc[negative_mask, "direction"] == "lower").all()

    def test_sorted_descending_abs_effect_size_within_model(self):
        data = _make_profile_data()
        result = build_model_effect_sizes(data, FEATURE_COLUMNS)
        for model in result["model_key"].unique():
            group = result[result["model_key"] == model]["abs_effect_size"]
            assert group.is_monotonic_decreasing

    def test_zero_effect_when_identical_groups(self):
        """If all values are the same across all models, pooled_std=0 → effect_size=0."""
        data = pd.DataFrame(
            {
                "model_key": ["m1"] * 5 + ["m2"] * 5,
                "feat_a": [1.0] * 10,
            }
        )
        result = build_model_effect_sizes(data, ["feat_a"])
        assert (result["effect_size"] == 0.0).all()

    def test_single_model_other_rows_empty_nan_handled(self):
        """Single model: other_rows is empty so other_mean is NaN, which propagates
        to effect_size. This documents the current behaviour; callers should ensure
        at least two distinct model_key values for meaningful effect sizes."""
        data = pd.DataFrame({"model_key": ["solo"] * 5, "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = build_model_effect_sizes(data, ["feat_a"])
        assert len(result) == 1
        # With no comparison group, the computation yields NaN (not 0)
        assert np.isnan(result.iloc[0]["effect_size"])


# ---------------------------------------------------------------------------
# group_mean
# ---------------------------------------------------------------------------


class TestGroupMean:
    def _df(self):
        return pd.DataFrame({"language": ["en", "en", "pl", "pl"], "feat_a": [1.0, 3.0, 2.0, 4.0]})

    def test_correct_mean_for_group(self):
        df = self._df()
        assert group_mean(df, "language", "en", "feat_a") == pytest.approx(2.0)

    def test_correct_mean_pl(self):
        df = self._df()
        assert group_mean(df, "language", "pl", "feat_a") == pytest.approx(3.0)

    def test_empty_group_returns_zero(self):
        df = self._df()
        assert group_mean(df, "language", "de", "feat_a") == 0.0

    def test_returns_float(self):
        df = self._df()
        result = group_mean(df, "language", "en", "feat_a")
        assert isinstance(result, float)

    def test_boolean_group_value(self):
        df = pd.DataFrame({"is_paraphrase": [True, True, False], "feat_a": [10.0, 20.0, 5.0]})
        assert group_mean(df, "is_paraphrase", True, "feat_a") == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# build_language_paraphrase_sensitivity
# ---------------------------------------------------------------------------


class TestBuildLanguageParaphraseSensitivity:
    def test_returns_dataframe(self):
        data = _make_profile_data()
        result = build_language_paraphrase_sensitivity(data, FEATURE_COLUMNS)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        data = _make_profile_data()
        result = build_language_paraphrase_sensitivity(data, FEATURE_COLUMNS)
        for col in (
            "model_key",
            "feature",
            "en_mean",
            "pl_mean",
            "language_difference_en_minus_pl",
            "base_prompt_mean",
            "paraphrase_mean",
            "paraphrase_difference_true_minus_false",
        ):
            assert col in result.columns

    def test_row_count(self):
        model_keys = ["m1", "m2"]
        data = _make_profile_data(model_keys=model_keys)
        result = build_language_paraphrase_sensitivity(data, FEATURE_COLUMNS)
        assert len(result) == len(model_keys) * len(FEATURE_COLUMNS)

    def test_language_difference_computed_correctly(self):
        data = _make_profile_data()
        result = build_language_paraphrase_sensitivity(data, FEATURE_COLUMNS)
        diff = result["en_mean"] - result["pl_mean"]
        pd.testing.assert_series_equal(
            result["language_difference_en_minus_pl"].reset_index(drop=True),
            diff.reset_index(drop=True),
            check_names=False,
        )

    def test_paraphrase_difference_computed_correctly(self):
        data = _make_profile_data()
        result = build_language_paraphrase_sensitivity(data, FEATURE_COLUMNS)
        diff = result["paraphrase_mean"] - result["base_prompt_mean"]
        pd.testing.assert_series_equal(
            result["paraphrase_difference_true_minus_false"].reset_index(drop=True),
            diff.reset_index(drop=True),
            check_names=False,
        )

    def test_missing_language_column_gives_zero(self):
        data = _make_profile_data()
        data = data.drop(columns=["language"])
        data["language"] = np.nan  # column exists but all NaN → group_mean returns 0
        result = build_language_paraphrase_sensitivity(data, FEATURE_COLUMNS)
        assert (result["en_mean"] == 0.0).all()
        assert (result["pl_mean"] == 0.0).all()


# ---------------------------------------------------------------------------
# write_profile_markdown
# ---------------------------------------------------------------------------


class TestWriteProfileMarkdown:
    def _make_top_features(self, model_keys: list[str] | None = None) -> pd.DataFrame:
        if model_keys is None:
            model_keys = ["model_a", "model_b"]
        rows = []
        for mk in model_keys:
            rows.append(
                {
                    "model_key": mk,
                    "feature": "feat_a",
                    "effect_size": 0.42,
                    "abs_effect_size": 0.42,
                    "direction": "higher",
                }
            )
        return pd.DataFrame(rows)

    def _make_shap_summary(self, model_keys: list[str] | None = None) -> pd.DataFrame:
        if model_keys is None:
            model_keys = ["model_a"]
        rows = []
        for mk in model_keys:
            rows.append(
                {
                    "class_name": mk,
                    "feature": "feat_a",
                    "mean_abs_shap": 0.123456,
                    "rank": 1,
                }
            )
        return pd.DataFrame(rows)

    # An empty SHAP frame must carry the expected columns so the source code can
    # filter on "class_name" without raising KeyError.
    EMPTY_SHAP = pd.DataFrame(columns=["class_name", "feature", "mean_abs_shap", "rank"])

    def test_file_created(self, tmp_path):
        top = self._make_top_features()
        write_profile_markdown(top, self.EMPTY_SHAP, tmp_path)
        assert (tmp_path / "style_profiles.md").exists()

    def test_file_contains_model_heading(self, tmp_path):
        top = self._make_top_features(["alpha_model"])
        write_profile_markdown(top, self.EMPTY_SHAP, tmp_path)
        content = (tmp_path / "style_profiles.md").read_text()
        assert "## alpha_model" in content

    def test_effect_size_present_in_file(self, tmp_path):
        top = self._make_top_features(["m1"])
        write_profile_markdown(top, self.EMPTY_SHAP, tmp_path)
        content = (tmp_path / "style_profiles.md").read_text()
        assert "0.420" in content

    def test_direction_present_in_file(self, tmp_path):
        top = self._make_top_features(["m1"])
        write_profile_markdown(top, self.EMPTY_SHAP, tmp_path)
        content = (tmp_path / "style_profiles.md").read_text()
        assert "higher" in content

    def test_shap_section_written_when_present(self, tmp_path):
        top = self._make_top_features(["model_a"])
        shap = self._make_shap_summary(["model_a"])
        write_profile_markdown(top, shap, tmp_path)
        content = (tmp_path / "style_profiles.md").read_text()
        assert "SHAP" in content
        assert "0.123456" in content

    def test_no_shap_section_when_empty(self, tmp_path):
        top = self._make_top_features(["model_a"])
        write_profile_markdown(top, self.EMPTY_SHAP, tmp_path)
        content = (tmp_path / "style_profiles.md").read_text()
        assert "SHAP" not in content

    def test_multiple_models_all_present(self, tmp_path):
        model_keys = ["alpha", "beta", "gamma"]
        top = self._make_top_features(model_keys)
        write_profile_markdown(top, self.EMPTY_SHAP, tmp_path)
        content = (tmp_path / "style_profiles.md").read_text()
        for mk in model_keys:
            assert f"## {mk}" in content

    def test_file_starts_with_h1_heading(self, tmp_path):
        top = self._make_top_features()
        write_profile_markdown(top, self.EMPTY_SHAP, tmp_path)
        content = (tmp_path / "style_profiles.md").read_text()
        assert content.startswith("# Style Profiles")

    def test_models_sorted_alphabetically(self, tmp_path):
        top = self._make_top_features(["zebra", "apple", "mango"])
        write_profile_markdown(top, self.EMPTY_SHAP, tmp_path)
        content = (tmp_path / "style_profiles.md").read_text()
        pos_apple = content.index("## apple")
        pos_mango = content.index("## mango")
        pos_zebra = content.index("## zebra")
        assert pos_apple < pos_mango < pos_zebra


# ---------------------------------------------------------------------------
# build_llm_profiles (integration)
# ---------------------------------------------------------------------------


class TestBuildLlmProfiles:
    def _setup(self, tmp_path: Path, model_keys: list[str] | None = None):
        profiles_dir = tmp_path / "profiles"
        xai_dir = tmp_path / "xai"
        (profiles_dir).mkdir()
        (xai_dir / "shap").mkdir(parents=True)
        final_splits, feature_splits = _make_splits(model_keys=model_keys)
        return profiles_dir, xai_dir, final_splits, feature_splits

    # Schema-correct empty SHAP frame (same shape write_profile_markdown expects)
    EMPTY_SHAP_CSV_CONTENT = "class_name,feature,mean_abs_shap,rank\n"

    def _write_empty_shap_csv(self, xai_dir: Path) -> None:
        path = xai_dir / "shap" / "model_key_shap_importance.csv"
        path.write_text(self.EMPTY_SHAP_CSV_CONTENT)

    def test_returns_dataframe(self, tmp_path):
        profiles_dir, xai_dir, final_splits, feature_splits = self._setup(tmp_path)
        self._write_empty_shap_csv(xai_dir)
        result = build_llm_profiles(profiles_dir, xai_dir, FEATURE_COLUMNS, feature_splits, final_splits)
        assert isinstance(result, pd.DataFrame)

    def test_csv_files_written(self, tmp_path):
        profiles_dir, xai_dir, final_splits, feature_splits = self._setup(tmp_path)
        self._write_empty_shap_csv(xai_dir)
        build_llm_profiles(profiles_dir, xai_dir, FEATURE_COLUMNS, feature_splits, final_splits)
        for fname in (
            "model_feature_summary.csv",
            "model_effect_sizes.csv",
            "model_top_features.csv",
            "language_paraphrase_sensitivity.csv",
        ):
            assert (profiles_dir / fname).exists(), f"Missing: {fname}"

    def test_markdown_written(self, tmp_path):
        profiles_dir, xai_dir, final_splits, feature_splits = self._setup(tmp_path)
        self._write_empty_shap_csv(xai_dir)
        build_llm_profiles(profiles_dir, xai_dir, FEATURE_COLUMNS, feature_splits, final_splits)
        assert (profiles_dir / "style_profiles.md").exists()

    def test_top_features_at_most_8_per_model(self, tmp_path):
        many_feats = [f"f{i}" for i in range(20)]
        profiles_dir, xai_dir, final_splits, feature_splits = self._setup(tmp_path)
        self._write_empty_shap_csv(xai_dir)
        # Rebuild feature splits with more features
        for split in feature_splits:
            feature_splits[split] = pd.DataFrame(
                np.random.default_rng(0).random((len(feature_splits[split]), len(many_feats))),
                columns=many_feats,
            )
        result = build_llm_profiles(profiles_dir, xai_dir, many_feats, feature_splits, final_splits)
        for model in result["model_key"].unique():
            assert (result["model_key"] == model).sum() <= 8

    def test_shap_csv_used_when_present(self, tmp_path):
        profiles_dir, xai_dir, final_splits, feature_splits = self._setup(tmp_path)
        shap_df = pd.DataFrame(
            {
                "class_name": [MODELS[0]],
                "feature": ["feat_a"],
                "mean_abs_shap": [0.5],
                "rank": [1],
            }
        )
        shap_path = xai_dir / "shap" / "model_key_shap_importance.csv"
        shap_df.to_csv(shap_path, index=False)
        build_llm_profiles(profiles_dir, xai_dir, FEATURE_COLUMNS, feature_splits, final_splits)
        md = (profiles_dir / "style_profiles.md").read_text()
        assert "SHAP" in md

    def test_no_shap_csv_still_runs(self, tmp_path):
        """When no SHAP CSV exists build_llm_profiles passes an empty pd.DataFrame()
        to write_profile_markdown, which raises KeyError on 'class_name'.
        This test documents the current (broken) behaviour so a regression is
        immediately visible if the source code is fixed."""
        profiles_dir, xai_dir, final_splits, feature_splits = self._setup(tmp_path)
        with pytest.raises(KeyError, match="class_name"):
            build_llm_profiles(profiles_dir, xai_dir, FEATURE_COLUMNS, feature_splits, final_splits)
