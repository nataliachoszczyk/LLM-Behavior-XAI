from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from llm_behavior_xai.model_training_and_profiles.split_datasets import create_split_overview, load_final_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    n: int = 10,
    model_keys: list[str] | None = None,
    languages: list[str] | None = None,
    paraphrase_values: list[bool] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(seed)
    model_keys = model_keys or ["model_a", "model_b"]
    languages = languages or ["en", "pl"]
    paraphrase_values = paraphrase_values if paraphrase_values is not None else [True, False]
    return pd.DataFrame(
        {
            "model_key": rng.choice(model_keys, n),
            "language": rng.choice(languages, n),
            "is_paraphrase": rng.choice(paraphrase_values, n),
        }
    )


def _write_csv(tmp_path, name: str, df: pd.DataFrame) -> Path:
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# load_final_splits
# ---------------------------------------------------------------------------


class TestLoadFinalSplits:
    def test_returns_dict(self, tmp_path):
        df = _make_df()
        paths = {"train": _write_csv(tmp_path, "train.csv", df)}
        result = load_final_splits(paths)
        assert isinstance(result, dict)

    def test_keys_match_input(self, tmp_path):
        dfs = {"train": _make_df(20), "val": _make_df(5), "test": _make_df(5)}
        paths = {split: _write_csv(tmp_path, f"{split}.csv", df) for split, df in dfs.items()}
        result = load_final_splits(paths)
        assert set(result.keys()) == {"train", "val", "test"}

    def test_values_are_dataframes(self, tmp_path):
        df = _make_df()
        paths = {"train": _write_csv(tmp_path, "train.csv", df)}
        result = load_final_splits(paths)
        assert isinstance(result["train"], pd.DataFrame)

    def test_row_count_preserved(self, tmp_path):
        df = _make_df(n=17)
        paths = {"train": _write_csv(tmp_path, "train.csv", df)}
        result = load_final_splits(paths)
        assert len(result["train"]) == 17

    def test_columns_preserved(self, tmp_path):
        df = _make_df()
        paths = {"train": _write_csv(tmp_path, "train.csv", df)}
        result = load_final_splits(paths)
        assert set(result["train"].columns) == set(df.columns)

    def test_data_values_preserved(self, tmp_path):
        df = _make_df(n=5, model_keys=["only_model"], languages=["en"])
        paths = {"train": _write_csv(tmp_path, "train.csv", df)}
        result = load_final_splits(paths)
        assert list(result["train"]["model_key"]) == ["only_model"] * 5

    def test_empty_dict_returns_empty_dict(self):
        result = load_final_splits({})
        assert result == {}

    def test_multiple_splits_loaded_independently(self, tmp_path):
        train_df = _make_df(n=30, seed=0)
        test_df = _make_df(n=10, seed=1)
        paths = {
            "train": _write_csv(tmp_path, "train.csv", train_df),
            "test": _write_csv(tmp_path, "test.csv", test_df),
        }
        result = load_final_splits(paths)
        assert len(result["train"]) == 30
        assert len(result["test"]) == 10

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_final_splits({"train": tmp_path / "nonexistent.csv"})


# ---------------------------------------------------------------------------
# create_split_overview
# ---------------------------------------------------------------------------


class TestCreateSplitOverview:
    def test_returns_list(self):
        splits = {"train": _make_df()}
        result = create_split_overview(splits)
        assert isinstance(result, list)

    def test_one_entry_per_split(self):
        splits = {"train": _make_df(), "val": _make_df(), "test": _make_df()}
        result = create_split_overview(splits)
        assert len(result) == 3

    def test_empty_dict_returns_empty_list(self):
        assert create_split_overview({}) == []

    def test_entry_is_dict(self):
        splits = {"train": _make_df()}
        result = create_split_overview(splits)
        assert isinstance(result[0], dict)

    def test_required_keys_present(self):
        splits = {"train": _make_df()}
        entry = create_split_overview(splits)[0]
        for key in ("split", "rows", "models", "languages", "paraphrase_values"):
            assert key in entry, f"Missing key: {key}"

    def test_split_name_correct(self):
        splits = {"my_split": _make_df()}
        entry = create_split_overview(splits)[0]
        assert entry["split"] == "my_split"

    def test_rows_count_correct(self):
        splits = {"train": _make_df(n=23)}
        entry = create_split_overview(splits)[0]
        assert entry["rows"] == 23

    def test_models_count_correct(self):
        df = _make_df(n=20, model_keys=["a", "b", "c"])
        # Guarantee all three appear
        df["model_key"] = ["a", "b", "c"] * 6 + ["a", "b"]
        splits = {"train": df}
        entry = create_split_overview(splits)[0]
        assert entry["models"] == 3

    def test_languages_count_correct(self):
        df = _make_df(n=10, languages=["en"])
        splits = {"train": df}
        entry = create_split_overview(splits)[0]
        assert entry["languages"] == 1

    def test_paraphrase_values_count_both(self):
        df = _make_df(n=10, paraphrase_values=[True, False])
        # Ensure both values present
        df["is_paraphrase"] = [True, False] * 5
        splits = {"train": df}
        entry = create_split_overview(splits)[0]
        assert entry["paraphrase_values"] == 2

    def test_paraphrase_values_count_one(self):
        df = _make_df(n=6, paraphrase_values=[True])
        splits = {"train": df}
        entry = create_split_overview(splits)[0]
        assert entry["paraphrase_values"] == 1

    def test_single_model_single_language(self):
        df = _make_df(n=5, model_keys=["solo"], languages=["en"], paraphrase_values=[False])
        entry = create_split_overview({"train": df})[0]
        assert entry["models"] == 1
        assert entry["languages"] == 1
        assert entry["paraphrase_values"] == 1

    def test_all_splits_represented(self):
        splits = {"train": _make_df(30), "val": _make_df(10), "test": _make_df(10)}
        result = create_split_overview(splits)
        names = {entry["split"] for entry in result}
        assert names == {"train", "val", "test"}

    def test_rows_match_per_split(self):
        splits = {"train": _make_df(n=40), "val": _make_df(n=8)}
        result = create_split_overview(splits)
        by_split = {entry["split"]: entry for entry in result}
        assert by_split["train"]["rows"] == 40
        assert by_split["val"]["rows"] == 8
