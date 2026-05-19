import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from llm_behavior_xai.llm_response_merger_and_validator.merge_datasets import load_and_merge_datasets


class TestLoadAndMergeDatasets:
    @pytest.fixture
    def temp_csv_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("prompt,response\n")
            f.write("q1,a1\n")
            f.write("q2,a2\n")
            temp_path = Path(f.name)

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def two_temp_csv_files(self):
        paths = []
        for i in range(2):
            with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
                f.write("prompt,response\n")
                f.write(f"q{i * 2 + 1},a{i * 2 + 1}\n")
                f.write(f"q{i * 2 + 2},a{i * 2 + 2}\n")
                paths.append(Path(f.name))

        yield paths
        for p in paths:
            os.unlink(p)

    def test_returns_dict(self, temp_csv_file):
        result = load_and_merge_datasets([("train", [temp_csv_file])])

        assert isinstance(result, dict)

    def test_split_names_are_keys(self, temp_csv_file):
        result = load_and_merge_datasets([("train", [temp_csv_file]), ("val", [temp_csv_file])])

        assert set(result.keys()) == {"train", "val"}

    def test_merges_rows_from_multiple_files(self, two_temp_csv_files):
        result = load_and_merge_datasets([("train", two_temp_csv_files)])

        assert len(result["train"]) == 4

    def test_nonexistent_file_produces_empty_dataframe(self):
        result = load_and_merge_datasets([("train", [Path("/nonexistent/file.csv")])])

        assert result["train"].empty

    def test_single_file_loaded_correctly(self, temp_csv_file):
        result = load_and_merge_datasets([("test", [temp_csv_file])])

        assert len(result["test"]) == 2
        assert list(result["test"].columns) == ["prompt", "response"]

    def test_multiple_splits_processed(self, temp_csv_file):
        datasets = [
            ("train", [temp_csv_file]),
            ("val", [temp_csv_file]),
            ("test", [temp_csv_file]),
        ]
        result = load_and_merge_datasets(datasets)

        assert len(result) == 3

    def test_empty_datasets_list_returns_empty_dict(self):
        result = load_and_merge_datasets([])

        assert result == {}

    def test_result_is_dataframe_per_split(self, temp_csv_file):
        result = load_and_merge_datasets([("train", [temp_csv_file])])

        assert isinstance(result["train"], pd.DataFrame)
