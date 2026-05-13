import pytest
import pandas as pd
import tempfile
import os
from pathlib import Path
from pandas import DataFrame
from file_utils import read_llm_results


class TestReadLLMResults:
    @pytest.fixture
    def temp_csv_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,question,answer\n")
            f.write("1,What is AI?,Artificial Intelligence\n")
            f.write("2,What is ML?,Machine Learning\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_empty_csv_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,question,answer\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_with_special_chars(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("id,text\n")
            f.write("1,Hello World!\n")
            f.write("2,Special chars: @#$%\n")
            f.write("3,Unicode: 你好世界\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_read_valid_csv_file(self, temp_csv_file):
        result = read_llm_results(temp_csv_file)

        assert len(result) == 2
        assert list(result.columns) == ["id", "question", "answer"]

    def test_read_csv_file_returns_dataframe(self, temp_csv_file):
        result = read_llm_results(temp_csv_file)

        assert isinstance(result, DataFrame)

    def test_read_csv_file_data_integrity(self, temp_csv_file):
        result = read_llm_results(temp_csv_file)

        assert result.loc[0, "id"] == 1
        assert result.loc[0, "question"] == "What is AI?"
        assert result.loc[1, "answer"] == "Machine Learning"

    def test_read_empty_csv_file(self, temp_empty_csv_file):
        result = read_llm_results(temp_empty_csv_file)

        assert isinstance(result, DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["id", "question", "answer"]

    def test_read_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            read_llm_results("/path/to/nonexistent/file.csv")

    def test_read_csv_with_special_characters(self, temp_csv_with_special_chars):
        result = read_llm_results(temp_csv_with_special_chars)

        assert isinstance(result, DataFrame)
        assert len(result) == 3
        assert "Hello World!" in result["text"].values
        assert "Unicode: 你好世界" in result["text"].values

    def test_read_csv_column_count(self, temp_csv_file):
        result = read_llm_results(temp_csv_file)

        assert result.shape[1] == 3

    def test_read_csv_row_count(self, temp_csv_file):
        result = read_llm_results(temp_csv_file)

        assert result.shape[0] == 2

    def test_read_csv_dtypes(self, temp_csv_file):
        result = read_llm_results(temp_csv_file)

        assert "id" in result.columns
        assert "question" in result.columns
        assert "answer" in result.columns

    def test_read_invalid_file_format(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is not a CSV file")
            temp_path = f.name

        try:
            result = read_llm_results(temp_path)

            assert isinstance(result, DataFrame)
        finally:
            os.unlink(temp_path)
