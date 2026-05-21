import os
import pytest
import tempfile

import numpy as np
import pandas as pd

from llm_behavior_xai.file_utils import read_llm_results, read_prompts, save_results


class TestReadLLMResults:
    @pytest.fixture
    def temp_csv_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,question,answer\n")
            f.write("1,What is AI?,Artificial Intelligence\n")
            f.write("2,What is ML?,Machine Learning\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_empty_csv_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,question,answer\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_with_special_chars(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
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

        assert isinstance(result, pd.DataFrame)

    def test_read_csv_file_data_integrity(self, temp_csv_file):
        result = read_llm_results(temp_csv_file)

        assert result.loc[0, "id"] == 1
        assert result.loc[0, "question"] == "What is AI?"
        assert result.loc[1, "answer"] == "Machine Learning"

    def test_read_empty_csv_file(self, temp_empty_csv_file):
        result = read_llm_results(temp_empty_csv_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["id", "question", "answer"]

    def test_read_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            read_llm_results("/path/to/nonexistent/file.csv")

    def test_read_csv_with_special_characters(self, temp_csv_with_special_chars):
        result = read_llm_results(temp_csv_with_special_chars)

        assert isinstance(result, pd.DataFrame)
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is not a CSV file")
            temp_path = f.name

        try:
            result = read_llm_results(temp_path)

            assert isinstance(result, pd.DataFrame)
        finally:
            os.unlink(temp_path)


class TestReadPrompts:
    @pytest.fixture
    def temp_prompts_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("id;prompt;category\n")
            f.write("1;What is artificial intelligence?;definition\n")
            f.write("2;Explain machine learning;explanation\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_empty_prompts_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("id;prompt;category\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_prompts_with_unicode(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("id;prompt;language\n")
            f.write("1;什么是人工智能?;chinese\n")
            f.write("2;Qu'est-ce que l'IA?;french\n")
            f.write("3;AI ってなんですか?;japanese\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_read_valid_prompts_file(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert len(result) == 2
        assert list(result.columns) == ["id", "prompt", "category"]

    def test_read_prompts_file_returns_dataframe(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert isinstance(result, pd.DataFrame)

    def test_read_prompts_file_data_integrity(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert result.loc[0, "id"] == 1
        assert result.loc[0, "prompt"] == "What is artificial intelligence?"
        assert result.loc[1, "category"] == "explanation"

    def test_read_empty_prompts_file(self, temp_empty_prompts_file):
        result = read_prompts(temp_empty_prompts_file)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["id", "prompt", "category"]

    def test_read_prompts_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            read_prompts("/path/to/nonexistent/prompts.csv")

    def test_read_prompts_with_unicode_characters(self, temp_prompts_with_unicode):
        result = read_prompts(temp_prompts_with_unicode)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "什么是人工智能?" in result["prompt"].values
        assert "Qu'est-ce que l'IA?" in result["prompt"].values
        assert "AI ってなんですか?" in result["prompt"].values

    def test_read_prompts_semicolon_separator(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert result.shape[1] == 3
        assert "prompt" in result.columns

    def test_read_prompts_utf8_encoding(self, temp_prompts_with_unicode):
        result = read_prompts(temp_prompts_with_unicode)

        assert any(
            "什么" in str(prompt) or "てなん" in str(prompt) or "français" in str(prompt)
            for prompt in result["prompt"].values
        )

    def test_read_prompts_column_count(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert result.shape[1] == 3

    def test_read_prompts_row_count(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert result.shape[0] == 2

    def test_read_prompts_with_special_characters(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("id;prompt\n")
            f.write("1;Question with commas, punctuation, and symbols: !@#$%^&*()\n")
            f.write('2;Quote\'s and "double quotes"\n')
            temp_path = f.name

        try:
            result = read_prompts(temp_path)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "!@#$%^&*()" in result["prompt"].values[0]
        finally:
            os.unlink(temp_path)


class TestSaveResults:
    @pytest.fixture
    def temp_output_path(self):
        temp_file = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        os.unlink(temp_path)
        yield temp_path

        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_save_results_creates_file(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        save_results(df, temp_output_path)

        assert os.path.exists(temp_output_path)

    def test_save_results_file_is_readable(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2, 3], "value": [10.5, 20.3, 30.1]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert isinstance(result, pd.DataFrame)

    def test_save_results_preserves_data(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert len(result) == 3
        assert list(result.columns) == ["id", "name"]
        assert result.loc[0, "name"] == "Alice"
        assert result.loc[2, "name"] == "Charlie"

    def test_save_results_preserves_column_names(self, temp_output_path):
        df = pd.DataFrame({"column1": [1, 2], "column2": [3, 4], "column3": [5, 6]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert list(result.columns) == ["column1", "column2", "column3"]

    def test_save_results_no_index_column(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2, 3], "value": ["a", "b", "c"]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert result.shape[1] == 2
        assert "Unnamed: 0" not in result.columns

    def test_save_results_with_different_data_types(self, temp_output_path):
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.5, 2.5, 3.5],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert len(result) == 3
        assert result.shape[1] == 4

    def test_save_results_with_unicode_characters(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2, 3], "text": ["Hello", "你好世界", "Bonjour"]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert "你好世界" in result["text"].values

    def test_save_results_with_special_characters(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2], "text": ["Hello, World!", "Special: @#$%^&*()"]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert "Hello, World!" in result["text"].values
        assert "Special: @#$%^&*()" in result["text"].values

    def test_save_results_with_commas_in_data(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2], "description": ["Item 1, with comma", "Item 2, also with comma"]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert result.loc[0, "description"] == "Item 1, with comma"
        assert result.loc[1, "description"] == "Item 2, also with comma"

    def test_save_results_empty_dataframe(self, temp_output_path):
        df = pd.DataFrame({"col1": [], "col2": [], "col3": []})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert len(result) == 0
        assert list(result.columns) == ["col1", "col2", "col3"]

    def test_save_results_single_row(self, temp_output_path):
        df = pd.DataFrame({"id": [1], "value": ["single"]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert len(result) == 1

    def test_save_results_large_dataframe(self, temp_output_path):
        df = pd.DataFrame({"id": list(range(1000)), "value": [f"value_{i}" for i in range(1000)]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert len(result) == 1000
        assert result.shape[1] == 2

    def test_save_results_overwrites_existing_file(self, temp_output_path):
        df1 = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
        save_results(df1, temp_output_path)

        df2 = pd.DataFrame({"id": [10, 20, 30], "value": ["x", "y", "z"]})
        save_results(df2, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert len(result) == 3
        assert result.loc[0, "id"] == 10

    def test_save_results_with_nan_values(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2, 3], "value": [1.5, np.nan, 3.5]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert len(result) == 3
        assert pd.isna(result.loc[1, "value"])

    def test_save_results_with_newlines_in_data(self, temp_output_path):
        df = pd.DataFrame({"id": [1, 2], "text": ["Line 1\nLine 2", "Single line"]})
        save_results(df, temp_output_path)

        result = read_llm_results(temp_output_path)

        assert "Line 1\nLine 2" in result["text"].values
