import pytest
import tempfile
import os
from pandas import DataFrame
from file_utils import read_llm_results, read_prompts


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


class TestReadPrompts:
    @pytest.fixture
    def temp_prompts_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("id;prompt;category\n")
            f.write("1;What is artificial intelligence?;definition\n")
            f.write("2;Explain machine learning;explanation\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_empty_prompts_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("id;prompt;category\n")
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_prompts_with_unicode(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
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

        assert isinstance(result, DataFrame)

    def test_read_prompts_file_data_integrity(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert result.loc[0, "id"] == 1
        assert result.loc[0, "prompt"] == "What is artificial intelligence?"
        assert result.loc[1, "category"] == "explanation"

    def test_read_empty_prompts_file(self, temp_empty_prompts_file):
        result = read_prompts(temp_empty_prompts_file)

        assert isinstance(result, DataFrame)
        assert len(result) == 0
        assert list(result.columns) == ["id", "prompt", "category"]

    def test_read_prompts_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            read_prompts("/path/to/nonexistent/prompts.csv")

    def test_read_prompts_with_unicode_characters(self, temp_prompts_with_unicode):
        result = read_prompts(temp_prompts_with_unicode)

        assert isinstance(result, DataFrame)
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

        assert any("什么" in str(prompt) or "てなん" in str(prompt) or "français" in str(prompt) for prompt in result["prompt"].values)

    def test_read_prompts_column_count(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert result.shape[1] == 3

    def test_read_prompts_row_count(self, temp_prompts_file):
        result = read_prompts(temp_prompts_file)

        assert result.shape[0] == 2

    def test_read_prompts_with_special_characters(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            f.write("id;prompt\n")
            f.write("1;Question with commas, punctuation, and symbols: !@#$%^&*()\n")
            f.write("2;Quote's and \"double quotes\"\n")
            temp_path = f.name

        try:
            result = read_prompts(temp_path)

            assert isinstance(result, DataFrame)
            assert len(result) == 2
            assert "!@#$%^&*()" in result["prompt"].values[0]
        finally:
            os.unlink(temp_path)

