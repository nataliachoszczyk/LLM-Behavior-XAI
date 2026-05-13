import pytest
from llm_response_analyzer.lexical_diversity_features import ttr


class TestTTR:
    def test_ttr_normal_text_with_repetition(self):
        text = "the cat sat on the mat"
        result = ttr(text)

        assert result == pytest.approx(5/6, rel=1e-3)

    def test_ttr_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = ttr(text)

        assert result == 1.0

    def test_ttr_all_same_word(self):
        text = "the the the the"
        result = ttr(text)

        assert result == 0.25

    def test_ttr_empty_text(self):
        text = ""
        result = ttr(text)

        assert result == 0.0

    def test_ttr_whitespace_only(self):
        text = "   \n\t  "
        result = ttr(text)

        assert result == 0.0

    def test_ttr_single_word(self):
        text = "hello"
        result = ttr(text)

        assert result == 1.0

    def test_ttr_with_punctuation(self):
        text = "Hello, world! How are you?"
        result = ttr(text)

        assert result == 1.0

    def test_ttr_case_insensitive(self):
        text = "The THE the"
        result = ttr(text)

        assert result == pytest.approx(1/3, rel=1e-3)

    def test_ttr_high_repetition(self):
        text = "a a a b b c"
        result = ttr(text)

        assert result == 0.5

    def test_ttr_numbers_and_words(self):
        text = "test 123 test 456"
        result = ttr(text)

        assert result == 0.75

    def test_ttr_mixed_case_and_punctuation(self):
        text = "Hello! hello, WORLD World."
        result = ttr(text)

        assert result == 0.5

    def test_ttr_very_long_text(self):
        text = " ".join(["word"] * 50 + ["unique"] * 50)
        result = ttr(text)

        assert result == 0.02

    def test_ttr_return_type(self):
        text = "hello world"
        result = ttr(text)

        assert isinstance(result, float)

    def test_ttr_with_hyphenated_words(self):
        text = "well-known well-known test"
        result = ttr(text)

        assert result == 0.6
