import pytest
from llm_response_analyzer.lexical_diversity_features import ttr, yule_k


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


class TestYuleK:
    def test_yule_k_normal_text(self):
        text = "the cat sat on the mat"
        result = yule_k(text)

        assert result == pytest.approx(555.5556, rel=1e-4)

    def test_yule_k_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = yule_k(text)

        assert result == 0.0

    def test_yule_k_all_same_word(self):
        text = "the the the the"
        result = yule_k(text)

        assert result == 7500.0

    def test_yule_k_empty_text(self):
        text = ""
        result = yule_k(text)

        assert result == 0.0

    def test_yule_k_whitespace_only(self):
        text = "   \n\t  "
        result = yule_k(text)

        assert result == 0.0

    def test_yule_k_single_word(self):
        text = "hello"
        result = yule_k(text)

        assert result == 0.0

    def test_yule_k_with_punctuation(self):
        text = "Hello, world! Hello world."
        result = yule_k(text)

        assert result == 2500.0

    def test_yule_k_case_insensitive(self):
        text = "The THE the Cat cat"
        result = yule_k(text)

        assert result == 3200.0

    def test_yule_k_high_repetition(self):
        text = "a a a b b c"
        result = yule_k(text)

        assert result == pytest.approx(2222.2222, rel=1e-4)

    def test_yule_k_numbers_and_words(self):
        text = "test 123 test 456 word"
        result = yule_k(text)

        assert result == 800.0

    def test_yule_k_return_type(self):
        text = "hello world test"
        result = yule_k(text)

        assert isinstance(result, float)

    def test_yule_k_complex_distribution(self):
        text = "a b c a b d e f g h i j"
        result = yule_k(text)

        assert result == pytest.approx(277.7778, rel=1e-4)

    def test_yule_k_very_high_values(self):
        text = "word " * 100
        result = yule_k(text)

        assert result == 9900.0

    def test_yule_k_mixed_frequencies(self):
        words = ["once"] * 1 + ["twice"] * 2 + ["thrice"] * 3 + ["four"] * 4
        text = " ".join(words)
        result = yule_k(text)

        assert result == 2000.0
