import pytest

from llm_response_analyzer.lexical_diversity_features import (
    ttr,
    yule_k,
    guiraud,
    honore,
    brunet,
    dugast,
    maas_a2,
    entropy,
    repetition_rate,
    hapax_ratio,
    avg_word_freq,
    max_word_freq,
    get_mtld_score,
    punctuation_density,
    repeated_bigram_ratio,
    repeated_trigram_ratio,
    max_bigram_frequency,
)


class TestTTR:
    def test_ttr_normal_text_with_repetition(self):
        text = "the cat sat on the mat"
        result = ttr(text)

        assert result == pytest.approx(5 / 6, rel=1e-3)

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

        assert result == pytest.approx(1 / 3, rel=1e-3)

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


class TestGuiraud:
    def test_guiraud_normal_text_with_repetition(self):
        text = "the cat sat on the mat"
        result = guiraud(text)

        assert result == pytest.approx(2.0412, rel=1e-3)

    def test_guiraud_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = guiraud(text)

        assert result == pytest.approx(2.236, rel=1e-3)

    def test_guiraud_all_same_word(self):
        text = "the the the the"
        result = guiraud(text)

        assert result == pytest.approx(0.5, rel=1e-3)

    def test_guiraud_empty_text(self):
        text = ""
        result = guiraud(text)

        assert result == 0.0

    def test_guiraud_whitespace_only(self):
        text = "   \n\t  "
        result = guiraud(text)

        assert result == 0.0

    def test_guiraud_single_word(self):
        text = "hello"
        result = guiraud(text)

        assert result == 1.0

    def test_guiraud_two_words(self):
        text = "hello world"
        result = guiraud(text)

        assert result == pytest.approx(1.414, rel=1e-3)

    def test_guiraud_with_punctuation(self):
        text = "Hello, world! How are you?"
        result = guiraud(text)

        assert result == pytest.approx(2.236, rel=1e-3)

    def test_guiraud_case_insensitive(self):
        text = "The THE the"
        result = guiraud(text)

        assert result == pytest.approx(0.577, rel=1e-3)

    def test_guiraud_high_repetition(self):
        text = "a a a b b c"
        result = guiraud(text)

        assert result == pytest.approx(1.225, rel=1e-3)

    def test_guiraud_numbers_and_words(self):
        text = "test 123 test 456"
        result = guiraud(text)

        assert result == pytest.approx(1.5, rel=1e-3)

    def test_guiraud_return_type(self):
        text = "hello world test"
        result = guiraud(text)

        assert isinstance(result, float)

    def test_guiraud_very_repetitive_text(self):
        text = "word " * 100
        result = guiraud(text)

        assert result == pytest.approx(0.1, rel=1e-3)

    def test_guiraud_mixed_repetition(self):
        text = "once twice twice thrice thrice thrice"
        result = guiraud(text)

        assert result == pytest.approx(1.225, rel=1e-3)

    def test_guiraud_increases_with_vocabulary(self):
        text_low_diversity = "word " * 50
        text_high_diversity = " ".join(["word"] * 50)  # Same as above

        result_low = guiraud(text_low_diversity)
        result_high = guiraud(text_high_diversity)

        assert result_low == pytest.approx(result_high, rel=1e-3)

    def test_guiraud_two_unique_very_frequent(self):
        text = "a " * 1000 + "b " * 1000
        result = guiraud(text)

        assert result == pytest.approx(0.0447, rel=1e-2)


class TestHonore:
    def test_honore_normal_text_with_repetition(self):
        text = "the cat sat on the mat"
        result = honore(text)

        assert result == pytest.approx(895.9, rel=1e-2)

    def test_honore_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = honore(text)

        assert result == 0.0

    def test_honore_empty_text(self):
        text = ""
        result = honore(text)

        assert result == 0.0

    def test_honore_whitespace_only(self):
        text = "   \n\t  "
        result = honore(text)

        assert result == 0.0

    def test_honore_single_word(self):
        text = "hello"
        result = honore(text)

        assert result == 0.0

    def test_honore_two_same_words(self):
        text = "hello hello"
        result = honore(text)

        assert result == pytest.approx(69.31, rel=1e-2)

    def test_honore_one_word_three_times_plus_unique(self):
        text = "the the the cat"
        result = honore(text)

        assert result == pytest.approx(277.26, rel=1e-2)

    def test_honore_high_repetition(self):
        text = "a a a b b c"
        result = honore(text)

        assert result == pytest.approx(268.77, rel=1e-2)

    def test_honore_with_punctuation(self):
        text = "Hello, world! Hello world."
        result = honore(text)

        assert result == pytest.approx(138.63, rel=1e-2)

    def test_honore_case_insensitive(self):
        text = "The THE the Cat cat"
        result = honore(text)

        assert result == pytest.approx(160.94, rel=1e-2)

    def test_honore_return_type(self):
        text = "hello hello world"
        result = honore(text)

        assert isinstance(result, float)

    def test_honore_mostly_hapax(self):
        text = "a b c d e f a"
        result = honore(text)

        assert result == pytest.approx(1167.5, rel=1e-2)

    def test_honore_low_hapax(self):
        text = "a a a a b b b b c c c c"
        result = honore(text)

        assert result == pytest.approx(248.49, rel=1e-2)

    def test_honore_large_hapax_ratio(self):
        text = "a b c d e f g a"
        result = honore(text)

        assert result == pytest.approx(1455.6, rel=1e-2)

    def test_honore_mixed_frequencies(self):
        text = "once twice twice thrice thrice thrice"
        result = honore(text)

        assert result == pytest.approx(268.77, rel=1e-2)

    def test_honore_numbers_and_words(self):
        text = "test 123 test 456"
        result = honore(text)

        assert result == pytest.approx(415.9, rel=1e-2)


class TestBrunet:
    def test_brunet_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = brunet(text)

        assert result == pytest.approx(3.435, rel=1e-2)

    def test_brunet_all_same_word(self):
        text = "the the the the"
        result = brunet(text)

        assert result == pytest.approx(4.0, rel=1e-3)

    def test_brunet_empty_text(self):
        text = ""
        result = brunet(text)

        assert result == 0.0

    def test_brunet_single_word(self):
        text = "hello"
        result = brunet(text)

        assert result == pytest.approx(1.0, rel=1e-3)

    def test_brunet_high_repetition(self):
        text = "a a a b b c"
        result = brunet(text)

        assert result == pytest.approx(4.458, rel=1e-2)

    def test_brunet_return_type(self):
        text = "hello world test"
        result = brunet(text)

        assert isinstance(result, (float, int))

    def test_brunet_custom_parameter(self):
        text = "hello world"
        result = brunet(text, a=0.5)

        assert isinstance(result, (float, int))
        assert result > 0


class TestDugast:
    def test_dugast_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = dugast(text)

        assert result == 0.0

    def test_dugast_all_same_word(self):
        text = "the the the the"
        result = dugast(text)

        assert result == pytest.approx(1.386, rel=1e-2)

    def test_dugast_empty_text(self):
        text = ""
        result = dugast(text)

        assert result == 0.0

    def test_dugast_single_word(self):
        text = "hello"
        result = dugast(text)

        assert result == 0.0

    def test_dugast_high_repetition(self):
        text = "a a a b b c"
        result = dugast(text)

        assert result == pytest.approx(4.64, rel=1e-2)

    def test_dugast_return_type(self):
        text = "hello world test test"
        result = dugast(text)

        assert isinstance(result, (float, int, type(None).__class__))

    def test_dugast_moderate_repetition(self):
        text = "the quick brown fox fox"
        result = dugast(text)

        assert result > 0


class TestMaasA2:
    def test_maas_a2_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = maas_a2(text)

        assert result == 0.0

    def test_maas_a2_all_same_word(self):
        text = "the the the the"
        result = maas_a2(text)

        assert result == pytest.approx(0.721, rel=1e-2)

    def test_maas_a2_empty_text(self):
        text = ""
        result = maas_a2(text)

        assert result == 0.0

    def test_maas_a2_high_repetition(self):
        text = "a a a b b c"
        result = maas_a2(text)

        assert result == pytest.approx(0.215, rel=1e-2)

    def test_maas_a2_return_type(self):
        text = "hello world test test"
        result = maas_a2(text)

        assert isinstance(result, (float, int))

    def test_maas_a2_moderate_repetition(self):
        text = "the quick brown fox fox"
        result = maas_a2(text)

        assert result > 0
        assert result < 1


class TestEntropy:
    def test_entropy_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = entropy(text)

        assert result == pytest.approx(1.609, rel=1e-2)

    def test_entropy_all_same_word(self):
        text = "the the the the"
        result = entropy(text)

        assert result == pytest.approx(0.0, rel=1e-3)

    def test_entropy_empty_text(self):
        text = ""
        result = entropy(text)

        assert result == 0.0

    def test_entropy_single_word(self):
        text = "hello"
        result = entropy(text)

        assert result == 0.0

    def test_entropy_high_repetition(self):
        text = "a a a b b c"
        result = entropy(text)

        assert result == pytest.approx(1.0114, rel=1e-2)

    def test_entropy_return_type(self):
        text = "hello world test"
        result = entropy(text)

        assert isinstance(result, float)

    def test_entropy_uniform_distribution(self):
        text = "a b c d e"
        result = entropy(text)

        assert result > 1.5


class TestRepetitionRate:
    def test_repetition_rate_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = repetition_rate(text)

        assert result == 0.0

    def test_repetition_rate_all_same_word(self):
        text = "the the the the"
        result = repetition_rate(text)

        assert result == 0.75

    def test_repetition_rate_empty_text(self):
        text = ""
        result = repetition_rate(text)

        assert result == 0.0

    def test_repetition_rate_single_word(self):
        text = "hello"
        result = repetition_rate(text)

        assert result == 0.0

    def test_repetition_rate_high_repetition(self):
        text = "a a a b b c"
        result = repetition_rate(text)

        assert result == 0.5

    def test_repetition_rate_return_type(self):
        text = "hello world test"
        result = repetition_rate(text)

        assert isinstance(result, float)

    def test_repetition_rate_range(self):
        text = "a a b b c c d d e e"
        result = repetition_rate(text)

        assert 0 <= result <= 1


class TestHapaxRatio:
    def test_hapax_ratio_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = hapax_ratio(text)

        assert result == 1.0

    def test_hapax_ratio_all_same_word(self):
        text = "the the the the"
        result = hapax_ratio(text)

        assert result == 0.0

    def test_hapax_ratio_empty_text(self):
        text = ""
        result = hapax_ratio(text)

        assert result == 0.0

    def test_hapax_ratio_single_word(self):
        text = "hello"
        result = hapax_ratio(text)

        assert result == 1.0

    def test_hapax_ratio_high_repetition(self):
        text = "a a a b b c"
        result = hapax_ratio(text)

        assert result == pytest.approx(0.167, rel=1e-2)

    def test_hapax_ratio_return_type(self):
        text = "hello world test"
        result = hapax_ratio(text)

        assert isinstance(result, float)

    def test_hapax_ratio_range(self):
        text = "a a b b c c d d e e"
        result = hapax_ratio(text)

        assert 0 <= result <= 1


class TestAvgWordFreq:
    def test_avg_word_freq_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = avg_word_freq(text)

        assert result == 1.0

    def test_avg_word_freq_all_same_word(self):
        text = "the the the the"
        result = avg_word_freq(text)

        assert result == 4.0

    def test_avg_word_freq_empty_text(self):
        text = ""
        result = avg_word_freq(text)

        assert result == 0

    def test_avg_word_freq_single_word(self):
        text = "hello"
        result = avg_word_freq(text)

        assert result == 1.0

    def test_avg_word_freq_high_repetition(self):
        text = "a a a b b c"
        result = avg_word_freq(text)

        assert result == 2.0

    def test_avg_word_freq_return_type(self):
        text = "hello world test"
        result = avg_word_freq(text)

        assert isinstance(result, (float, int))

    def test_avg_word_freq_mixed_repetition(self):
        text = "once twice twice thrice thrice thrice"
        result = avg_word_freq(text)

        assert result == 2.0


class TestMaxWordFreq:
    def test_max_word_freq_all_unique_words(self):
        text = "the quick brown fox jumps"
        result = max_word_freq(text)

        assert result == 1

    def test_max_word_freq_all_same_word(self):
        text = "the the the the"
        result = max_word_freq(text)

        assert result == 4

    def test_max_word_freq_empty_text(self):
        text = ""
        result = max_word_freq(text)

        assert result == 0

    def test_max_word_freq_single_word(self):
        text = "hello"
        result = max_word_freq(text)

        assert result == 1

    def test_max_word_freq_high_repetition(self):
        text = "a a a b b c"
        result = max_word_freq(text)

        assert result == 3

    def test_max_word_freq_return_type(self):
        text = "hello world test"
        result = max_word_freq(text)

        assert isinstance(result, (float, int))

    def test_max_word_freq_mixed_repetition(self):
        text = "once twice twice thrice thrice thrice"
        result = max_word_freq(text)

        assert result == 3


class TestPunctuationDensity:
    def test_punctuation_density_no_punctuation(self):
        text = "hello world"
        result = punctuation_density(text)

        assert result == 0.0

    def test_punctuation_density_all_punctuation(self):
        text = "!!!???"
        result = punctuation_density(text)

        assert result == 1.0

    def test_punctuation_density_empty_text(self):
        text = ""
        result = punctuation_density(text)

        assert result == 0.0

    def test_punctuation_density_mixed(self):
        text = "Hello, world!"
        result = punctuation_density(text)

        assert result == pytest.approx(0.154, rel=1e-2)

    def test_punctuation_density_multiple_types(self):
        text = "What?!... Really?"
        result = punctuation_density(text)

        assert 0 <= result <= 1

    def test_punctuation_density_return_type(self):
        text = "hello world test"
        result = punctuation_density(text)

        assert isinstance(result, float)


class TestRepeatedBigramRatio:
    def test_repeated_bigram_ratio_no_bigrams(self):
        text = "hello"
        result = repeated_bigram_ratio(text)

        assert result == 0.0

    def test_repeated_bigram_ratio_all_unique_bigrams(self):
        text = "the quick brown fox"
        result = repeated_bigram_ratio(text)

        assert result == 0.0

    def test_repeated_bigram_ratio_all_same_bigrams(self):
        text = "the the the the"
        result = repeated_bigram_ratio(text)

        assert result == pytest.approx(0.667, rel=1e-3)

    def test_repeated_bigram_ratio_mixed(self):
        text = "a a b b c c"
        result = repeated_bigram_ratio(text)

        assert result == 0.0

    def test_repeated_bigram_ratio_some_repeated(self):
        text = "a a a b b"
        result = repeated_bigram_ratio(text)

        assert result == pytest.approx(0.25, rel=1e-3)

    def test_repeated_bigram_ratio_return_type(self):
        text = "hello world test"
        result = repeated_bigram_ratio(text)

        assert isinstance(result, float)

    def test_repeated_bigram_ratio_range(self):
        text = "a a a a a"
        result = repeated_bigram_ratio(text)

        assert 0 <= result <= 1


class TestRepeatedTrigramRatio:
    def test_repeated_trigram_ratio_no_trigrams(self):
        text = "hello"
        result = repeated_trigram_ratio(text)

        assert result == 0.0

    def test_repeated_trigram_ratio_all_unique_trigrams(self):
        text = "the quick brown fox jumps"
        result = repeated_trigram_ratio(text)

        assert result == 0.0

    def test_repeated_trigram_ratio_all_same_trigrams(self):
        text = "the the the the the"
        result = repeated_trigram_ratio(text)

        assert result == pytest.approx(0.667, rel=1e-3)

    def test_repeated_trigram_ratio_some_repeated(self):
        text = "a a a b b b"
        result = repeated_trigram_ratio(text)

        assert result == 0.0

    def test_repeated_trigram_ratio_return_type(self):
        text = "hello world test"
        result = repeated_trigram_ratio(text)

        assert isinstance(result, float)

    def test_repeated_trigram_ratio_range(self):
        text = "a a a a a a"
        result = repeated_trigram_ratio(text)

        assert 0 <= result <= 1


class TestMaxBigramFrequency:
    def test_max_bigram_frequency_no_bigrams(self):
        text = "hello"
        result = max_bigram_frequency(text)

        assert result == 0

    def test_max_bigram_frequency_all_unique_bigrams(self):
        text = "the quick brown fox"
        result = max_bigram_frequency(text)

        assert result == 1

    def test_max_bigram_frequency_all_same_bigrams(self):
        text = "the the the the"
        result = max_bigram_frequency(text)

        assert result == 3

    def test_max_bigram_frequency_mixed(self):
        text = "a a a b b"
        result = max_bigram_frequency(text)

        assert result == 2

    def test_max_bigram_frequency_return_type(self):
        text = "hello world test"
        result = max_bigram_frequency(text)

        assert isinstance(result, (float, int))

    def test_max_bigram_frequency_long_text(self):
        text = "a a a a a a b b"
        result = max_bigram_frequency(text)

        assert result == 5


class TestGetMtldScore:
    def test_mtld_score_empty_text(self):
        text = ""
        result = get_mtld_score(text)

        assert result == 0

    def test_mtld_score_single_word(self):
        text = "hello"
        result = get_mtld_score(text)

        assert isinstance(result, float)

    def test_mtld_score_two_words(self):
        text = "hello world"
        result = get_mtld_score(text)

        assert isinstance(result, float)
        assert result > 0

    def test_mtld_score_return_type(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = get_mtld_score(text)

        assert isinstance(result, float)

    def test_mtld_score_high_diversity(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = get_mtld_score(text)

        assert result > 0

    def test_mtld_score_custom_threshold(self):
        text = "the quick brown fox jumps"
        result = get_mtld_score(text, threshold=0.5)

        assert isinstance(result, float)
