from collections import Counter

from llm_response_analyzer.text_utils import (
    get_list_of_sentences,
    count_sentences,
    get_list_of_words,
    get_token_count,
    prepare_text_stats,
)


class TestGetListOfSentences:
    def test_get_list_of_sentences_basic(self):
        text = "Hello world. This is a test."
        result = get_list_of_sentences(text)

        assert isinstance(result, list)
        assert len(result) == 2
        assert "Hello world." in result
        assert "This is a test." in result

    def test_get_list_of_sentences_single_sentence(self):
        text = "This is a single sentence"
        result = get_list_of_sentences(text)

        assert len(result) == 1
        assert result[0] == "This is a single sentence"

    def test_get_list_of_sentences_empty_text(self):
        text = ""
        result = get_list_of_sentences(text)

        assert result == []
        assert len(result) == 0

    def test_get_list_of_sentences_multiple_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        result = get_list_of_sentences(text)

        assert len(result) == 3

    def test_get_list_of_sentences_with_punctuation(self):
        text = "Is this a question? Yes! Definitely."
        result = get_list_of_sentences(text)

        assert len(result) == 3
        assert "Is this a question?" in result
        assert "Yes!" in result
        assert "Definitely." in result

    def test_get_list_of_sentences_with_abbreviations(self):
        text = "Dr. Smith went to the store. He bought milk."
        result = get_list_of_sentences(text)

        assert len(result) >= 1
        assert isinstance(result[0], str)

    def test_get_list_of_sentences_no_ending_punctuation(self):
        text = "This sentence has no ending"
        result = get_list_of_sentences(text)

        assert len(result) >= 1
        assert result[0] == "This sentence has no ending"

    def test_get_list_of_sentences_with_newlines(self):
        text = "First sentence.\nSecond sentence.\nThird sentence."
        result = get_list_of_sentences(text)

        assert len(result) >= 2

    def test_get_list_of_sentences_with_quotes(self):
        text = 'He said, "Hello world." She replied, "Hi there!"'
        result = get_list_of_sentences(text)

        assert len(result) >= 1
        assert isinstance(result, list)

    def test_get_list_of_sentences_return_type(self):
        text = "One. Two. Three."
        result = get_list_of_sentences(text)

        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)


class TestCountSentences:
    def test_count_sentences_basic(self):
        text = "Hello world. This is a test."
        result = count_sentences(text)

        assert result == 2

    def test_count_sentences_single_sentence(self):
        text = "This is a single sentence."
        result = count_sentences(text)

        assert result == 1

    def test_count_sentences_empty_text(self):
        text = ""
        result = count_sentences(text)

        assert result == 0

    def test_count_sentences_multiple_sentences(self):
        text = "One. Two. Three. Four. Five."
        result = count_sentences(text)

        assert result == 5

    def test_count_sentences_with_questions(self):
        text = "What? Why? How?"
        result = count_sentences(text)

        assert result == 3

    def test_count_sentences_with_exclamations(self):
        text = "Wow! Amazing! Great!"
        result = count_sentences(text)

        assert result == 3

    def test_count_sentences_mixed_punctuation(self):
        text = "Is this right? Yes! Definitely."
        result = count_sentences(text)

        assert result == 3

    def test_count_sentences_return_type(self):
        text = "One. Two. Three."
        result = count_sentences(text)

        assert isinstance(result, int)
        assert result > 0

    def test_count_sentences_long_text(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        result = count_sentences(text)

        assert result == 5

    def test_count_sentences_no_ending_punctuation(self):
        text = "First sentence. Second sentence without punctuation"
        result = count_sentences(text)

        assert result >= 1


class TestGetListOfWords:
    def test_get_list_of_words_basic(self):
        text = "hello world"
        result = get_list_of_words(text)

        assert result == ["hello", "world"]

    def test_get_list_of_words_empty_text(self):
        text = ""
        result = get_list_of_words(text)

        assert result == []

    def test_get_list_of_words_single_word(self):
        text = "hello"
        result = get_list_of_words(text)

        assert result == ["hello"]

    def test_get_list_of_words_case_insensitive(self):
        text = "Hello WORLD Test"
        result = get_list_of_words(text)

        assert result == ["hello", "world", "test"]
        assert all(word.islower() for word in result)

    def test_get_list_of_words_with_punctuation(self):
        text = "Hello, world! How are you?"
        result = get_list_of_words(text)

        assert result == ["hello", "world", "how", "are", "you"]

    def test_get_list_of_words_with_numbers(self):
        text = "There are 123 apples and 456 oranges"
        result = get_list_of_words(text)

        assert "123" in result
        assert "456" in result
        assert len(result) == 7

    def test_get_list_of_words_with_hyphens(self):
        text = "well-known test-case"
        result = get_list_of_words(text)

        assert "well" in result
        assert "known" in result
        assert "test" in result
        assert "case" in result

    def test_get_list_of_words_with_apostrophes(self):
        text = "don't can't won't"
        result = get_list_of_words(text)

        assert "don" in result or "don't" in result
        assert len(result) > 0

    def test_get_list_of_words_multiple_spaces(self):
        text = "hello    world    test"
        result = get_list_of_words(text)

        assert result == ["hello", "world", "test"]

    def test_get_list_of_words_return_type(self):
        text = "hello world test"
        result = get_list_of_words(text)

        assert isinstance(result, list)
        assert all(isinstance(word, str) for word in result)

    def test_get_list_of_words_whitespace_only(self):
        text = "   \n\t  "
        result = get_list_of_words(text)

        assert result == []

    def test_get_list_of_words_long_text(self):
        text = " ".join(["word"] * 100)
        result = get_list_of_words(text)

        assert len(result) == 100
        assert all(word == "word" for word in result)


class TestGetTokenCount:
    def test_get_token_count_basic(self):
        text = "hello world"
        result = get_token_count(text)

        assert isinstance(result, int)
        assert result > 0

    def test_get_token_count_empty_text(self):
        text = ""
        result = get_token_count(text)

        assert result == 0

    def test_get_token_count_single_word(self):
        text = "hello"
        result = get_token_count(text)

        assert result >= 1

    def test_get_token_count_increases_with_length(self):
        short_text = "hello"
        long_text = "hello world this is a much longer text with many words"

        short_count = get_token_count(short_text)
        long_count = get_token_count(long_text)

        assert long_count > short_count

    def test_get_token_count_with_punctuation(self):
        text = "Hello, world! How are you?"
        result = get_token_count(text)

        assert result > 0
        assert isinstance(result, int)

    def test_get_token_count_with_numbers(self):
        text = "I have 123 apples"
        result = get_token_count(text)

        assert result > 0

    def test_get_token_count_return_type(self):
        text = "test text"
        result = get_token_count(text)

        assert isinstance(result, int)

    def test_get_token_count_sensible_ratio(self):
        text = "one two three four five"
        token_count = get_token_count(text)
        word_count = len(text.split())

        assert token_count > 0
        assert word_count == 5

    def test_get_token_count_with_special_characters(self):
        text = "Test@#$%^&*()_+-=[]{}|;:',.<>?/"
        result = get_token_count(text)

        assert result > 0

    def test_get_token_count_unicode(self):
        text = "Hello 你好 مرحبا"
        result = get_token_count(text)

        assert result > 0
        assert isinstance(result, int)


class TestPrepareTextStats:
    def test_prepare_text_stats_basic(self):
        text = "hello world hello"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert isinstance(words, list)
        assert isinstance(N, int)
        assert isinstance(word_frequency, Counter)
        assert isinstance(V, int)

        assert words == ["hello", "world", "hello"]
        assert N == 3
        assert V == 2
        assert word_frequency["hello"] == 2
        assert word_frequency["world"] == 1

    def test_prepare_text_stats_empty_text(self):
        text = ""
        words, N, word_frequency, V = prepare_text_stats(text)

        assert words == []
        assert N == 0
        assert len(word_frequency) == 0
        assert V == 0

    def test_prepare_text_stats_single_word(self):
        text = "hello"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert words == ["hello"]
        assert N == 1
        assert V == 1
        assert word_frequency["hello"] == 1

    def test_prepare_text_stats_all_unique(self):
        text = "one two three four five"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert N == 5
        assert V == 5
        assert all(freq == 1 for freq in word_frequency.values())

    def test_prepare_text_stats_all_same_word(self):
        text = "test test test test"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert N == 4
        assert V == 1
        assert word_frequency["test"] == 4

    def test_prepare_text_stats_case_insensitive(self):
        text = "Test test TEST"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert N == 3
        assert V == 1
        assert word_frequency["test"] == 3

    def test_prepare_text_stats_removes_punctuation(self):
        text = "Hello, world! How are you?"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert "hello" in words
        assert "world" in words
        assert "how" in words
        assert "are" in words
        assert "you" in words
        assert all("," not in word and "!" not in word and "?" not in word for word in words)

    def test_prepare_text_stats_mixed_repetition(self):
        text = "a a a b b c"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert N == 6
        assert V == 3
        assert word_frequency["a"] == 3
        assert word_frequency["b"] == 2
        assert word_frequency["c"] == 1

    def test_prepare_text_stats_long_text(self):
        text = " ".join(["word"] * 100)
        words, N, word_frequency, V = prepare_text_stats(text)

        assert N == 100
        assert V == 1
        assert word_frequency["word"] == 100

    def test_prepare_text_stats_return_types(self):
        text = "test text"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert isinstance(words, list)
        assert isinstance(N, int)
        assert isinstance(word_frequency, Counter)
        assert isinstance(V, int)

    def test_prepare_text_stats_counter_properties(self):
        text = "apple banana apple"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert word_frequency["apple"] == 2
        assert word_frequency["banana"] == 1
        assert word_frequency["nonexistent"] == 0  # Counter returns 0 for non-existent keys

    def test_prepare_text_stats_whitespace_only(self):
        text = "   \n\t  "
        words, N, word_frequency, V = prepare_text_stats(text)

        assert words == []
        assert N == 0
        assert V == 0

    def test_prepare_text_stats_with_numbers(self):
        text = "one 1 two 2 three 3"
        words, N, word_frequency, V = prepare_text_stats(text)

        assert "1" in words
        assert "2" in words
        assert "3" in words
        assert N == 6


class TestIntegrationTextUtils:
    def test_full_text_analysis(self):
        text = "Hello world. This is a test. Hello again!"

        sentences = get_list_of_sentences(text)
        sentence_count = count_sentences(text)
        words = get_list_of_words(text)
        token_count = get_token_count(text)
        stats = prepare_text_stats(text)

        assert len(sentences) == sentence_count
        assert len(words) == stats[1]
        assert token_count > 0

    def test_empty_text_consistency(self):
        text = ""

        assert get_list_of_sentences(text) == []
        assert count_sentences(text) == 0
        assert get_list_of_words(text) == []
        assert get_token_count(text) == 0

        words, N, word_freq, V = prepare_text_stats(text)

        assert words == []
        assert N == 0
        assert V == 0

    def test_single_word_consistency(self):
        text = "hello"

        assert len(get_list_of_sentences(text)) >= 1
        assert count_sentences(text) >= 1
        assert get_list_of_words(text) == ["hello"]
        assert get_token_count(text) >= 1

    def test_multiple_sentences_analysis(self):
        text = "First. Second. Third."

        sentences = get_list_of_sentences(text)

        assert len(sentences) == 3

        words = get_list_of_words(text)

        assert len(words) == 3
        assert words == ["first", "second", "third"]

    def test_complex_text_analysis(self):
        text = "Dr. Smith went to the store. He bought 5 apples and 10 oranges. Wow!"

        sentences = get_list_of_sentences(text)

        assert len(sentences) >= 2

        words = get_list_of_words(text)

        assert "smith" in words
        assert "store" in words
        assert "5" in words
        assert "10" in words

        stats = prepare_text_stats(text)

        assert stats[1] > 0
        assert stats[3] > 0
