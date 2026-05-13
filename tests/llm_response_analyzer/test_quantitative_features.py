import pytest
import pandas as pd

from llm_response_analyzer.quantitative_features import token_count_feature, average_length_features, length_features


class TestTokenCountFeature:
    def test_token_count_feature_basic(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["hello world", "the quick brown fox", "a b c d e"])

        token_count_feature(response_features, responses)

        assert "token_count" in response_features.columns
        assert len(response_features) == 3
        assert response_features["token_count"].iloc[0] == 2
        assert response_features["token_count"].iloc[1] == 4
        assert response_features["token_count"].iloc[2] == 5

    def test_token_count_feature_empty_text(self):
        response_features = pd.DataFrame()
        responses = pd.Series([""])

        token_count_feature(response_features, responses)

        assert "token_count" in response_features.columns
        assert response_features["token_count"].iloc[0] == 0

    def test_token_count_feature_single_word(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["hello"])

        token_count_feature(response_features, responses)

        assert response_features["token_count"].iloc[0] == 1

    def test_token_count_feature_with_punctuation(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["Hello, world!", "What? Really!", "One. Two. Three."])

        token_count_feature(response_features, responses)

        assert "token_count" in response_features.columns
        assert len(response_features) == 3
        assert response_features["token_count"].iloc[0] == 4
        assert response_features["token_count"].iloc[1] == 4
        assert response_features["token_count"].iloc[2] == 6

    def test_token_count_feature_with_numbers(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["There are 123 apples", "The year is 2024"])

        token_count_feature(response_features, responses)

        assert response_features["token_count"].iloc[0] == 5
        assert response_features["token_count"].iloc[1] == 6

    def test_token_count_feature_column_added(self):
        response_features = pd.DataFrame({"existing_column": [1, 2, 3]})
        responses = pd.Series(["hello world", "test case", "foo bar baz"])

        token_count_feature(response_features, responses)

        assert "token_count" in response_features.columns
        assert "existing_column" in response_features.columns
        assert len(response_features.columns) == 2

    def test_token_count_feature_large_text(self):
        response_features = pd.DataFrame()
        responses = pd.Series([" ".join(["word"] * 100)])

        token_count_feature(response_features, responses)

        assert response_features["token_count"].iloc[0] == 100

    def test_token_count_feature_whitespace(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["hello    world", "a  b  c", "one   two    three"])

        token_count_feature(response_features, responses)

        assert response_features["token_count"].iloc[0] == 3
        assert response_features["token_count"].iloc[1] == 5
        assert response_features["token_count"].iloc[2] == 5


class TestLengthFeatures:
    def test_length_features_basic(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["hello world", "test", "the quick brown fox jumps"])

        length_features(response_features, responses)

        assert "response_length" in response_features.columns
        assert "response_number_of_words" in response_features.columns
        assert "response_number_of_unique_words" in response_features.columns
        assert "response_number_of_sentences" in response_features.columns

        assert response_features["response_length"].iloc[0] == 11
        assert response_features["response_number_of_words"].iloc[0] == 2
        assert response_features["response_number_of_unique_words"].iloc[0] == 2

    def test_length_features_empty_text(self):
        response_features = pd.DataFrame()
        responses = pd.Series([""])

        length_features(response_features, responses)

        assert response_features["response_length"].iloc[0] == 0
        assert response_features["response_number_of_words"].iloc[0] == 0
        assert response_features["response_number_of_unique_words"].iloc[0] == 0
        assert response_features["response_number_of_sentences"].iloc[0] == 0

    def test_length_features_single_word(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["hello"])

        length_features(response_features, responses)

        assert response_features["response_length"].iloc[0] == 5
        assert response_features["response_number_of_words"].iloc[0] == 1
        assert response_features["response_number_of_unique_words"].iloc[0] == 1

    def test_length_features_repeated_words(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["the the the", "hello world hello"])

        length_features(response_features, responses)

        assert response_features["response_number_of_words"].iloc[0] == 3
        assert response_features["response_number_of_unique_words"].iloc[0] == 1

        assert response_features["response_number_of_words"].iloc[1] == 3
        assert response_features["response_number_of_unique_words"].iloc[1] == 2

    def test_length_features_with_sentences(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["Hello world. This is a test.", "One. Two. Three.", "No ending punctuation"])

        length_features(response_features, responses)

        assert response_features["response_number_of_sentences"].iloc[0] == 2
        assert response_features["response_number_of_sentences"].iloc[1] == 3

    def test_length_features_with_punctuation(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["Hello, world!", "What? Really!", "One. Two. Three."])

        length_features(response_features, responses)

        assert response_features["response_length"].iloc[0] == 13
        assert response_features["response_length"].iloc[1] == 13

    def test_length_features_multiple_responses(self):
        response_features = pd.DataFrame()
        responses = pd.Series(
            [
                "First response",
                "Second response here",
                "Short",
                "This is a much longer response with many words and sentences. It has punctuation!",
            ]
        )

        length_features(response_features, responses)

        assert len(response_features) == 4
        assert response_features["response_number_of_words"].iloc[0] == 2
        assert response_features["response_number_of_words"].iloc[1] == 3

    def test_length_features_with_numbers(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["I have 123 apples", "The year 2024 is here"])

        length_features(response_features, responses)

        assert response_features["response_number_of_words"].iloc[0] == 4
        assert response_features["response_number_of_words"].iloc[1] == 5

    def test_length_features_unicode(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["Hello 你好", "Bonjour Ñiño"])

        length_features(response_features, responses)

        assert response_features["response_number_of_words"].iloc[0] == 2
        assert response_features["response_number_of_words"].iloc[1] == 2


class TestAverageLengthFeatures:
    def test_average_length_features_basic(self):
        response_features = pd.DataFrame(
            {
                "response_length": [11, 4, 25],
                "response_number_of_words": [2, 1, 5],
                "response_number_of_sentences": [1, 1, 2],
            }
        )

        average_length_features(response_features)

        assert "avg_word_length" in response_features.columns
        assert "avg_words_per_sentence" in response_features.columns
        assert "avg_sentence_length" in response_features.columns

    def test_average_length_features_values(self):
        response_features = pd.DataFrame(
            {"response_length": [20], "response_number_of_words": [4], "response_number_of_sentences": [2]}
        )

        average_length_features(response_features)

        assert response_features["avg_word_length"].iloc[0] == 5.0
        assert response_features["avg_words_per_sentence"].iloc[0] == 2.0
        assert response_features["avg_sentence_length"].iloc[0] == 10.0

    def test_average_length_features_division_by_zero_words(self):
        response_features = pd.DataFrame(
            {"response_length": [10], "response_number_of_words": [0], "response_number_of_sentences": [1]}
        )

        average_length_features(response_features)

        assert response_features["avg_word_length"].iloc[0] == 10.0

    def test_average_length_features_division_by_zero_sentences(self):
        response_features = pd.DataFrame(
            {"response_length": [10], "response_number_of_words": [2], "response_number_of_sentences": [0]}
        )

        average_length_features(response_features)

        assert response_features["avg_words_per_sentence"].iloc[0] == 2.0
        assert response_features["avg_sentence_length"].iloc[0] == 10.0

    def test_average_length_features_multiple_rows(self):
        response_features = pd.DataFrame(
            {
                "response_length": [20, 30, 40],
                "response_number_of_words": [4, 5, 8],
                "response_number_of_sentences": [2, 3, 4],
            }
        )

        average_length_features(response_features)

        assert response_features["avg_word_length"].iloc[0] == 5.0
        assert response_features["avg_words_per_sentence"].iloc[0] == 2.0
        assert response_features["avg_sentence_length"].iloc[0] == 10.0

        assert response_features["avg_word_length"].iloc[1] == 6.0
        assert pytest.approx(response_features["avg_words_per_sentence"].iloc[1], rel=1e-2) == 5.0 / 3.0
        assert response_features["avg_sentence_length"].iloc[1] == 10.0

        assert response_features["avg_word_length"].iloc[2] == 5.0
        assert response_features["avg_words_per_sentence"].iloc[2] == 2.0
        assert response_features["avg_sentence_length"].iloc[2] == 10.0

    def test_average_length_features_preserves_existing_columns(self):
        response_features = pd.DataFrame(
            {
                "existing_column": [1, 2, 3],
                "response_length": [20, 30, 40],
                "response_number_of_words": [4, 5, 8],
                "response_number_of_sentences": [2, 3, 4],
            }
        )

        average_length_features(response_features)

        assert "existing_column" in response_features.columns
        assert response_features["existing_column"].iloc[0] == 1

    def test_average_length_features_fractional_results(self):
        response_features = pd.DataFrame(
            {"response_length": [10], "response_number_of_words": [3], "response_number_of_sentences": [2]}
        )

        average_length_features(response_features)

        assert pytest.approx(response_features["avg_word_length"].iloc[0], rel=1e-2) == 10.0 / 3.0
        assert response_features["avg_words_per_sentence"].iloc[0] == 1.5
        assert response_features["avg_sentence_length"].iloc[0] == 5.0

    def test_average_length_features_single_word_single_sentence(self):
        response_features = pd.DataFrame(
            {"response_length": [5], "response_number_of_words": [1], "response_number_of_sentences": [1]}
        )

        average_length_features(response_features)

        assert response_features["avg_word_length"].iloc[0] == 5.0
        assert response_features["avg_words_per_sentence"].iloc[0] == 1.0
        assert response_features["avg_sentence_length"].iloc[0] == 5.0

    def test_average_length_features_zero_length_response(self):
        response_features = pd.DataFrame(
            {"response_length": [0], "response_number_of_words": [0], "response_number_of_sentences": [0]}
        )

        average_length_features(response_features)

        assert response_features["avg_word_length"].iloc[0] == 0.0


class TestIntegrationQuantitativeFeatures:
    def test_full_pipeline_single_response(self):
        response_features = pd.DataFrame()
        responses = pd.Series(["Hello world. This is a test. Great!"])

        token_count_feature(response_features, responses)
        length_features(response_features, responses)
        average_length_features(response_features)

        assert "token_count" in response_features.columns
        assert "response_length" in response_features.columns
        assert "response_number_of_words" in response_features.columns
        assert "response_number_of_unique_words" in response_features.columns
        assert "response_number_of_sentences" in response_features.columns
        assert "avg_word_length" in response_features.columns
        assert "avg_words_per_sentence" in response_features.columns
        assert "avg_sentence_length" in response_features.columns

    def test_full_pipeline_multiple_responses(self):
        response_features = pd.DataFrame()
        responses = pd.Series(
            [
                "Hello world. This is amazing!",
                "Short.",
                "This is a very long response with many words and several sentences. Each sentence adds value. The response is comprehensive!",
            ]
        )

        token_count_feature(response_features, responses)
        length_features(response_features, responses)
        average_length_features(response_features)

        assert len(response_features) == 3
        assert all(response_features["token_count"] > 0)
        assert all(response_features["response_length"] > 0)
        assert all(response_features["avg_word_length"] > 0)

    def test_full_pipeline_with_edge_cases(self):
        response_features = pd.DataFrame()
        responses = pd.Series(
            ["", "a", "One. Two. Three. Four. Five.", "The quick brown fox jumps over the lazy dog " * 5]
        )

        token_count_feature(response_features, responses)
        length_features(response_features, responses)
        average_length_features(response_features)

        assert len(response_features) == 4
        assert response_features["response_length"].iloc[0] == 0
        assert response_features["response_length"].iloc[1] == 1
