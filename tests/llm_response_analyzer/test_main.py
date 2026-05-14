import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import numpy as np

from llm_response_analyzer.main import create_response_features, generate_features_plots


@pytest.fixture
def sample_llm_results_df():
    data = {
        "prompt_id": [1, 2, 3, 4, 5],
        "category": ["definition", "explanation", "reasoning", "synthesis", "analysis"],
        "language": ["en", "en", "en", "en", "en"],
        "is_paraphrase": [False, False, True, False, True],
        "model_key": ["gpt4", "gpt4", "claude", "claude", "gpt4"],
        "provider": ["openai", "openai", "anthropic", "anthropic", "openai"],
        "response": [
            "Artificial intelligence is the simulation of human intelligence by machines.",
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Deep learning uses neural networks with multiple layers.",
            "Neural networks are inspired by biological neurons in the brain.",
            None,
        ],
        "elapsed_seconds": [1.5, 2.0, 1.8, 1.6, 0.5],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_llm_results_df_empty_responses():
    data = {
        "prompt_id": [1, 2, 3],
        "category": ["definition", "explanation", "reasoning"],
        "language": ["en", "en", "en"],
        "is_paraphrase": [False, False, True],
        "model_key": ["gpt4", "gpt4", "claude"],
        "provider": ["openai", "openai", "anthropic"],
        "response": ["", None, ""],
        "elapsed_seconds": [1.5, 2.0, 1.8],
    }
    return pd.DataFrame(data)


class TestCreateResponseFeatures:
    def test_create_response_features_returns_dataframe(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        assert isinstance(result, pd.DataFrame)

    def test_create_response_features_contains_required_columns(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        required_columns = [
            "prompt_id",
            "category",
            "language",
            "is_paraphrase",
            "model_key",
            "provider",
            "response",
            "elapsed_seconds",
            "response_length",
            "response_number_of_words",
            "response_number_of_unique_words",
            "response_number_of_sentences",
            "avg_word_length",
            "avg_words_per_sentence",
            "avg_sentence_length",
            "token_count",
            "type_token_ratio",
            "yule_k",
            "guiraud",
            "honore",
            "brunet",
            "dugast",
            "maas_a2",
            "entropy",
            "repetition_rate",
            "hapax_ratio",
            "avg_word_freq",
            "max_word_freq",
            "mtld",
            "lexical_density",
            "punctuation_density",
            "repeated_bigram_ratio",
            "repeated_trigram_ratio",
            "max_bigram_frequency",
            "sentiment_negative",
            "sentiment_neutral",
            "sentiment_positive",
            "sentiment_compound",
            "semantic_diversity",
            "first_person_pronoun_count",
            "first_person_pronoun_density",
            "sentence_coherence",
            "embedding_variance",
            "flesch_reading_ease",
            "flesch_kincaid_grade",
        ]

        for col in required_columns:
            assert col in result.columns

    def test_create_response_features_correct_number_of_rows(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        assert len(result) == len(sample_llm_results_df)

    def test_create_response_features_handles_nan_responses(self, sample_llm_results_df_empty_responses):
        result = create_response_features(sample_llm_results_df_empty_responses)

        assert result.loc[1, "response_length"] == 0
        assert result.loc[1, "response_number_of_words"] == 0

    def test_create_response_features_response_length_calculated(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        assert result.loc[0, "response_length"] > 0

    def test_create_response_features_token_count_calculated(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        assert result.loc[0, "token_count"] > 0

    def test_create_response_features_type_token_ratio_calculated(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        assert 0 <= result.loc[0, "type_token_ratio"] <= 1
        assert isinstance(result.loc[0, "type_token_ratio"], (int, float))

    def test_create_response_features_sentiment_scores_calculated(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        for col in ["sentiment_negative", "sentiment_neutral", "sentiment_positive", "sentiment_compound"]:
            assert col in result.columns
            assert 0 <= result.loc[0, col] <= 1 or col == "sentiment_compound"

    def test_create_response_features_lexical_diversity_metrics_calculated(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        diversity_metrics = [
            "yule_k",
            "guiraud",
            "honore",
            "brunet",
            "dugast",
            "maas_a2",
            "entropy",
            "repetition_rate",
            "hapax_ratio",
        ]

        for metric in diversity_metrics:
            assert metric in result.columns
            assert isinstance(result.loc[0, metric], (int, float, np.number))

    def test_create_response_features_average_metrics_calculated(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        average_metrics = ["avg_word_length", "avg_words_per_sentence", "avg_sentence_length"]

        for metric in average_metrics:
            assert metric in result.columns
            assert isinstance(result.loc[0, metric], (int, float, np.number))

    def test_create_response_features_punctuation_density_calculated(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        assert "punctuation_density" in result.columns
        assert 0 <= result.loc[0, "punctuation_density"] <= 1

    def test_create_response_features_readability_metrics_calculated(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        assert "flesch_reading_ease" in result.columns
        assert "flesch_kincaid_grade" in result.columns
        assert isinstance(result.loc[0, "flesch_reading_ease"], (int, float, np.number))
        assert isinstance(result.loc[0, "flesch_kincaid_grade"], (int, float, np.number))

    def test_create_response_features_preserves_original_columns(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        original_columns = ["prompt_id", "category", "language", "is_paraphrase", "model_key", "provider"]

        for col in original_columns:
            assert col in result.columns
            pd.testing.assert_series_equal(
                result[col].reset_index(drop=True), sample_llm_results_df[col].reset_index(drop=True), check_names=True
            )

    def test_create_response_features_no_nan_in_features(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        feature_cols = [
            "type_token_ratio",
            "yule_k",
            "guiraud",
            "sentiment_compound",
            "flesch_reading_ease",
        ]

        for col in feature_cols:
            assert result[col].dtype in [float, int] or result[col].dtype == object

    def test_create_response_features_all_values_numeric(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        numeric_columns = [
            "response_length",
            "response_number_of_words",
            "response_number_of_unique_words",
            "response_number_of_sentences",
            "token_count",
            "type_token_ratio",
            "sentiment_negative",
            "sentiment_positive",
        ]

        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(result[col]), f"Column {col} should be numeric"

    def test_create_response_features_features_are_finite(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            assert not np.isinf(result[col].replace([np.inf, -np.inf], np.nan)).any()

    def test_create_response_features_different_languages(self):
        data = {
            "prompt_id": [1, 2],
            "category": ["definition", "explanation"],
            "language": ["en", "pl"],
            "is_paraphrase": [False, False],
            "model_key": ["gpt4", "gpt4"],
            "provider": ["openai", "openai"],
            "response": ["English response text here", "Polski tekst odpowiedzi tutaj"],
            "elapsed_seconds": [1.5, 2.0],
        }

        df = pd.DataFrame(data)
        result = create_response_features(df)

        assert len(result) == 2
        assert result.loc[0, "response_length"] > 0
        assert result.loc[1, "response_length"] > 0

    def test_create_response_features_multiple_models(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)
        unique_models = result["model_key"].unique()

        assert len(unique_models) == 2

        for model in unique_models:
            model_data = result[result["model_key"] == model]

            assert model_data["token_count"].sum() > 0

    @patch("llm_response_analyzer.main.SentenceTransformer")
    def test_create_response_features_with_mocked_sentence_transformer(self, mock_st, sample_llm_results_df):
        mock_instance = MagicMock()
        mock_st.return_value = mock_instance
        mock_instance.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        result = create_response_features(sample_llm_results_df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_llm_results_df)

    def test_create_response_features_elapsed_seconds_preserved(self, sample_llm_results_df):
        result = create_response_features(sample_llm_results_df)

        pd.testing.assert_series_equal(
            result["elapsed_seconds"].reset_index(drop=True),
            sample_llm_results_df["elapsed_seconds"].reset_index(drop=True),
            check_names=True,
        )

    def test_create_response_features_single_row(self):
        data = {
            "prompt_id": [1],
            "category": ["definition"],
            "language": ["en"],
            "is_paraphrase": [False],
            "model_key": ["gpt4"],
            "provider": ["openai"],
            "response": ["This is a test response for a single row."],
            "elapsed_seconds": [1.5],
        }

        df = pd.DataFrame(data)
        result = create_response_features(df)

        assert len(result) == 1
        assert result["response_length"].iloc[0] > 0

    def test_create_response_features_all_empty_responses(self):
        data = {
            "prompt_id": [1, 2, 3],
            "category": ["definition", "explanation", "reasoning"],
            "language": ["en", "en", "en"],
            "is_paraphrase": [False, False, True],
            "model_key": ["gpt4", "gpt4", "claude"],
            "provider": ["openai", "openai", "anthropic"],
            "response": ["", "", ""],
            "elapsed_seconds": [1.5, 2.0, 1.8],
        }

        df = pd.DataFrame(data)
        result = create_response_features(df)

        assert len(result) == 3
        assert result["response_length"].sum() == 0
        assert result["response_number_of_words"].sum() == 0
