from pathlib import Path

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import numpy as np

from llm_response_analyzer.main import create_response_features, generate_features_plots, main


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


@pytest.mark.slow
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


@pytest.mark.slow
class TestGenerateFeaturesPlots:
    @pytest.fixture
    def sample_response_features(self):
        data = {
            "model_key": ["model_a", "model_b"] * 10,
            "response_length": [100, 150] * 10,
            "response_number_of_words": [20, 30] * 10,
            "response_number_of_unique_words": [15, 20] * 10,
            "response_number_of_sentences": [2, 3] * 10,
            "token_count": [25, 35] * 10,
            "type_token_ratio": [0.75, 0.67] * 10,
            "yule_k": [500, 600] * 10,
            "guiraud": [2.5, 2.0] * 10,
            "honore": [100, 150] * 10,
            "brunet": [3.0, 2.5] * 10,
            "dugast": [5.0, 4.5] * 10,
            "maas_a2": [0.2, 0.25] * 10,
            "entropy": [1.5, 1.6] * 10,
            "repetition_rate": [0.25, 0.33] * 10,
            "hapax_ratio": [0.75, 0.67] * 10,
            "avg_word_freq": [1.5, 1.6] * 10,
            "max_word_freq": [3, 4] * 10,
            "mtld": [50, 60] * 10,
            "lexical_density": [0.7, 0.65] * 10,
            "punctuation_density": [0.1, 0.15] * 10,
            "repeated_bigram_ratio": [0.2, 0.25] * 10,
            "repeated_trigram_ratio": [0.15, 0.2] * 10,
            "max_bigram_frequency": [3, 4] * 10,
            "elapsed_seconds": [1.5, 2.0] * 10,
            "sentiment_negative": [0.1, 0.15] * 10,
            "sentiment_neutral": [0.7, 0.65] * 10,
            "sentiment_positive": [0.2, 0.2] * 10,
            "sentiment_compound": [0.5, 0.4] * 10,
            "semantic_diversity": [0.6, 0.65] * 10,
            "first_person_pronoun_count": [1, 2] * 10,
            "first_person_pronoun_density": [0.05, 0.1] * 10,
            "sentence_coherence": [0.8, 0.75] * 10,
            "embedding_variance": [0.5, 0.6] * 10,
            "flesch_reading_ease": [60, 70] * 10,
            "flesch_kincaid_grade": [8, 7] * 10,
            "avg_word_length": [5.0, 5.5] * 10,
            "avg_words_per_sentence": [10, 10] * 10,
            "avg_sentence_length": [50, 50] * 10,
        }
        return pd.DataFrame(data)

    @patch("llm_response_analyzer.main.plot_correlation_heatmap")
    @patch("llm_response_analyzer.main.plot_flesch_kincaid_grade_level_feature")
    @patch("llm_response_analyzer.main.plot_flesch_reading_ease_feature")
    @patch("llm_response_analyzer.main.plot_embedding_variance_feature")
    @patch("llm_response_analyzer.main.plot_sentence_coherence_feature")
    @patch("llm_response_analyzer.main.plot_person_pronouns_features")
    @patch("llm_response_analyzer.main.plot_semantic_diversity")
    @patch("llm_response_analyzer.main.plot_sentiment_features")
    @patch("llm_response_analyzer.main.plot_max_bigram_frequency")
    @patch("llm_response_analyzer.main.plot_repeated_trigram_ratio")
    @patch("llm_response_analyzer.main.plot_repeated_bigram_ratio")
    @patch("llm_response_analyzer.main.plot_punctuation_density")
    @patch("llm_response_analyzer.main.plot_lexical_density")
    @patch("llm_response_analyzer.main.plot_mtld")
    @patch("llm_response_analyzer.main.plot_max_word_freq")
    @patch("llm_response_analyzer.main.plot_avg_word_freq")
    @patch("llm_response_analyzer.main.plot_hapax_ratio")
    @patch("llm_response_analyzer.main.plot_repetition_rate")
    @patch("llm_response_analyzer.main.plot_entropy")
    @patch("llm_response_analyzer.main.plot_maas_a2")
    @patch("llm_response_analyzer.main.plot_dugast")
    @patch("llm_response_analyzer.main.plot_brunet")
    @patch("llm_response_analyzer.main.plot_honore")
    @patch("llm_response_analyzer.main.plot_guiraud")
    @patch("llm_response_analyzer.main.plot_yule_k")
    @patch("llm_response_analyzer.main.plot_type_token_ratio")
    @patch("llm_response_analyzer.main.plot_token_count_feature")
    @patch("llm_response_analyzer.main.plot_response_average_length_features")
    @patch("llm_response_analyzer.main.plot_response_length_features")
    @patch("llm_response_analyzer.main.plot_elapsed_seconds")
    def test_generate_features_plots_calls_all_plot_functions(
        self,
        mock_elapsed,
        mock_resp_length,
        mock_resp_avg_length,
        mock_token_count,
        mock_type_token_ratio,
        mock_yule_k,
        mock_guiraud,
        mock_honore,
        mock_brunet,
        mock_dugast,
        mock_maas_a2,
        mock_entropy,
        mock_repetition_rate,
        mock_hapax_ratio,
        mock_avg_word_freq,
        mock_max_word_freq,
        mock_mtld,
        mock_lexical_density,
        mock_punctuation_density,
        mock_repeated_bigram_ratio,
        mock_repeated_trigram_ratio,
        mock_max_bigram_frequency,
        mock_sentiment_features,
        mock_semantic_diversity,
        mock_person_pronouns_features,
        mock_sentence_coherence_feature,
        mock_embedding_variance_feature,
        mock_flesch_reading_ease_feature,
        mock_flesch_kincaid_grade_level_feature,
        mock_correlation_heatmap,
        sample_response_features,
        temp_dir,
    ):
        output_dir = Path(temp_dir)
        generate_features_plots(output_dir, sample_response_features)

        mock_elapsed.assert_called_once()
        mock_resp_length.assert_called_once()
        mock_resp_avg_length.assert_called_once()
        mock_token_count.assert_called_once()
        mock_type_token_ratio.assert_called_once()
        mock_correlation_heatmap.assert_called_once()


@pytest.fixture
def temp_dir(tmpdir):
    return str(tmpdir)


@pytest.mark.slow
class TestMain:
    @patch("llm_response_analyzer.main.generate_features_plots")
    @patch("llm_response_analyzer.main.save_results")
    @patch("llm_response_analyzer.main.create_response_features")
    @patch("llm_response_analyzer.main.read_llm_results")
    @patch("llm_response_analyzer.main.Path.mkdir")
    def test_main_orchestrates_pipeline(self, mock_mkdir, mock_read, mock_create, mock_save, mock_plot, monkeypatch):
        mock_train_path = Path("/path/to/train.csv")
        mock_train_features_path = Path("/path/to/train_features.csv")
        mock_train_plots_dir = Path("/path/to/train_plots")

        mock_val_path = Path("/path/to/val.csv")
        mock_val_features_path = Path("/path/to/val_features.csv")
        mock_val_plots_dir = Path("/path/to/val_plots")

        mock_test_path = Path("/path/to/test.csv")
        mock_test_features_path = Path("/path/to/test_features.csv")
        mock_test_plots_dir = Path("/path/to/test_plots")

        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESULTS_TRAIN_PROMPTS", mock_train_path)
        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESPONSES_TRAIN_FEATURES", mock_train_features_path)
        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESPONSES_TRAIN_FEATURES_PLOTS_DIR", mock_train_plots_dir)

        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESULTS_VAL_PROMPTS", mock_val_path)
        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESPONSES_VAL_FEATURES", mock_val_features_path)
        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESPONSES_VAL_FEATURES_PLOTS_DIR", mock_val_plots_dir)

        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESULTS_TEST_PROMPTS", mock_test_path)
        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESPONSES_TEST_FEATURES", mock_test_features_path)
        monkeypatch.setattr("llm_response_analyzer.main.LLM_RESPONSES_TEST_FEATURES_PLOTS_DIR", mock_test_plots_dir)

        mock_df = MagicMock()
        mock_read.return_value = mock_df
        mock_features_df = MagicMock()
        mock_create.return_value = mock_features_df

        main()

        assert mock_read.call_count == 3
        assert mock_create.call_count == 3
        assert mock_save.call_count == 3
        assert mock_plot.call_count == 3
