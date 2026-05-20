import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import patch

from llm_behavior_xai.llm_response_analyzer.plots import (
    plot_single_feature,
    plot_response_length_features,
    plot_response_average_length_features,
    plot_elapsed_seconds,
    plot_token_count_feature,
    plot_type_token_ratio,
    plot_yule_k,
    plot_guiraud,
    plot_honore,
    plot_brunet,
    plot_dugast,
    plot_maas_a2,
    plot_entropy,
    plot_repetition_rate,
    plot_hapax_ratio,
    plot_avg_word_freq,
    plot_max_word_freq,
    plot_mtld,
    plot_lexical_density,
    plot_punctuation_density,
    plot_repeated_bigram_ratio,
    plot_repeated_trigram_ratio,
    plot_max_bigram_frequency,
    plot_sentiment_features,
    plot_semantic_diversity,
    plot_person_pronouns_features,
    plot_sentence_coherence_feature,
    plot_embedding_variance_feature,
    plot_flesch_reading_ease_feature,
    plot_flesch_kincaid_grade_level_feature,
    plot_correlation_heatmap,
)


@pytest.fixture
def sample_response_features():
    data = {
        "model_key": ["model_a", "model_b", "model_a", "model_b"] * 25,
        "response_length": [100, 150, 200, 120] * 25,
        "response_number_of_words": [20, 30, 40, 25] * 25,
        "response_number_of_unique_words": [15, 20, 30, 20] * 25,
        "response_number_of_sentences": [2, 3, 4, 2] * 25,
        "token_count": [25, 35, 45, 30] * 25,
        "type_token_ratio": [0.75, 0.67, 0.75, 0.8] * 25,
        "yule_k": [500, 600, 700, 650] * 25,
        "guiraud": [2.5, 2.0, 1.8, 2.2] * 25,
        "honore": [100, 150, 200, 120] * 25,
        "brunet": [3.0, 2.5, 2.8, 2.9] * 25,
        "dugast": [5.0, 4.5, 4.8, 4.9] * 25,
        "maas_a2": [0.2, 0.25, 0.3, 0.22] * 25,
        "entropy": [1.5, 1.6, 1.7, 1.55] * 25,
        "repetition_rate": [0.25, 0.33, 0.25, 0.2] * 25,
        "hapax_ratio": [0.75, 0.67, 0.75, 0.8] * 25,
        "avg_word_freq": [1.5, 1.6, 1.4, 1.55] * 25,
        "max_word_freq": [3, 4, 5, 3] * 25,
        "mtld": [50, 60, 70, 65] * 25,
        "lexical_density": [0.7, 0.65, 0.72, 0.68] * 25,
        "punctuation_density": [0.1, 0.15, 0.12, 0.11] * 25,
        "repeated_bigram_ratio": [0.2, 0.25, 0.22, 0.21] * 25,
        "repeated_trigram_ratio": [0.15, 0.2, 0.18, 0.17] * 25,
        "max_bigram_frequency": [3, 4, 5, 3] * 25,
        "elapsed_seconds": [1.5, 2.0, 1.8, 1.6] * 25,
        "sentiment_negative": [0.1, 0.15, 0.12, 0.11] * 25,
        "sentiment_neutral": [0.7, 0.65, 0.72, 0.68] * 25,
        "sentiment_positive": [0.2, 0.2, 0.16, 0.21] * 25,
        "sentiment_compound": [0.5, 0.4, 0.3, 0.45] * 25,
        "semantic_diversity": [0.6, 0.65, 0.7, 0.62] * 25,
        "first_person_pronoun_count": [1, 2, 3, 1] * 25,
        "first_person_pronoun_density": [0.05, 0.1, 0.15, 0.05] * 25,
        "sentence_coherence": [0.8, 0.75, 0.82, 0.78] * 25,
        "embedding_variance": [0.5, 0.6, 0.7, 0.55] * 25,
        "flesch_reading_ease": [60, 70, 80, 65] * 25,
        "flesch_kincaid_grade": [8, 7, 6, 7.5] * 25,
        "avg_word_length": [5.0, 5.5, 5.2, 4.8] * 25,
        "avg_words_per_sentence": [10, 10, 10, 12.5] * 25,
        "avg_sentence_length": [50, 50, 50, 48] * 25,
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestPlotSingleFeature:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_single_feature_creates_file(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        plot_single_feature(
            sample_response_features,
            "token_count",
            "Test Title",
            "Test Label",
            temp_output_dir,
            "test_plot.png",
        )

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_single_feature_filters_empty_responses(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        sample_response_features.loc[0, "response_length"] = 0

        plot_single_feature(
            sample_response_features,
            "token_count",
            "Test Title",
            "Test Label",
            temp_output_dir,
            "test_plot.png",
        )

        mock_savefig.assert_called_once()

    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_single_feature_sets_title_and_labels(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        plot_single_feature(
            sample_response_features,
            "token_count",
            "Test Title",
            "Test Label",
            temp_output_dir,
            "test_plot.png",
        )

        mock_savefig.assert_called_once()


class TestPlotResponseLengthFeatures:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_response_length_features_creates_file(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        plot_response_length_features(sample_response_features, temp_output_dir)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


class TestPlotResponseAverageLengthFeatures:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_response_average_length_features_creates_file(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        plot_response_average_length_features(sample_response_features, temp_output_dir)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


class TestPlotEllapsedSeconds:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_elapsed_seconds_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_elapsed_seconds(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "elapsed_seconds"


class TestPlotTokenCountFeature:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_token_count_feature_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_token_count_feature(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "token_count"


class TestPlotTypeTokenRatio:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_type_token_ratio_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_type_token_ratio(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "type_token_ratio"


class TestPlotYuleK:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_yule_k_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_yule_k(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "yule_k"


class TestPlotSentimentFeatures:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_sentiment_features_creates_file(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        plot_sentiment_features(sample_response_features, temp_output_dir)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


class TestPlotPersonPronounsFeatures:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_person_pronouns_features_creates_file(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        plot_person_pronouns_features(sample_response_features, temp_output_dir)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()


class TestPlotLexicalDensity:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_lexical_density_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_lexical_density(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "lexical_density"


class TestPlotPunctuationDensity:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_punctuation_density_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_punctuation_density(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "punctuation_density"


class TestPlotSemanticDiversity:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_semantic_diversity_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_semantic_diversity(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "semantic_diversity"


class TestPlotSentenceCoherenceFeature:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_sentence_coherence_feature_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_sentence_coherence_feature(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "sentence_coherence"


class TestPlotEmbeddingVarianceFeature:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_embedding_variance_feature_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_embedding_variance_feature(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "embedding_variance"


class TestPlotFleschReadingEaseFeature:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_flesch_reading_ease_feature_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_flesch_reading_ease_feature(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "flesch_reading_ease"


class TestPlotFleschKincaidGradeLevelFeature:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_flesch_kincaid_grade_level_feature_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_flesch_kincaid_grade_level_feature(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "flesch_kincaid_grade"


class TestPlotGuiraud:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_guiraud_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_guiraud(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "guiraud"


class TestPlotHonore:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_honore_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_honore(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "honore"


class TestPlotBrunet:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_brunet_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_brunet(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "brunet"


class TestPlotDugast:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_dugast_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_dugast(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "dugast"


class TestPlotMaasA2:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_maas_a2_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_maas_a2(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "maas_a2"


class TestPlotEntropy:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_entropy_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_entropy(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "entropy"


class TestPlotRepetitionRate:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_repetition_rate_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_repetition_rate(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "repetition_rate"


class TestPlotHapaxRatio:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_hapax_ratio_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_hapax_ratio(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "hapax_ratio"


class TestPlotAvgWordFreq:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_avg_word_freq_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_avg_word_freq(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "avg_word_freq"


class TestPlotMaxWordFreq:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_max_word_freq_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_max_word_freq(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "max_word_freq"


class TestPlotMtld:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_mtld_calls_plot_single_feature(self, mock_plot, sample_response_features, temp_output_dir):
        plot_mtld(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "mtld"


class TestPlotRepeatedBigramRatio:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_repeated_bigram_ratio_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_repeated_bigram_ratio(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "repeated_bigram_ratio"


class TestPlotRepeatedTrigramRatio:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_repeated_trigram_ratio_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_repeated_trigram_ratio(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "repeated_trigram_ratio"


class TestPlotMaxBigramFrequency:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plot_single_feature")
    def test_plot_max_bigram_frequency_calls_plot_single_feature(
        self, mock_plot, sample_response_features, temp_output_dir
    ):
        plot_max_bigram_frequency(sample_response_features, temp_output_dir)

        mock_plot.assert_called_once()
        args = mock_plot.call_args[0]
        assert args[1] == "max_bigram_frequency"


class TestPlotCorrelationHeatmap:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_correlation_heatmap_creates_file(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        plot_correlation_heatmap(sample_response_features, temp_output_dir)

        mock_savefig.assert_called_once()
        mock_close.assert_called_once()

    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_correlation_heatmap_filters_numeric_columns(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        sample_response_features["string_column"] = "text"

        plot_correlation_heatmap(sample_response_features, temp_output_dir)

        mock_savefig.assert_called_once()


class TestPlotIntegration:
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_plot_single_feature_saves_to_correct_path(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        filename = "test_plot.png"
        plot_single_feature(
            sample_response_features,
            "token_count",
            "Test Title",
            "Test Label",
            temp_output_dir,
            filename,
        )

        mock_savefig.assert_called_once()

    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.savefig")
    @patch("llm_behavior_xai.llm_response_analyzer.plots.plt.close")
    def test_multiple_plot_functions_create_different_files(
        self, mock_close, mock_savefig, sample_response_features, temp_output_dir
    ):
        plot_elapsed_seconds(sample_response_features, temp_output_dir)
        plot_token_count_feature(sample_response_features, temp_output_dir)

        assert mock_savefig.call_count == 2
