import pytest
import numpy as np
from unittest.mock import Mock
from sentence_transformers import SentenceTransformer
from llm_behavior_xai.llm_response_analyzer.stylistic_and_semantic_features import (
    get_sentiment_scores,
    calculate_semantic_diversity,
    get_first_person_pronouns_count_and_density,
    sentence_coherence,
    embedding_variance,
)


class TestGetSentimentScores:
    def test_get_sentiment_scores_positive_text(self):
        text = "I love this amazing product! It's fantastic and wonderful."
        result = get_sentiment_scores(text)
        neg, neu, pos, compound = result

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert pos > neg
        assert compound > 0

    def test_get_sentiment_scores_negative_text(self):
        text = "This is terrible and awful. I hate it completely."
        result = get_sentiment_scores(text)
        neg, neu, pos, compound = result

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert neg > pos
        assert compound < 0

    def test_get_sentiment_scores_mixed_sentiment(self):
        text = "The product is good but expensive."
        result = get_sentiment_scores(text)

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(score, float) for score in result)

    def test_get_sentiment_scores_empty_text(self):
        text = ""
        result = get_sentiment_scores(text)

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(score == 0 for score in result)

    def test_get_sentiment_scores_single_word(self):
        text = "great"
        result = get_sentiment_scores(text)
        neg, neu, pos, compound = result

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert pos > 0

    def test_get_sentiment_scores_return_type(self):
        text = "This is a test sentence."
        result = get_sentiment_scores(text)

        assert isinstance(result, tuple)
        assert len(result) == 4
        assert all(isinstance(score, float) for score in result)

    def test_get_sentiment_scores_score_ranges(self):
        text = "This is absolutely fantastic!"
        result = get_sentiment_scores(text)
        neg, neu, pos, compound = result

        assert 0 <= neg <= 1
        assert 0 <= neu <= 1
        assert 0 <= pos <= 1
        assert -1 <= compound <= 1

    def test_get_sentiment_scores_punctuation_handling(self):
        text = "Wow!!! This is incredible!!!"
        result = get_sentiment_scores(text)
        neg, neu, pos, compound = result

        assert isinstance(result, tuple)
        assert compound > 0

    def test_get_sentiment_scores_case_insensitive(self):
        text_upper = "I LOVE THIS PRODUCT!"
        text_lower = "i love this product!"
        result_upper = get_sentiment_scores(text_upper)
        result_lower = get_sentiment_scores(text_lower)

        assert result_upper[3] == pytest.approx(result_lower[3], rel=1e-2)

    def test_get_sentiment_scores_long_text(self):
        text = "This is a very long text that contains many words and should still work properly with the sentiment analysis. It has positive and negative elements mixed together."
        result = get_sentiment_scores(text)

        assert isinstance(result, tuple)
        assert len(result) == 4


class TestCalculateSemanticDiversity:
    @pytest.fixture
    def mock_sbert_model(self):
        model = Mock(spec=SentenceTransformer)

        return model

    def test_calculate_semantic_diversity_single_sentence(self, mock_sbert_model):
        text = "This is a single sentence."
        result = calculate_semantic_diversity(text, mock_sbert_model)

        assert result == 0.0

    def test_calculate_semantic_diversity_empty_text(self, mock_sbert_model):
        text = ""
        result = calculate_semantic_diversity(text, mock_sbert_model)

        assert result == 0.0

    def test_calculate_semantic_diversity_two_sentences(self, mock_sbert_model):
        text = "First sentence. Second sentence."
        mock_embeddings = np.array([[1, 0, 0], [0, 1, 0]])
        mock_sbert_model.encode.return_value = mock_embeddings

        result = calculate_semantic_diversity(text, mock_sbert_model)

        assert isinstance(result, float)
        assert result > 0

    def test_calculate_semantic_diversity_multiple_sentences(self, mock_sbert_model):
        text = "First. Second. Third. Fourth."
        mock_embeddings = np.array([[1, 0, 0], [0.9, 0.1, 0], [0.8, 0.2, 0], [0.7, 0.3, 0]])
        mock_sbert_model.encode.return_value = mock_embeddings

        result = calculate_semantic_diversity(text, mock_sbert_model)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_calculate_semantic_diversity_return_type(self, mock_sbert_model):
        text = "Sentence one. Sentence two."
        mock_embeddings = np.array([[1, 0], [0, 1]])
        mock_sbert_model.encode.return_value = mock_embeddings

        result = calculate_semantic_diversity(text, mock_sbert_model)

        assert isinstance(result, float)

    def test_calculate_semantic_diversity_range(self, mock_sbert_model):
        text = "First sentence. Second sentence."
        mock_embeddings = np.array([[1, 0], [0, 1]])
        mock_sbert_model.encode.return_value = mock_embeddings

        result = calculate_semantic_diversity(text, mock_sbert_model)

        assert 0 <= result <= 1

    def test_calculate_semantic_diversity_similar_sentences(self, mock_sbert_model):
        text = "I like cats. I love cats."
        mock_embeddings = np.array([[1, 0.1], [0.9, 0.1]])
        mock_sbert_model.encode.return_value = mock_embeddings

        result = calculate_semantic_diversity(text, mock_sbert_model)

        assert isinstance(result, float)
        assert result < 0.5

    def test_calculate_semantic_diversity_dissimilar_sentences(self, mock_sbert_model):
        text = "I like cats. Physics is complex."
        mock_embeddings = np.array([[1, 0], [0, 1]])
        mock_sbert_model.encode.return_value = mock_embeddings

        result = calculate_semantic_diversity(text, mock_sbert_model)

        assert isinstance(result, float)
        assert result > 0.4


class TestGetFirstPersonPronounsCountAndDensity:
    def test_get_first_person_pronouns_count_and_density_english_no_pronouns(self):
        text = "The cat sat on the mat."
        count, density = get_first_person_pronouns_count_and_density(text, "en")

        assert count == 0
        assert density == 0.0

    def test_get_first_person_pronouns_count_and_density_english_single_pronoun(self):
        text = "I love programming."
        count, density = get_first_person_pronouns_count_and_density(text, "en")

        assert count == 1
        assert density == pytest.approx(0.333, rel=1e-2)

    def test_get_first_person_pronouns_count_and_density_english_multiple_pronouns(self):
        text = "I think we should go. We are ready."
        count, density = get_first_person_pronouns_count_and_density(text, "en")

        assert count == 3
        assert density == pytest.approx(0.375, rel=1e-2)

    def test_get_first_person_pronouns_count_and_density_polish_no_pronouns(self):
        text = "Kot siedzi na macie."
        count, density = get_first_person_pronouns_count_and_density(text, "pl")

        assert count == 0
        assert density == 0.0

    def test_get_first_person_pronouns_count_and_density_polish_single_pronoun(self):
        text = "Ja kocham programowanie."
        count, density = get_first_person_pronouns_count_and_density(text, "pl")

        assert count == 1
        assert density > 0

    def test_get_first_person_pronouns_count_and_density_polish_multiple_pronouns(self):
        text = "Ja myślę, że my powinniśmy iść."
        count, density = get_first_person_pronouns_count_and_density(text, "pl")

        assert count == 2
        assert density > 0

    def test_get_first_person_pronouns_count_and_density_empty_text(self):
        text = ""
        count, density = get_first_person_pronouns_count_and_density(text, "en")

        assert count == 0
        assert density == 0

    def test_get_first_person_pronouns_count_and_density_return_type(self):
        text = "I am here."
        result = get_first_person_pronouns_count_and_density(text, "en")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_get_first_person_pronouns_count_and_density_case_insensitive(self):
        text = "WE should go."
        count, density = get_first_person_pronouns_count_and_density(text, "en")

        assert count == 1

    def test_get_first_person_pronouns_count_and_density_punctuation_handling(self):
        text = "I, we, and they are here."
        count, density = get_first_person_pronouns_count_and_density(text, "en")

        assert count == 2

    def test_get_first_person_pronouns_count_and_density_mixed_pronouns(self):
        text = "I think you should go, but we will stay."
        count, density = get_first_person_pronouns_count_and_density(text, "en")

        assert count == 2

    def test_get_first_person_pronouns_count_and_density_long_text(self):
        text = "I believe that we can achieve great things together. We should work hard."
        count, density = get_first_person_pronouns_count_and_density(text, "en")

        assert count == 3
        assert density > 0


class TestSentenceCoherence:
    @pytest.fixture
    def mock_embedding_model(self):
        model = Mock(spec=SentenceTransformer)

        return model

    def test_sentence_coherence_single_sentence(self, mock_embedding_model):
        text = "This is a single sentence."
        result = sentence_coherence(text, mock_embedding_model)

        assert result == 0

    def test_sentence_coherence_empty_text(self, mock_embedding_model):
        text = ""
        result = sentence_coherence(text, mock_embedding_model)

        assert result == 0

    def test_sentence_coherence_two_sentences(self, mock_embedding_model):
        text = "First sentence. Second sentence."
        mock_embeddings = np.array([[1, 0.9], [0.9, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = sentence_coherence(text, mock_embedding_model)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_sentence_coherence_multiple_sentences(self, mock_embedding_model):
        text = "First. Second. Third."
        mock_embeddings = np.array([[1, 0.8], [0.9, 0.9], [0.8, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = sentence_coherence(text, mock_embedding_model)

        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_sentence_coherence_return_type(self, mock_embedding_model):
        text = "Sentence one. Sentence two."
        mock_embeddings = np.array([[1, 0], [0, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = sentence_coherence(text, mock_embedding_model)

        assert isinstance(result, float)

    def test_sentence_coherence_range(self, mock_embedding_model):
        text = "First sentence. Second sentence."
        mock_embeddings = np.array([[1, 0.5], [0.5, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = sentence_coherence(text, mock_embedding_model)

        assert 0 <= result <= 1

    def test_sentence_coherence_coherent_text(self, mock_embedding_model):
        text = "I went to the store. I bought some milk."
        mock_embeddings = np.array([[1, 0.9], [0.9, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = sentence_coherence(text, mock_embedding_model)

        assert result > 0.8

    def test_sentence_coherence_incoherent_text(self, mock_embedding_model):
        text = "I love cats. Physics is complex."
        mock_embeddings = np.array([[1, 0], [0, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = sentence_coherence(text, mock_embedding_model)

        assert result < 0.2


class TestEmbeddingVariance:
    @pytest.fixture
    def mock_embedding_model(self):
        model = Mock(spec=SentenceTransformer)

        return model

    def test_embedding_variance_single_sentence(self, mock_embedding_model):
        text = "This is a single sentence."
        result = embedding_variance(text, mock_embedding_model)

        assert result == 0

    def test_embedding_variance_empty_text(self, mock_embedding_model):
        text = ""
        result = embedding_variance(text, mock_embedding_model)

        assert result == 0

    def test_embedding_variance_two_sentences(self, mock_embedding_model):
        text = "First sentence. Second sentence."
        mock_embeddings = np.array([[1, 2], [3, 4]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = embedding_variance(text, mock_embedding_model)

        assert isinstance(result, float)
        assert result >= 0

    def test_embedding_variance_multiple_sentences(self, mock_embedding_model):
        text = "First. Second. Third."
        mock_embeddings = np.array([[1, 1], [2, 2], [3, 3]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = embedding_variance(text, mock_embedding_model)

        assert isinstance(result, float)
        assert result > 0

    def test_embedding_variance_return_type(self, mock_embedding_model):
        text = "Sentence one. Sentence two."
        mock_embeddings = np.array([[1, 0], [0, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = embedding_variance(text, mock_embedding_model)

        assert isinstance(result, float)

    def test_embedding_variance_non_negative(self, mock_embedding_model):
        text = "First sentence. Second sentence."
        mock_embeddings = np.array([[1, 1], [1, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = embedding_variance(text, mock_embedding_model)

        assert result >= 0

    def test_embedding_variance_similar_sentences(self, mock_embedding_model):
        text = "I like cats. I love cats."
        mock_embeddings = np.array([[1, 1.1], [1, 1.2]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = embedding_variance(text, mock_embedding_model)

        assert result < 0.1

    def test_embedding_variance_dissimilar_sentences(self, mock_embedding_model):
        text = "I like cats. Physics is complex."
        mock_embeddings = np.array([[1, 0], [0, 1]])
        mock_embedding_model.encode.return_value = mock_embeddings

        result = embedding_variance(text, mock_embedding_model)

        assert result > 0.2
