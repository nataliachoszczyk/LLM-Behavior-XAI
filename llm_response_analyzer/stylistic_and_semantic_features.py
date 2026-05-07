import nltk
import spacy

import numpy as np

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import cosine_similarity

from llm_response_analyzer.text_utils import get_list_of_sentences


def get_sentiment_scores(text: str) -> tuple:
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)

    negative_score = scores.get("neg", 0)
    neutral_score = scores.get("neu", 0)
    positive_score = scores.get("pos", 0)
    compound_score = scores.get("compound", 0)

    return negative_score, neutral_score, positive_score, compound_score


def calculate_semantic_diversity(text: str, sbert_model: SentenceTransformer) -> float:
    sentences = nltk.sent_tokenize(text)

    if len(sentences) < 2:
        return 0.0

    embeddings = sbert_model.encode(sentences)
    cosine_distances = pdist(embeddings, metric='cosine')
    scaled_distances = cosine_distances / 2.0

    return scaled_distances.mean()


def get_first_person_pronouns_count_and_density(text:str, language: str = "en") -> tuple:
    if language == "pl":
        nlp = spacy.load("pl_core_news_sm")
        first_person_lemmas = {'ja', 'my'}
    else:
        nlp = spacy.load("en_core_web_sm")
        first_person_lemmas = {'I', 'we'}

    doc = nlp(text)

    pronoun_count = sum(
        1 for token in doc
        if token.pos_ == 'PRON' and token.lemma_ in first_person_lemmas
    )

    total_words = len([token for token in doc if not token.is_punct])

    pronoun_density = pronoun_count / total_words if total_words > 0 else 0

    return pronoun_count, pronoun_density


def sentence_coherence(text: str, embedding_model: SentenceTransformer) -> float:
    sentences = get_list_of_sentences(text)

    if len(sentences) < 2:
        return 0

    embeddings = embedding_model.encode(sentences)

    similarities = []

    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(similarity)

    return float(np.mean(similarities))


def embedding_variance(text: str, embedding_model: SentenceTransformer) -> float:
    sentences = get_list_of_sentences(text)

    if len(sentences) < 2:
        return 0

    embeddings = embedding_model.encode(sentences)

    return float(np.var(embeddings))
