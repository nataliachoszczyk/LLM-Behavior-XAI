import string
import spacy

import numpy as np

from typing import Any
from collections import Counter
from nltk.util import ngrams
from lexicalrichness import LexicalRichness
from numpy import floating, dtype, float64, ndarray
from _typeshed import SupportsDunderGT, SupportsDunderLT

from llm_response_analyzer.text_utils import prepare_text_stats, get_list_of_words


def ttr(text: str) -> float:
    words, N, word_frequency, V = prepare_text_stats(text)

    return V / N if N > 0 else 0


def yule_k(text: str) -> float:
    words, N, word_frequency, V = prepare_text_stats(text)

    if N == 0:
        return 0

    freq_of_freq = Counter(word_frequency.values())
    sum_i2_fi = sum((i**2) * fi for i, fi in freq_of_freq.items())

    return 1e4 * (sum_i2_fi - N) / (N**2)


def guiraud(text: str) -> float:
    words, N, word_frequency, V = prepare_text_stats(text)

    return V / np.sqrt(N) if N > 0 else 0


def honore(text: str) -> float:
    words, N, word_frequency, V = prepare_text_stats(text)

    if N == 0 or V == 0:
        return 0

    V1 = sum(1 for c in word_frequency.values() if c == 1)

    if V == V1:
        return 0

    return 100 * np.log(N) / (1 - V1 / V)


def brunet(text, a=0.165):
    words, N, word_frequency, V = prepare_text_stats(text)

    if N == 0 or V == 0:
        return 0

    return N ** (V ** (-a))


def dugast(text: str) -> int | ndarray[tuple[Any, ...], dtype[float64]]:
    words, N, word_frequency, V = prepare_text_stats(text)

    if N == 0 or V == 0 or V == N:
        return 0

    return (np.log(N) ** 2) / (np.log(N) - np.log(V))


def maas_a2(text: str) -> int | ndarray[tuple[Any, ...], dtype[float64]]:
    words, N, word_frequency, V = prepare_text_stats(text)

    if N == 0 or V == 0:
        return 0

    return (np.log(N) - np.log(V)) / (np.log(N) ** 2)


def entropy(text: str) -> float:
    words, N, word_frequency, V = prepare_text_stats(text)

    if N == 0:
        return 0

    probs = np.array(list(word_frequency.values())) / N

    return -np.sum(probs * np.log(probs))


def repetition_rate(text: str) -> float:
    words, N, word_frequency, V = prepare_text_stats(text)

    return 1 - (V / N) if N > 0 else 0


def hapax_ratio(text: str) -> float:
    words, N, word_frequency, V = prepare_text_stats(text)

    if N == 0:
        return 0

    hapax = sum(1 for c in word_frequency.values() if c == 1)

    return hapax / N


def avg_word_freq(text: str) -> floating[Any] | int:
    words, N, word_frequency, V = prepare_text_stats(text)

    return np.mean(list(word_frequency.values())) if word_frequency else 0


def max_word_freq(text: str) -> SupportsDunderLT[Any] | SupportsDunderGT[Any] | int:
    words, N, word_frequency, V = prepare_text_stats(text)

    return max(word_frequency.values()) if word_frequency else 0


def get_mtld_score(text: str, threshold: float = 0.72) -> float:
    if not text:
        return 0

    lex = LexicalRichness(text)
    mtld_score = lex.mtld(threshold=threshold)

    return mtld_score


def get_lexical_density(text: str, language: str = "en") -> float:
    if not text:
        return 0

    if language == "pl":
        nlp = spacy.load("pl_core_news_sm")
    else:
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)

    lexical_tags = {"NOUN", "VERB", "ADJ", "ADV"}

    lexical_words = [token for token in doc if token.pos_ in lexical_tags]
    total_words = [token for token in doc if not token.is_punct]

    if len(total_words) > 0:
        lexical_density = len(lexical_words) / len(total_words)
    else:
        lexical_density = 0.0

    return lexical_density


def punctuation_density(text: str) -> float:
    if len(text) == 0:
        return 0

    punctuation_count = sum(1 for char in text if char in string.punctuation)

    return punctuation_count / len(text)


def repeated_bigram_ratio(text: str) -> float:
    tokens = get_list_of_words(text)

    if len(tokens) < 2:
        return 0

    bigrams = list(ngrams(tokens, 2))

    total = len(bigrams)
    unique = len(set(bigrams))

    return 1 - (unique / total)


def repeated_trigram_ratio(text: str) -> float:
    tokens = get_list_of_words(text)

    if len(tokens) < 3:
        return 0

    trigrams = list(ngrams(tokens, 3))

    total = len(trigrams)
    unique = len(set(trigrams))

    return 1 - (unique / total)


def max_bigram_frequency(text: str) -> float:
    tokens = get_list_of_words(text)

    if len(tokens) < 2:
        return 0

    bigrams = list(ngrams(tokens, 2))
    counts = Counter(bigrams)

    return max(counts.values())
