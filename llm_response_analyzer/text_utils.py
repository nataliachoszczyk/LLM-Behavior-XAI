import tiktoken

from collections import Counter
from nltk.tokenize import RegexpTokenizer, sent_tokenize


def get_list_of_sentences(text: str) -> list[str]:
    if not text:
        return []

    return sent_tokenize(text)


def count_sentences(text: str) -> int:
    if not text:
        return 0

    sentences = get_list_of_sentences(text)

    return len(sentences)


def get_list_of_words(text: str) -> list[str]:
    if not text:
        return []

    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text.lower())

    return tokens


def get_token_count(text: str) -> int:
    if not text:
        return 0

    enc = tiktoken.get_encoding("o200k_base")
    tokens = enc.encode(text)

    return len(tokens)


def prepare_text_stats(text: str) -> tuple[list[str], int, Counter[str], int]:
    words = get_list_of_words(text)
    N = len(words)

    word_frequency = Counter(words)
    V = len(word_frequency)

    return words, N, word_frequency, V
