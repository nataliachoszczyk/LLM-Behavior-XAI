from __future__ import annotations

import json
import math
import re
import string

import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter


def tokenize_words(text: str, word_re: re.Pattern[str]) -> list[str]:
    return [word.lower() for word in word_re.findall(text)]


def split_sentences(text: str, sentence_re: re.Pattern[str]) -> list[str]:
    sentences = [sentence.strip() for sentence in sentence_re.findall(text) if sentence.strip()]
    return sentences if sentences else ([text.strip()] if text.strip() else [])


def safe_divide(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def mean(values: list[int] | list[float]) -> float:
    return 0.0 if not values else float(np.mean(values))


def count_terms(words: list[str], terms: set[str]) -> int:
    return sum(1 for word in words if word in terms)


def repeated_ngram_ratio(words: list[str], ngram_size: int) -> float:
    if len(words) < ngram_size:
        return 0.0

    ngrams = list(zip(*(words[index:] for index in range(ngram_size))))
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)

    return safe_divide(repeated, len(ngrams))


def word_entropy(word_counts: Counter[str], word_count: int) -> float:
    if word_count == 0:
        return 0.0

    probabilities = [count / word_count for count in word_counts.values()]

    return float(-sum(probability * math.log2(probability) for probability in probabilities))


def count_paragraphs(text: str) -> int:
    return len([paragraph for paragraph in re.split(r"\n\s*\n", text.strip()) if paragraph.strip()])


def extract_response_text_features(
    text: object,
    word_re: re.Pattern[str],
    sentence_re: re.Pattern[str],
    list_marker_re: re.Pattern[str],
    heading_re: re.Pattern[str],
    first_person_pronouns: set[str],
    second_person_pronouns: set[str],
    hedge_words: set[str],
    negation_words: set[str],
) -> dict[str, float]:

    response = "" if pd.isna(text) else str(text)

    words = tokenize_words(response, word_re)
    word_counts = Counter(words)
    sentences = split_sentences(response, sentence_re)
    sentence_char_lengths = [len(sentence) for sentence in sentences]
    sentence_word_lengths = [len(tokenize_words(sentence, word_re)) for sentence in sentences]
    punctuation_count = sum(1 for char in response if char in string.punctuation)
    uppercase_words = [word for word in word_re.findall(response) if len(word) > 1 and word.isupper()]
    char_count = len(response)
    word_count = len(words)
    unique_word_count = len(word_counts)

    return {
        "text_char_count": float(char_count),
        "text_word_count": float(word_count),
        "text_unique_word_count": float(unique_word_count),
        "text_sentence_count": float(len(sentences)),
        "text_paragraph_count": float(count_paragraphs(response)),
        "text_newline_count": float(response.count("\n")),
        "text_avg_word_length": mean([len(word) for word in words]),
        "text_avg_sentence_words": mean(sentence_word_lengths),
        "text_avg_sentence_chars": mean(sentence_char_lengths),
        "text_type_token_ratio": safe_divide(unique_word_count, word_count),
        "text_hapax_ratio": safe_divide(sum(1 for count in word_counts.values() if count == 1), word_count),
        "text_entropy": word_entropy(word_counts, word_count),
        "text_repetition_rate": 1.0 - safe_divide(unique_word_count, word_count),
        "text_avg_word_frequency": mean(list(word_counts.values())),
        "text_max_word_frequency": float(max(word_counts.values(), default=0)),
        "text_repeated_bigram_ratio": repeated_ngram_ratio(words, 2),
        "text_repeated_trigram_ratio": repeated_ngram_ratio(words, 3),
        "text_punctuation_count": float(punctuation_count),
        "text_punctuation_density": safe_divide(punctuation_count, char_count),
        "text_comma_density": safe_divide(response.count(","), char_count),
        "text_colon_density": safe_divide(response.count(":"), char_count),
        "text_semicolon_density": safe_divide(response.count(";"), char_count),
        "text_question_mark_density": safe_divide(response.count("?"), char_count),
        "text_exclamation_mark_density": safe_divide(response.count("!"), char_count),
        "text_digit_density": safe_divide(sum(char.isdigit() for char in response), char_count),
        "text_uppercase_word_ratio": safe_divide(len(uppercase_words), word_count),
        "text_list_marker_count": float(len(list_marker_re.findall(response))),
        "text_markdown_heading_count": float(len(heading_re.findall(response))),
        "text_markdown_bold_count": float(response.count("**") // 2),
        "text_code_fence_count": float(response.count("```") // 2),
        "text_first_person_pronoun_count": float(count_terms(words, first_person_pronouns)),
        "text_first_person_pronoun_density": safe_divide(count_terms(words, first_person_pronouns), word_count),
        "text_second_person_pronoun_count": float(count_terms(words, second_person_pronouns)),
        "text_second_person_pronoun_density": safe_divide(count_terms(words, second_person_pronouns), word_count),
        "text_hedge_word_count": float(count_terms(words, hedge_words)),
        "text_hedge_word_density": safe_divide(count_terms(words, hedge_words), word_count),
        "text_negation_word_count": float(count_terms(words, negation_words)),
        "text_negation_word_density": safe_divide(count_terms(words, negation_words), word_count),
    }


def build_feature_frame(
    final_results_df: pd.DataFrame,
    source_numeric_columns: tuple,
    word_re: re.Pattern[str],
    sentence_re: re.Pattern[str],
    list_marker_re: re.Pattern[str],
    heading_re: re.Pattern[str],
    first_person_pronouns: set[str],
    second_person_pronouns: set[str],
    hedge_words: set[str],
    negation_words: set[str],
) -> pd.DataFrame:

    responses = final_results_df.get("response", pd.Series([""] * len(final_results_df))).fillna("")
    text_features = pd.DataFrame.from_records(
        [
            extract_response_text_features(
                response,
                word_re,
                sentence_re,
                list_marker_re,
                heading_re,
                first_person_pronouns,
                second_person_pronouns,
                hedge_words,
                negation_words,
            )
            for response in responses
        ],
        index=final_results_df.index,
    )

    source_features = pd.DataFrame(index=final_results_df.index)

    for column in source_numeric_columns:
        if column in final_results_df.columns:
            source_features[f"source_{column}"] = pd.to_numeric(final_results_df[column], errors="coerce")

    features = pd.concat([source_features, text_features], axis=1)

    return features.replace([np.inf, -np.inf], np.nan).astype(float)


def build_feature_splits(
    final_splits: dict[str, pd.DataFrame],
    source_numeric_columns: tuple,
    word_re: re.Pattern[str],
    sentence_re: re.Pattern[str],
    list_marker_re: re.Pattern[str],
    heading_re: re.Pattern[str],
    first_person_pronouns: set[str],
    second_person_pronouns: set[str],
    hedge_words: set[str],
    negation_words: set[str],
) -> tuple[dict[str, pd.DataFrame], list[str], pd.Series]:

    raw_features = {
        split: build_feature_frame(
            df,
            source_numeric_columns,
            word_re,
            sentence_re,
            list_marker_re,
            heading_re,
            first_person_pronouns,
            second_person_pronouns,
            hedge_words,
            negation_words,
        )
        for split, df in final_splits.items()
    }

    feature_columns = sorted(raw_features["train"].columns)
    fill_values = raw_features["train"].reindex(columns=feature_columns).median(numeric_only=True).fillna(0.0)
    feature_splits = {}

    for split, features in raw_features.items():
        clean = features.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan)
        feature_splits[split] = clean.fillna(fill_values).fillna(0.0).astype(float)

    return feature_splits, feature_columns, fill_values


def create_feature_descriptions(xai_dir: Path, feature_columns: list[str]) -> pd.DataFrame:
    FEATURE_DESCRIPTIONS = {
        "source_avg_logprob": "CSV value: average token log probability, if available; confidence/fluency proxy.",
        "source_generated_tokens": "CSV value: number of generated tokens, if available; response length proxy.",
        "source_perplexity": "CSV value: perplexity, if available; generation uncertainty proxy.",
        "source_response_length": "CSV value: response length in characters.",
        "source_sum_logprob": "CSV value: total token log probability, if available; confidence plus length proxy.",
        "text_avg_sentence_chars": "Average sentence length in characters.",
        "text_avg_sentence_words": "Average sentence length in words.",
        "text_avg_word_frequency": "Average repetition count per used word.",
        "text_avg_word_length": "Average word length in characters.",
        "text_char_count": "Response length in characters recalculated from response text.",
        "text_code_fence_count": "Number of fenced code blocks marked with triple backticks.",
        "text_colon_density": "Colons per character; often marks lists, explanations, or definitions.",
        "text_comma_density": "Commas per character; rough punctuation/complexity indicator.",
        "text_digit_density": "Digits per character; shows numerical/detail-heavy answers.",
        "text_entropy": "Lexical entropy; higher means word usage is more diverse/uniform.",
        "text_exclamation_mark_density": "Exclamation marks per character; emphasis/expressiveness proxy.",
        "text_first_person_pronoun_count": "Count of first-person words like I/we/ja/my.",
        "text_first_person_pronoun_density": "First-person words divided by total words.",
        "text_hapax_ratio": "Share of words appearing only once; lexical variety proxy.",
        "text_hedge_word_count": "Count of uncertainty words like maybe/could/chyba.",
        "text_hedge_word_density": "Uncertainty words divided by total words.",
        "text_list_marker_count": "Number of bullet or numbered-list markers.",
        "text_markdown_bold_count": "Number of bold markdown spans marked with **.",
        "text_markdown_heading_count": "Number of markdown headings marked with #.",
        "text_max_word_frequency": "Highest repetition count of a single word.",
        "text_negation_word_count": "Count of negation words like not/never/nie.",
        "text_negation_word_density": "Negation words divided by total words.",
        "text_newline_count": "Number of line breaks; formatting/structure proxy.",
        "text_paragraph_count": "Number of paragraph blocks.",
        "text_punctuation_count": "Total punctuation marks.",
        "text_punctuation_density": "Punctuation marks per character.",
        "text_question_mark_density": "Question marks per character.",
        "text_repeated_bigram_ratio": "Share of repeated two-word phrases.",
        "text_repeated_trigram_ratio": "Share of repeated three-word phrases.",
        "text_repetition_rate": "One minus type-token ratio; higher means more repeated vocabulary.",
        "text_second_person_pronoun_count": "Count of second-person words like you/your/ty/wy.",
        "text_second_person_pronoun_density": "Second-person words divided by total words.",
        "text_semicolon_density": "Semicolons per character; complex punctuation indicator.",
        "text_sentence_count": "Number of detected sentences.",
        "text_type_token_ratio": "Unique words divided by total words; lexical diversity proxy.",
        "text_unique_word_count": "Number of distinct words.",
        "text_uppercase_word_ratio": "Uppercase words divided by total words; emphasis/acronym proxy.",
        "text_word_count": "Number of words in the response.",
    }

    feature_meanings = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_source": [
                "direct CSV numeric column" if feature.startswith("source_") else "engineered from CSV response"
                for feature in feature_columns
            ],
            "what_it_shows": [FEATURE_DESCRIPTIONS.get(feature, "No description yet.") for feature in feature_columns],
        }
    )
    feature_meanings.to_csv(xai_dir / "features" / "feature_descriptions.csv", index=False)

    return feature_meanings


def create_feature_list(
    target_columns: tuple,
    xai_dir: Path,
    feature_columns: list[str],
    fill_values: pd.Series,
) -> pd.DataFrame:

    feature_list = pd.DataFrame({"feature": feature_columns})
    feature_list.to_csv(xai_dir / "features" / "feature_list.csv", index=False)

    with (xai_dir / "features" / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_columns": feature_columns,
                "fill_values": fill_values.to_dict(),
                "source": "data/processed/final",
                "targets": target_columns,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    return feature_list
