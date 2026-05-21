from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from llm_behavior_xai.model_training_and_profiles.build_numeric_training_features import (
    build_feature_frame,
    build_feature_splits,
    count_paragraphs,
    count_terms,
    create_feature_descriptions,
    create_feature_list,
    extract_response_text_features,
    mean,
    repeated_ngram_ratio,
    safe_divide,
    split_sentences,
    tokenize_words,
    word_entropy,
)

WORD_RE = re.compile(r"\b[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ]+\b")
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")
LIST_MARKER_RE = re.compile(r"^\s*[-*•]\s+|\d+\.\s+", re.MULTILINE)
HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)

FIRST_PERSON = {"i", "me", "my", "we", "our", "us"}
SECOND_PERSON = {"you", "your", "yours"}
HEDGE_WORDS = {"maybe", "perhaps", "possibly", "could", "might"}
NEGATION_WORDS = {"not", "never", "no", "neither", "nor"}


@pytest.fixture
def nlp_args():
    return dict(
        word_re=WORD_RE,
        sentence_re=SENTENCE_RE,
        list_marker_re=LIST_MARKER_RE,
        heading_re=HEADING_RE,
        first_person_pronouns=FIRST_PERSON,
        second_person_pronouns=SECOND_PERSON,
        hedge_words=HEDGE_WORDS,
        negation_words=NEGATION_WORDS,
    )


class TestTokenizeWords:
    def test_basic(self):
        assert tokenize_words("Hello World", WORD_RE) == ["hello", "world"]

    def test_lowercases(self):
        assert tokenize_words("FOO BAR", WORD_RE) == ["foo", "bar"]

    def test_strips_punctuation(self):
        words = tokenize_words("Hello, world!", WORD_RE)
        assert words == ["hello", "world"]

    def test_empty_string(self):
        assert tokenize_words("", WORD_RE) == []

    def test_only_punctuation(self):
        assert tokenize_words("!!! ???", WORD_RE) == []

    def test_numbers_not_included(self):
        words = tokenize_words("42 bottles", WORD_RE)
        assert "42" not in words
        assert "bottles" in words


class TestSplitSentences:
    def test_splits_on_period(self):
        sentences = split_sentences("Hello world. How are you?", SENTENCE_RE)
        assert len(sentences) == 2

    def test_no_terminal_punct_returns_text(self):
        sentences = split_sentences("No punctuation here", SENTENCE_RE)
        assert sentences == ["No punctuation here"]

    def test_empty_string_returns_empty(self):
        assert split_sentences("", SENTENCE_RE) == []

    def test_strips_whitespace(self):
        sentences = split_sentences("  Hello.  Bye.  ", SENTENCE_RE)
        for s in sentences:
            assert s == s.strip()

    def test_multiple_sentences(self):
        text = "One. Two. Three."
        sentences = split_sentences(text, SENTENCE_RE)
        assert len(sentences) == 3


class TestSafeDivide:
    def test_normal(self):
        assert safe_divide(10, 2) == pytest.approx(5.0)

    def test_zero_denominator(self):
        assert safe_divide(5, 0) == 0.0

    def test_zero_numerator(self):
        assert safe_divide(0, 5) == 0.0

    def test_float_result(self):
        result = safe_divide(1, 3)
        assert isinstance(result, float)
        assert result == pytest.approx(1 / 3)


class TestMean:
    def test_integers(self):
        assert mean([1, 2, 3, 4]) == pytest.approx(2.5)

    def test_empty_list(self):
        assert mean([]) == 0.0

    def test_single_element(self):
        assert mean([7]) == pytest.approx(7.0)

    def test_floats(self):
        assert mean([1.5, 2.5]) == pytest.approx(2.0)


class TestCountTerms:
    def test_counts_matches(self):
        words = ["i", "think", "maybe", "you", "are", "right"]
        assert count_terms(words, {"i", "maybe"}) == 2

    def test_no_matches(self):
        assert count_terms(["hello", "world"], {"foo"}) == 0

    def test_empty_words(self):
        assert count_terms([], {"foo"}) == 0

    def test_empty_terms(self):
        assert count_terms(["hello"], set()) == 0


class TestRepeatedNgramRatio:
    def test_no_repeats(self):
        words = ["a", "b", "c", "d"]
        assert repeated_ngram_ratio(words, 2) == pytest.approx(0.0)

    def test_all_same_bigrams(self):
        words = ["a", "b", "a", "b"]
        ratio = repeated_ngram_ratio(words, 2)
        assert ratio > 0.0

    def test_shorter_than_ngram(self):
        assert repeated_ngram_ratio(["a"], 2) == 0.0

    def test_trigram(self):
        words = ["the", "cat", "sat", "the", "cat", "sat"]
        ratio = repeated_ngram_ratio(words, 3)
        assert ratio > 0.0

    def test_returns_float(self):
        result = repeated_ngram_ratio(["a", "b", "a", "b"], 2)
        assert isinstance(result, float)


class TestWordEntropy:
    def test_uniform_distribution_maximises_entropy(self):
        counts = Counter({"a": 1, "b": 1, "c": 1, "d": 1})
        entropy = word_entropy(counts, 4)
        assert entropy == pytest.approx(2.0)

    def test_single_word_zero_entropy(self):
        counts = Counter({"a": 5})
        assert word_entropy(counts, 5) == pytest.approx(0.0)

    def test_zero_word_count(self):
        assert word_entropy(Counter(), 0) == 0.0

    def test_entropy_positive(self):
        counts = Counter({"a": 3, "b": 1})
        assert word_entropy(counts, 4) > 0.0


class TestCountParagraphs:
    def test_single_paragraph(self):
        assert count_paragraphs("Hello world.") == 1

    def test_two_paragraphs(self):
        assert count_paragraphs("Para one.\n\nPara two.") == 2

    def test_empty_string(self):
        assert count_paragraphs("") == 0

    def test_extra_blank_lines(self):
        assert count_paragraphs("One.\n\n\n\nTwo.") == 2

    def test_only_whitespace(self):
        assert count_paragraphs("   \n\n   ") == 0


class TestExtractResponseTextFeatures:
    def test_returns_dict(self, nlp_args):
        result = extract_response_text_features("Hello world.", **nlp_args)
        assert isinstance(result, dict)

    def test_expected_keys_present(self, nlp_args):
        result = extract_response_text_features("Hello world.", **nlp_args)
        for key in (
            "text_char_count",
            "text_word_count",
            "text_sentence_count",
            "text_entropy",
            "text_type_token_ratio",
        ):
            assert key in result, f"Missing key: {key}"

    def test_all_values_are_float(self, nlp_args):
        result = extract_response_text_features("Some text here.", **nlp_args)
        for k, v in result.items():
            assert isinstance(v, float), f"{k} is {type(v)}, expected float"

    def test_empty_string(self, nlp_args):
        result = extract_response_text_features("", **nlp_args)
        assert result["text_char_count"] == 0.0
        assert result["text_word_count"] == 0.0

    def test_nan_input(self, nlp_args):
        result = extract_response_text_features(float("nan"), **nlp_args)
        assert result["text_char_count"] == 0.0

    def test_char_count(self, nlp_args):
        text = "Hello!"
        result = extract_response_text_features(text, **nlp_args)
        assert result["text_char_count"] == float(len(text))

    def test_word_count(self, nlp_args):
        result = extract_response_text_features("one two three", **nlp_args)
        assert result["text_word_count"] == 3.0

    def test_first_person_pronouns(self, nlp_args):
        result = extract_response_text_features("I think we should go.", **nlp_args)
        assert result["text_first_person_pronoun_count"] == 2.0

    def test_hedge_word_density(self, nlp_args):
        result = extract_response_text_features("maybe you could try this", **nlp_args)
        assert result["text_hedge_word_count"] == 2.0

    def test_negation_words(self, nlp_args):
        result = extract_response_text_features("I am not never going.", **nlp_args)
        assert result["text_negation_word_count"] == 2.0

    def test_markdown_bold_count(self, nlp_args):
        result = extract_response_text_features("This is **bold** and **also bold**.", **nlp_args)
        assert result["text_markdown_bold_count"] == 2.0

    def test_code_fence_count(self, nlp_args):
        result = extract_response_text_features("```python\ncode\n```", **nlp_args)
        assert result["text_code_fence_count"] == 1.0

    def test_list_marker_count(self, nlp_args):
        text = "- item one\n- item two\n- item three"
        result = extract_response_text_features(text, **nlp_args)
        assert result["text_list_marker_count"] == 3.0

    def test_markdown_heading_count(self, nlp_args):
        text = "# Heading\n## Subheading\nBody text."
        result = extract_response_text_features(text, **nlp_args)
        assert result["text_markdown_heading_count"] == 2.0

    def test_type_token_ratio_all_unique(self, nlp_args):
        result = extract_response_text_features("apple banana cherry", **nlp_args)
        assert result["text_type_token_ratio"] == pytest.approx(1.0)

    def test_repetition_rate_all_same(self, nlp_args):
        result = extract_response_text_features("cat cat cat", **nlp_args)
        assert result["text_repetition_rate"] == pytest.approx(2 / 3)


class TestBuildFeatureFrame:
    def _make_df(self, responses):
        return pd.DataFrame({"response": responses})

    def test_returns_dataframe(self, nlp_args):
        df = self._make_df(["Hello world."])
        result = build_feature_frame(df, (), **nlp_args)
        assert isinstance(result, pd.DataFrame)

    def test_row_count_matches(self, nlp_args):
        df = self._make_df(["First.", "Second.", "Third."])
        result = build_feature_frame(df, (), **nlp_args)
        assert len(result) == 3

    def test_no_inf_values(self, nlp_args):
        df = self._make_df(["", "Normal text."])
        result = build_feature_frame(df, (), **nlp_args)
        assert not np.isinf(result.values).any()

    def test_source_column_included(self, nlp_args):
        df = pd.DataFrame({"response": ["Hello."], "response_length": [100]})
        result = build_feature_frame(df, ("response_length",), **nlp_args)
        assert "source_response_length" in result.columns

    def test_missing_source_column_skipped(self, nlp_args):
        df = self._make_df(["Hello."])
        result = build_feature_frame(df, ("nonexistent_col",), **nlp_args)
        assert "source_nonexistent_col" not in result.columns

    def test_missing_response_column(self, nlp_args):
        df = pd.DataFrame({"other": [1, 2]})
        result = build_feature_frame(df, (), **nlp_args)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestBuildFeatureSplits:
    def _make_splits(self, train_texts, test_texts):
        return {
            "train": pd.DataFrame({"response": train_texts}),
            "test": pd.DataFrame({"response": test_texts}),
        }

    def test_returns_tuple(self, nlp_args):
        splits = self._make_splits(["Hello."], ["World."])
        result = build_feature_splits(splits, (), **nlp_args)
        assert isinstance(result, tuple) and len(result) == 3

    def test_split_keys_preserved(self, nlp_args):
        splits = self._make_splits(["Hello."], ["World."])
        feature_splits, _, _ = build_feature_splits(splits, (), **nlp_args)
        assert set(feature_splits.keys()) == {"train", "test"}

    def test_feature_columns_sorted(self, nlp_args):
        splits = self._make_splits(["Hello world."], ["Bye."])
        _, feature_columns, _ = build_feature_splits(splits, (), **nlp_args)
        assert feature_columns == sorted(feature_columns)

    def test_no_nan_in_output(self, nlp_args):
        splits = self._make_splits(["Hello.", ""], ["World."])
        feature_splits, _, _ = build_feature_splits(splits, (), **nlp_args)
        for df in feature_splits.values():
            assert not df.isnull().any().any()

    def test_fill_values_is_series(self, nlp_args):
        splits = self._make_splits(["Hello."], ["World."])
        _, _, fill_values = build_feature_splits(splits, (), **nlp_args)
        assert isinstance(fill_values, pd.Series)

    def test_consistent_columns_across_splits(self, nlp_args):
        splits = self._make_splits(["Hello."], ["World."])
        feature_splits, feature_columns, _ = build_feature_splits(splits, (), **nlp_args)
        for df in feature_splits.values():
            assert list(df.columns) == feature_columns


class TestCreateFeatureDescriptions:
    def test_returns_dataframe(self, tmp_path):
        (tmp_path / "features").mkdir()
        columns = ["text_word_count", "text_char_count", "source_perplexity"]
        result = create_feature_descriptions(tmp_path, columns)
        assert isinstance(result, pd.DataFrame)

    def test_csv_written(self, tmp_path):
        (tmp_path / "features").mkdir()
        create_feature_descriptions(tmp_path, ["text_word_count"])
        assert (tmp_path / "features" / "feature_descriptions.csv").exists()

    def test_feature_column_in_output(self, tmp_path):
        (tmp_path / "features").mkdir()
        columns = ["text_word_count", "text_char_count"]
        result = create_feature_descriptions(tmp_path, columns)
        assert list(result["feature"]) == columns

    def test_unknown_feature_gets_placeholder(self, tmp_path):
        (tmp_path / "features").mkdir()
        result = create_feature_descriptions(tmp_path, ["unknown_feature_xyz"])
        assert result.loc[0, "what_it_shows"] == "No description yet."

    def test_source_feature_labelled_correctly(self, tmp_path):
        (tmp_path / "features").mkdir()
        result = create_feature_descriptions(tmp_path, ["source_perplexity"])
        assert result.loc[0, "feature_source"] == "direct CSV numeric column"

    def test_text_feature_labelled_correctly(self, tmp_path):
        (tmp_path / "features").mkdir()
        result = create_feature_descriptions(tmp_path, ["text_word_count"])
        assert result.loc[0, "feature_source"] == "engineered from CSV response"


class TestCreateFeatureList:
    def _setup(self, tmp_path):
        (tmp_path / "features").mkdir()
        columns = ["text_word_count", "text_char_count"]
        fill_values = pd.Series({"text_word_count": 10.0, "text_char_count": 50.0})
        targets = ("label",)
        return columns, fill_values, targets

    def test_returns_dataframe(self, tmp_path):
        columns, fill_values, targets = self._setup(tmp_path)
        result = create_feature_list(targets, tmp_path, columns, fill_values)
        assert isinstance(result, pd.DataFrame)

    def test_feature_list_csv_written(self, tmp_path):
        columns, fill_values, targets = self._setup(tmp_path)
        create_feature_list(targets, tmp_path, columns, fill_values)
        assert (tmp_path / "features" / "feature_list.csv").exists()

    def test_metadata_json_written(self, tmp_path):
        columns, fill_values, targets = self._setup(tmp_path)
        create_feature_list(targets, tmp_path, columns, fill_values)
        assert (tmp_path / "features" / "feature_metadata.json").exists()

    def test_metadata_json_content(self, tmp_path):
        columns, fill_values, targets = self._setup(tmp_path)
        create_feature_list(targets, tmp_path, columns, fill_values)
        with (tmp_path / "features" / "feature_metadata.json").open() as f:
            meta = json.load(f)
        assert meta["feature_columns"] == columns
        assert meta["targets"] == list(targets)
        assert "fill_values" in meta

    def test_feature_list_rows_match_columns(self, tmp_path):
        columns, fill_values, targets = self._setup(tmp_path)
        result = create_feature_list(targets, tmp_path, columns, fill_values)
        assert list(result["feature"]) == columns
