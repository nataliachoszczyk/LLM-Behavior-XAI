from typing import Any

from pandas import Series, DataFrame

from llm_behavior_xai.llm_response_analyzer.text_utils import get_token_count, get_list_of_words, count_sentences


def token_count_feature(response_features: Series | DataFrame | Any, responses: Series | DataFrame | Any):
    response_features["token_count"] = responses.apply(get_token_count)


def average_length_features(response_features: Series | DataFrame | Any):
    response_features["avg_word_length"] = response_features["response_length"] / response_features[
        "response_number_of_words"
    ].replace(0, 1)

    response_features["avg_words_per_sentence"] = response_features["response_number_of_words"] / response_features[
        "response_number_of_sentences"
    ].replace(0, 1)

    response_features["avg_sentence_length"] = response_features["response_length"] / response_features[
        "response_number_of_sentences"
    ].replace(0, 1)


def length_features(response_features: Series | DataFrame | Any, responses: Series | DataFrame | Any):
    response_features["response_length"] = responses.str.len()
    response_features["response_number_of_words"] = responses.apply(lambda x: len(get_list_of_words(x)))
    response_features["response_number_of_unique_words"] = responses.apply(lambda x: len(set(get_list_of_words(x))))
    response_features["response_number_of_sentences"] = responses.apply(count_sentences)
