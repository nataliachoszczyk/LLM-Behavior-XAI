from pathlib import Path

import pandas as pd

from pandas import DataFrame
from pandas.io.parsers import TextFileReader
from sentence_transformers import SentenceTransformer

from config import (
    LLM_RESULTS_TRAIN_PROMPTS,
    LLM_RESULTS_VAL_PROMPTS,
    LLM_RESULTS_TEST_PROMPTS,
    LLM_RESPONSES_TRAIN_FEATURES,
    LLM_RESPONSES_VAL_FEATURES,
    LLM_RESPONSES_TEST_FEATURES,
    LLM_RESPONSES_TRAIN_FEATURES_PLOTS_DIR,
    LLM_RESPONSES_VAL_FEATURES_PLOTS_DIR,
    LLM_RESPONSES_TEST_FEATURES_PLOTS_DIR,
)
from file_utils import read_llm_results, save_results
from llm_response_analyzer.lexical_diversity_features import (
    ttr,
    yule_k,
    guiraud,
    honore,
    brunet,
    dugast,
    maas_a2,
    entropy,
    repetition_rate,
    hapax_ratio,
    avg_word_freq,
    max_word_freq,
    get_mtld_score,
    get_lexical_density,
    punctuation_density,
    repeated_bigram_ratio,
    repeated_trigram_ratio,
    max_bigram_frequency,
)
from llm_response_analyzer.plots import (
    plot_response_length_features,
    plot_response_average_length_features,
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
    plot_elapsed_seconds,
)
from llm_response_analyzer.quantitative_features import token_count_feature, average_length_features, length_features
from llm_response_analyzer.readability_features import get_flesch_reading_ease, get_flesch_kincaid_grade
from llm_response_analyzer.stylistic_and_semantic_features import (
    get_sentiment_scores,
    calculate_semantic_diversity,
    get_first_person_pronouns_count_and_density,
    sentence_coherence,
    embedding_variance,
)


def create_response_features(llm_results_df: TextFileReader | DataFrame):
    response_features = llm_results_df[
        ["prompt_id", "category", "language", "is_paraphrase", "model_key", "provider", "response", "elapsed_seconds"]
    ].copy()

    response_features["response"] = response_features["response"].fillna("")
    responses = llm_results_df["response"].fillna("")

    length_features(response_features, responses)
    average_length_features(response_features)
    token_count_feature(response_features, responses)

    response_features["type_token_ratio"] = responses.apply(ttr)
    response_features["yule_k"] = responses.apply(yule_k)
    response_features["guiraud"] = responses.apply(guiraud)
    response_features["honore"] = responses.apply(honore)
    response_features["brunet"] = responses.apply(brunet)
    response_features["dugast"] = responses.apply(dugast)
    response_features["maas_a2"] = responses.apply(maas_a2)
    response_features["entropy"] = responses.apply(entropy)
    response_features["repetition_rate"] = responses.apply(repetition_rate)
    response_features["hapax_ratio"] = responses.apply(hapax_ratio)
    response_features["avg_word_freq"] = responses.apply(avg_word_freq)
    response_features["max_word_freq"] = responses.apply(max_word_freq)
    response_features["mtld"] = responses.apply(get_mtld_score)
    response_features["lexical_density"] = llm_results_df.apply(
        lambda row: get_lexical_density(row["response"] if pd.notna(row["response"]) else "", row["language"]), axis=1
    )
    response_features["punctuation_density"] = responses.apply(punctuation_density)
    response_features["repeated_bigram_ratio"] = responses.apply(repeated_bigram_ratio)
    response_features["repeated_trigram_ratio"] = responses.apply(repeated_trigram_ratio)
    response_features["max_bigram_frequency"] = responses.apply(max_bigram_frequency)

    sentiment_scores = responses.apply(get_sentiment_scores)
    response_features[["sentiment_negative", "sentiment_neutral", "sentiment_positive", "sentiment_compound"]] = (
        pd.DataFrame(sentiment_scores.tolist(), index=response_features.index)
    )

    sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")

    response_features["semantic_diversity"] = responses.apply(
        lambda text: calculate_semantic_diversity(text, sentence_transformer_model)
    )

    first_person_results = llm_results_df.apply(
        lambda row: get_first_person_pronouns_count_and_density(
            row["response"] if pd.notna(row["response"]) else "", row["language"]
        ),
        axis=1,
    )
    response_features[["first_person_pronoun_count", "first_person_pronoun_density"]] = pd.DataFrame(
        first_person_results.tolist(), index=response_features.index
    )

    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    response_features["sentence_coherence"] = responses.apply(lambda text: sentence_coherence(text, embedding_model))
    response_features["embedding_variance"] = responses.apply(lambda text: embedding_variance(text, embedding_model))

    response_features["flesch_reading_ease"] = llm_results_df.apply(
        lambda row: get_flesch_reading_ease(row["response"] if pd.notna(row["response"]) else "", row["language"]),
        axis=1,
    )
    response_features["flesch_kincaid_grade"] = llm_results_df.apply(
        lambda row: get_flesch_kincaid_grade(row["response"] if pd.notna(row["response"]) else "", row["language"]),
        axis=1,
    )

    return response_features


def generate_features_plots(llm_results_features_plots_dir: Path, response_features: pd.DataFrame):
    plot_elapsed_seconds(response_features, llm_results_features_plots_dir)
    plot_response_length_features(response_features, llm_results_features_plots_dir)
    plot_response_average_length_features(response_features, llm_results_features_plots_dir)
    plot_token_count_feature(response_features, llm_results_features_plots_dir)
    plot_type_token_ratio(response_features, llm_results_features_plots_dir)
    plot_yule_k(response_features, llm_results_features_plots_dir)
    plot_guiraud(response_features, llm_results_features_plots_dir)
    plot_honore(response_features, llm_results_features_plots_dir)
    plot_brunet(response_features, llm_results_features_plots_dir)
    plot_dugast(response_features, llm_results_features_plots_dir)
    plot_maas_a2(response_features, llm_results_features_plots_dir)
    plot_entropy(response_features, llm_results_features_plots_dir)
    plot_repetition_rate(response_features, llm_results_features_plots_dir)
    plot_hapax_ratio(response_features, llm_results_features_plots_dir)
    plot_avg_word_freq(response_features, llm_results_features_plots_dir)
    plot_max_word_freq(response_features, llm_results_features_plots_dir)
    plot_mtld(response_features, llm_results_features_plots_dir)
    plot_lexical_density(response_features, llm_results_features_plots_dir)
    plot_punctuation_density(response_features, llm_results_features_plots_dir)
    plot_repeated_bigram_ratio(response_features, llm_results_features_plots_dir)
    plot_repeated_trigram_ratio(response_features, llm_results_features_plots_dir)
    plot_max_bigram_frequency(response_features, llm_results_features_plots_dir)
    plot_sentiment_features(response_features, llm_results_features_plots_dir)
    plot_semantic_diversity(response_features, llm_results_features_plots_dir)
    plot_person_pronouns_features(response_features, llm_results_features_plots_dir)
    plot_sentence_coherence_feature(response_features, llm_results_features_plots_dir)
    plot_embedding_variance_feature(response_features, llm_results_features_plots_dir)
    plot_flesch_reading_ease_feature(response_features, llm_results_features_plots_dir)
    plot_flesch_kincaid_grade_level_feature(response_features, llm_results_features_plots_dir)
    plot_correlation_heatmap(response_features, llm_results_features_plots_dir)


def main():
    llm_results_paths = [
        (LLM_RESULTS_TRAIN_PROMPTS, LLM_RESPONSES_TRAIN_FEATURES, LLM_RESPONSES_TRAIN_FEATURES_PLOTS_DIR),
        (LLM_RESULTS_VAL_PROMPTS, LLM_RESPONSES_VAL_FEATURES, LLM_RESPONSES_VAL_FEATURES_PLOTS_DIR),
        (LLM_RESULTS_TEST_PROMPTS, LLM_RESPONSES_TEST_FEATURES, LLM_RESPONSES_TEST_FEATURES_PLOTS_DIR),
    ]

    for llm_results_path, llm_results_features_path, llm_results_features_plots_dir in llm_results_paths:
        llm_results_df = read_llm_results(llm_results_path)

        response_features = create_response_features(llm_results_df)
        save_results(response_features, llm_results_features_path)

        llm_results_features_plots_dir.mkdir(parents=True, exist_ok=True)
        generate_features_plots(llm_results_features_plots_dir, response_features)


if __name__ == "__main__":
    main()
