import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


def plot_single_feature(
    response_features: pd.DataFrame, feature: str, title: str, xlabel: str, output_dir: Path, filename: str
) -> None:

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x=feature,
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def plot_response_length_features(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    features = [
        "response_length",
        "response_number_of_words",
        "response_number_of_unique_words",
        "response_number_of_sentences",
    ]
    titles = ["Response Length (chars)", "Number of Words", "Number of Unique Words", "Number of Sentences"]

    for ax, feature, title in zip(axes, features, titles):
        sns.histplot(
            data=response_features[response_features["response_length"] > 0],
            x=feature,
            hue="model_key",
            kde=True,
            ax=ax,
            bins=30,
            alpha=0.5,
        )

        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")

    plt.suptitle("Distribution of Response Length Features by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "response_length_features.png")
    plt.close()


def plot_response_average_length_features(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    features = ["avg_word_length", "avg_words_per_sentence", "avg_sentence_length"]
    titles = ["Avg Word Length (chars/word)", "Avg Words per Sentence", "Avg Sentence Length (chars)"]

    for ax, feature, title in zip(axes, features, titles):
        sns.histplot(
            data=response_features[response_features["response_length"] > 0],
            x=feature,
            hue="model_key",
            kde=True,
            ax=ax,
            bins=30,
            alpha=0.5,
        )

        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")

    plt.suptitle("Distribution of Avg Length Features by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "avg_length_features.png")
    plt.close()


def plot_elapsed_seconds(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "elapsed_seconds",
        "Distribution of Times Needed to Generate a Response by Model",
        "Time [seconds]",
        output_dir,
        "elapsed_seconds.png",
    )


def plot_token_count_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "token_count",
        "Token Count Distribution by Model",
        "Token Count",
        output_dir,
        "token_count_feature.png",
    )


def plot_type_token_ratio(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "type_token_ratio",
        "Type-Token Ratio (TTR) Distribution by Model",
        "TTR",
        output_dir,
        "token_type_ratio.png",
    )


def plot_yule_k(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features, "yule_k", "Yule's K Distribution by Model", "Yule's K", output_dir, "yule_k.png"
    )


def plot_guiraud(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features, "guiraud", "Guiraud's R Distribution by Model", "Guiraud's R", output_dir, "guiraud.png"
    )


def plot_honore(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features, "honore", "Honore's H Distribution by Model", "Honore's H", output_dir, "honore.png"
    )


def plot_brunet(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features, "brunet", "Brunet's W Distribution by Model", "Brunet's W", output_dir, "brunet.png"
    )


def plot_dugast(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features, "dugast", "Dugast's U Distribution by Model", "Dugast's U", output_dir, "dugast.png"
    )


def plot_maas_a2(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features, "maas_a2", "Maas's a² Distribution by Model", "Maas's a²", output_dir, "maas_a2.png"
    )


def plot_entropy(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features, "entropy", "Entropy Distribution by Model", "Entropy", output_dir, "entropy.png"
    )


def plot_repetition_rate(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "repetition_rate",
        "Repetition Rate Distribution by Model",
        "Repetition Rate",
        output_dir,
        "repetition_rate.png",
    )


def plot_hapax_ratio(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "hapax_ratio",
        "Hapax Ratio Distribution by Model",
        "Hapax Ratio",
        output_dir,
        "hapax_ratio.png",
    )


def plot_avg_word_freq(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "avg_word_freq",
        "Average Word Frequency Distribution by Model",
        "Avg Word Frequency",
        output_dir,
        "avg_word_freq.png",
    )


def plot_max_word_freq(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "max_word_freq",
        "Max Word Frequency Distribution by Model",
        "Max Word Frequency",
        output_dir,
        "max_word_freq.png",
    )


def plot_mtld(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "mtld",
        "Measure of Textual Lexical Diversity Distribution by Model",
        "Measure of Textual Lexical Diversity",
        output_dir,
        "mtld.png",
    )


def plot_lexical_density(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "lexical_density",
        "Lexical Density Distribution by Model",
        "Lexical Density",
        output_dir,
        "lexical_density.png",
    )


def plot_punctuation_density(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "punctuation_density",
        "Punctuation Density Distribution by Model",
        "Punctuation Density",
        output_dir,
        "punctuation_density.png",
    )


def plot_repeated_bigram_ratio(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "repeated_bigram_ratio",
        "Repeated Bigram Ratio Distribution by Model",
        "Repeated Bigram Ratio",
        output_dir,
        "repeated_bigram_ratio.png",
    )


def plot_repeated_trigram_ratio(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "repeated_trigram_ratio",
        "Repeated Trigram Ratio Distribution by Model",
        "Repeated Trigram Ratio",
        output_dir,
        "repeated_trigram_ratio.png",
    )


def plot_max_bigram_frequency(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "max_bigram_frequency",
        "Max Bigram Frequency Distribution by Model",
        "Max Bigram Frequency",
        output_dir,
        "max_bigram_frequency.png",
    )


def plot_sentiment_features(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    sentiment_features = ["sentiment_negative", "sentiment_neutral", "sentiment_positive", "sentiment_compound"]
    titles = ["Negative Sentiment", "Neutral Sentiment", "Positive Sentiment", "Compound Sentiment"]

    for ax, feature, title in zip(axes, sentiment_features, titles):
        sns.histplot(
            data=response_features[response_features["response_length"] > 0],
            x=feature,
            hue="model_key",
            kde=True,
            ax=ax,
            bins=30,
            alpha=0.5,
        )

        ax.set_title(title)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")

    plt.suptitle("Distribution of Sentiment Scores by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "sentiment_features.png")
    plt.close()


def plot_semantic_diversity(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "semantic_diversity",
        "Semantic Diversity Distribution by Model",
        "Semantic Diversity",
        output_dir,
        "semantic_diversity.png",
    )


def plot_person_pronouns_features(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    features = ["first_person_pronoun_count", "first_person_pronoun_density"]
    titles = ["First Person Pronoun Count", "First Person Pronoun Density"]

    for ax, feature, title in zip(axes, features, titles):
        sns.histplot(
            data=response_features[response_features["response_length"] > 0],
            x=feature,
            hue="model_key",
            kde=True,
            bins=30,
            alpha=0.5,
            ax=ax,
        )

        ax.set_title(f"{title} Distribution by Model")
        ax.set_xlabel(title)
        ax.set_ylabel("Count")

    plt.suptitle("Distribution of First Person Pronoun Features by Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "person_pronoun_features.png")
    plt.close()


def plot_sentence_coherence_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "sentence_coherence",
        "Sentence Coherence Distribution by Model",
        "Sentence Coherence",
        output_dir,
        "sentence_coherence.png",
    )


def plot_embedding_variance_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "embedding_variance",
        "Embedding Variance Distribution by Model",
        "Embedding Variance",
        output_dir,
        "embedding_variance.png",
    )


def plot_flesch_reading_ease_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "flesch_reading_ease",
        "Flesch Reading Ease Distribution by Model",
        "Flesch Reading Ease",
        output_dir,
        "flesch_reading_ease.png",
    )


def plot_flesch_kincaid_grade_level_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    plot_single_feature(
        response_features,
        "flesch_kincaid_grade",
        "Flesch-Kincaid Grade Distribution by Model",
        "Flesch-Kincaid Grade",
        output_dir,
        "flesch_kincaid_grade.png",
    )


def plot_correlation_heatmap(response_features: pd.DataFrame, output_dir: Path) -> None:
    numeric_cols = response_features.select_dtypes(include="number").columns
    corr = response_features[numeric_cols].corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 7})
    plt.title("Correlation Heatmap for Response Features")

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()
