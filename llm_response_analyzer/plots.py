import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


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


def plot_token_count_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="token_count",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Token Count Distribution by Model")
    ax.set_xlabel("Token Count")

    plt.tight_layout()
    plt.savefig(output_dir / "token_count_feature.png")


def plot_type_token_ratio(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="type_token_ratio",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Type-Token Ratio (TTR) Distribution by Model")
    ax.set_xlabel("TTR")

    plt.tight_layout()
    plt.savefig(output_dir / "token_type_ratio.png")


def plot_yule_k(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="yule_k",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Yule's K Distribution by Model")
    ax.set_xlabel("Yule's K")

    plt.tight_layout()
    plt.savefig(output_dir / "yule_k.png")


def plot_guiraud(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="guiraud",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Guiraud's R Distribution by Model")
    ax.set_xlabel("Guiraud's R")

    plt.tight_layout()
    plt.savefig(output_dir / "guiraud.png")


def plot_honore(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="honore",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Honore's H Distribution by Model")
    ax.set_xlabel("Honore's H")

    plt.tight_layout()
    plt.savefig(output_dir / "honore.png")


def plot_brunet(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="brunet",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Brunet's W Distribution by Model")
    ax.set_xlabel("Brunet's W")

    plt.tight_layout()
    plt.savefig(output_dir / "brunet.png")


def plot_dugast(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="dugast",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Dugast's U Distribution by Model")
    ax.set_xlabel("Dugast's U")

    plt.tight_layout()
    plt.savefig(output_dir / "dugast.png")


def plot_maas_a2(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="maas_a2",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Maas's a² Distribution by Model")
    ax.set_xlabel("Maas's a²")

    plt.tight_layout()
    plt.savefig(output_dir / "maas_a2.png")


def plot_entropy(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="entropy",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Entropy Distribution by Model")
    ax.set_xlabel("Entropy")

    plt.tight_layout()
    plt.savefig(output_dir / "entropy.png")


def plot_repetition_rate(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="repetition_rate",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Repetition Rate Distribution by Model")
    ax.set_xlabel("Repetition Rate")

    plt.tight_layout()
    plt.savefig(output_dir / "repetition_rate.png")


def plot_hapax_ratio(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="hapax_ratio",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Hapax Ratio Distribution by Model")
    ax.set_xlabel("Hapax Ratio")

    plt.tight_layout()
    plt.savefig(output_dir / "hapax_ratio.png")


def plot_avg_word_freq(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="avg_word_freq",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Average Word Frequency Distribution by Model")
    ax.set_xlabel("Avg Word Frequency")

    plt.tight_layout()
    plt.savefig(output_dir / "avg_word_freq.png")


def plot_max_word_freq(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="max_word_freq",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Max Word Frequency Distribution by Model")
    ax.set_xlabel("Max Word Frequency")

    plt.tight_layout()
    plt.savefig(output_dir / "max_word_freq.png")


def plot_mtld(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="mtld",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Measure of Textual Lexical Diversity Distribution by Model")
    ax.set_xlabel("Measure of Textual Lexical Diversity")

    plt.tight_layout()
    plt.savefig(output_dir / "mtld.png")


def plot_lexical_density(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="lexical_density",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Lexical Density Distribution by Model")
    ax.set_xlabel("Lexical Density")

    plt.tight_layout()
    plt.savefig(output_dir / "lexical_density.png")


def plot_punctuation_density(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="punctuation_density",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Punctuation Density Distribution by Model")
    ax.set_xlabel("Punctuation Density")

    plt.tight_layout()
    plt.savefig(output_dir / "punctuation_density.png")


def plot_repeated_bigram_ratio(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="repeated_bigram_ratio",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Repeated Bigram Ratio Distribution by Model")
    ax.set_xlabel("Repeated Bigram Ratio")

    plt.tight_layout()
    plt.savefig(output_dir / "repeated_bigram_ratio.png")


def plot_repeated_trigram_ratio(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="repeated_trigram_ratio",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Repeated Trigram Ratio Distribution by Model")
    ax.set_xlabel("Repeated Trigram Ratio")

    plt.tight_layout()
    plt.savefig(output_dir / "repeated_trigram_ratio.png")


def plot_max_bigram_frequency(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="max_bigram_frequency",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Max Bigram Frequency Distribution by Model")
    ax.set_xlabel("Max Bigram Frequency")

    plt.tight_layout()
    plt.savefig(output_dir / "max_bigram_frequency.png")


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


def plot_semantic_diversity(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="semantic_diversity",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Semantic Diversity Distribution by Model")
    ax.set_xlabel("Semantic Diversity")

    plt.tight_layout()
    plt.savefig(output_dir / "semantic_diversity.png")


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


def plot_sentence_coherence_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="sentence_coherence",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Sentence Coherence Distribution by Model")
    ax.set_xlabel("Sentence Coherence")

    plt.tight_layout()
    plt.savefig(output_dir / "sentence_coherence.png")


def plot_embedding_variance_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="embedding_variance",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Embedding Variance Distribution by Model")
    ax.set_xlabel("Embedding Variance")

    plt.tight_layout()
    plt.savefig(output_dir / "embedding_variance.png")


def plot_flesch_reading_ease_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="flesch_reading_ease",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Flesch Reading Ease Distribution by Model")
    ax.set_xlabel("Flesch Reading Ease")

    plt.tight_layout()
    plt.savefig(output_dir / "flesch_reading_ease.png")


def plot_flesch_kincaid_grade_level_feature(response_features: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.histplot(
        data=response_features[response_features["response_length"] > 0],
        x="flesch_kincaid_grade",
        hue="model_key",
        kde=True,
        bins=30,
        alpha=0.5,
        ax=ax,
    )

    ax.set_title("Flesch-Kincaid Grade Distribution by Model")
    ax.set_xlabel("Flesch-Kincaid Grade")

    plt.tight_layout()
    plt.savefig(output_dir / "flesch_kincaid_grade.png")


def plot_correlation_heatmap(response_features: pd.DataFrame, output_dir: Path) -> None:
    numeric_cols = response_features.select_dtypes(include="number").columns
    corr = response_features[numeric_cols].corr()

    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 7})
    plt.title("Correlation Heatmap for Response Features")

    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
