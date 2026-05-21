from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

from sklearn.pipeline import Pipeline


def estimator_from_model(model: Any) -> Any | Pipeline:
    return model.named_steps["model"] if isinstance(model, Pipeline) else model


def built_in_importance(target: str, model: Any, feature_columns: list[str]) -> pd.DataFrame:
    estimator = estimator_from_model(model)
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        kind = "feature_importances"
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).mean(axis=0)
        kind = "absolute_coefficients"
    else:
        values = np.zeros(len(feature_columns))
        kind = "not_available"
    return pd.DataFrame(
        {
            "target": target,
            "feature": feature_columns,
            "importance": values,
            "importance_kind": kind,
        }
    ).sort_values("importance", ascending=False)


def permutation_importance_frame(
    target: str,
    model: Any,
    best_by_target: dict[str, Any],
    feature_splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
    random_state: int,
) -> pd.DataFrame:

    bundle = best_by_target[target]
    result = permutation_importance(
        model,
        feature_splits["val"],
        bundle["encoded_targets"]["val"],
        scoring="f1_macro",
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1,
    )

    return pd.DataFrame(
        {
            "target": target,
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def shap_values_to_importance(
    target: str, values: np.ndarray, class_names: list[str], feature_columns: list[str]
) -> pd.DataFrame:
    records = []

    if values.ndim == 2:
        values = values[:, :, None]

    for class_index, class_name in enumerate(class_names[: values.shape[2]]):
        mean_abs = np.abs(values[:, :, class_index]).mean(axis=0)

        for rank, feature_index in enumerate(np.argsort(mean_abs)[::-1], start=1):
            records.append(
                {
                    "target": target,
                    "class_name": class_name,
                    "feature": feature_columns[feature_index],
                    "mean_abs_shap": mean_abs[feature_index],
                    "rank": rank,
                    "importance_kind": "shap",
                    "note": "",
                }
            )

    overall = np.abs(values).mean(axis=(0, 2))

    for rank, feature_index in enumerate(np.argsort(overall)[::-1], start=1):
        records.append(
            {
                "target": target,
                "class_name": "__overall__",
                "feature": feature_columns[feature_index],
                "mean_abs_shap": overall[feature_index],
                "rank": rank,
                "importance_kind": "shap",
                "note": "",
            }
        )

    return pd.DataFrame(records)


def fallback_shap_frame(target: str, model: Any, reason: str, feature_columns: list[str]) -> pd.DataFrame:
    fallback = built_in_importance(target, model, feature_columns)
    fallback = fallback.rename(columns={"importance": "mean_abs_shap"})
    fallback["class_name"] = "__overall__"
    fallback["rank"] = range(1, len(fallback) + 1)
    fallback["importance_kind"] = "fallback_importance"
    fallback["note"] = reason

    return fallback[["target", "class_name", "feature", "mean_abs_shap", "rank", "importance_kind", "note"]]


def explain_with_shap(
    target: str, model: Any, best_by_target, feature_splits, feature_columns, random_state
) -> pd.DataFrame:

    class_names = best_by_target[target]["class_names"]
    background = feature_splits["train"].sample(n=min(100, len(feature_splits["train"])), random_state=random_state)
    sample = feature_splits["val"].sample(n=min(80, len(feature_splits["val"])), random_state=random_state)

    try:
        import shap

        explainer = shap.Explainer(model.predict_proba, background, algorithm="permutation")
        shap_values = explainer(sample, max_evals=2 * len(feature_columns) + 1)

        return shap_values_to_importance(target, np.asarray(shap_values.values), class_names, feature_columns)
    except Exception as exc:
        return fallback_shap_frame(target, model, str(exc), feature_columns)


def plot_top_importance(df: pd.DataFrame, value_column: str, title: str, output_path: Path, top_n: int = 20) -> None:
    top = df.sort_values(value_column, ascending=False).head(top_n).sort_values(value_column)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top[value_column])
    ax.set_title(title)
    ax.set_xlabel(value_column)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def calculate_outputs_importances(
    random_state: int,
    xai_dir: Path,
    best_by_target: dict[Any, Any],
    feature_columns: list[str],
    feature_splits: dict[str, pd.DataFrame],
) -> tuple[list[Any], list[Any], list[Any]]:

    importance_outputs = []
    permutation_outputs = []
    shap_outputs = []

    for target, bundle in best_by_target.items():
        model = bundle["model"]

        built_in = built_in_importance(target, model, feature_columns)
        built_in.to_csv(xai_dir / "importance" / f"{target}_built_in_importance.csv", index=False)
        plot_top_importance(
            built_in,
            "importance",
            f"Built-in importance for {target}",
            xai_dir / "importance" / f"{target}_built_in_importance.png",
        )
        importance_outputs.append(built_in.assign(method="built_in"))

        permutation_df = permutation_importance_frame(
            target, model, best_by_target, feature_splits, feature_columns, random_state
        )
        permutation_df.to_csv(xai_dir / "importance" / f"{target}_permutation_importance.csv", index=False)
        plot_top_importance(
            permutation_df,
            "importance_mean",
            f"Permutation importance for {target}",
            xai_dir / "importance" / f"{target}_permutation_importance.png",
        )
        permutation_outputs.append(permutation_df.assign(method="permutation"))

        shap_df = explain_with_shap(target, model, best_by_target, feature_splits, feature_columns, random_state)
        shap_df.to_csv(xai_dir / "shap" / f"{target}_shap_importance.csv", index=False)
        overall_shap = shap_df[shap_df["class_name"] == "__overall__"]
        plot_top_importance(
            overall_shap.rename(columns={"mean_abs_shap": "importance"}),
            "importance",
            f"SHAP importance for {target}",
            xai_dir / "shap" / f"{target}_shap_importance.png",
        )
        shap_outputs.append(shap_df)
    return importance_outputs, permutation_outputs, shap_outputs


def calculate_feature_group_importance(
    target_columns: tuple[str, ...],
    xai_dir: Path,
    importance_outputs: list[Any],
    permutation_outputs: list[Any],
    shap_outputs: list[Any],
) -> pd.DataFrame:

    group_importance_parts = []

    built_in_all = pd.concat(importance_outputs, ignore_index=True)
    built_in_all["feature_group"] = built_in_all["feature"].map(feature_group)
    built_in_groups = built_in_all.groupby(["target", "method", "feature_group"], as_index=False)["importance"].sum()
    group_importance_parts.append(normalize_group_importance(built_in_groups, "importance"))

    permutation_all = pd.concat(permutation_outputs, ignore_index=True)
    permutation_all["feature_group"] = permutation_all["feature"].map(feature_group)
    permutation_groups = (
        permutation_all.groupby(["target", "method", "feature_group"], as_index=False)["importance_mean"]
        .sum()
        .rename(columns={"importance_mean": "importance"})
    )
    group_importance_parts.append(normalize_group_importance(permutation_groups, "importance"))

    shap_all = pd.concat(shap_outputs, ignore_index=True)
    shap_overall = shap_all[shap_all["class_name"] == "__overall__"].copy()
    shap_overall["method"] = "shap"
    shap_overall["feature_group"] = shap_overall["feature"].map(feature_group)
    shap_groups = (
        shap_overall.groupby(["target", "method", "feature_group"], as_index=False)["mean_abs_shap"]
        .sum()
        .rename(columns={"mean_abs_shap": "importance"})
    )
    group_importance_parts.append(normalize_group_importance(shap_groups, "importance"))

    feature_group_importance = pd.concat(group_importance_parts, ignore_index=True)
    feature_group_importance.to_csv(xai_dir / "importance" / "feature_group_importance.csv", index=False)

    for target in target_columns:
        target_df = feature_group_importance[
            (feature_group_importance["target"] == target) & (feature_group_importance["method"] == "shap")
        ].sort_values("importance_share")
        if not target_df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(target_df["feature_group"], target_df["importance_share"])
            ax.set_title(f"Feature group importance for {target} (SHAP)")
            ax.set_xlabel("Share of importance")
            fig.tight_layout()
            fig.savefig(xai_dir / "importance" / f"{target}_feature_group_importance.png")
            plt.close(fig)
    return feature_group_importance


def feature_group(feature: str) -> str:
    if feature in {"source_avg_logprob", "source_sum_logprob", "source_perplexity"}:
        return "generation_confidence"
    if feature in {
        "source_response_length",
        "source_generated_tokens",
        "text_char_count",
        "text_word_count",
        "text_sentence_count",
        "text_paragraph_count",
        "text_newline_count",
        "text_avg_sentence_chars",
        "text_avg_sentence_words",
        "text_avg_word_length",
    }:
        return "length_and_structure"
    if feature in {
        "text_unique_word_count",
        "text_type_token_ratio",
        "text_hapax_ratio",
        "text_entropy",
        "text_repetition_rate",
        "text_avg_word_frequency",
        "text_max_word_frequency",
        "text_repeated_bigram_ratio",
        "text_repeated_trigram_ratio",
    }:
        return "lexical_diversity_and_repetition"
    if feature in {
        "text_punctuation_count",
        "text_punctuation_density",
        "text_comma_density",
        "text_colon_density",
        "text_semicolon_density",
        "text_question_mark_density",
        "text_exclamation_mark_density",
        "text_digit_density",
        "text_uppercase_word_ratio",
        "text_list_marker_count",
        "text_markdown_bold_count",
        "text_markdown_heading_count",
        "text_code_fence_count",
    }:
        return "formatting_and_punctuation"
    if feature in {
        "text_first_person_pronoun_count",
        "text_first_person_pronoun_density",
        "text_second_person_pronoun_count",
        "text_second_person_pronoun_density",
        "text_hedge_word_count",
        "text_hedge_word_density",
        "text_negation_word_count",
        "text_negation_word_density",
    }:
        return "stance_pronouns_and_uncertainty"
    return "other"


def normalize_group_importance(group_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    group_df = group_df.copy()
    group_df[value_column] = group_df[value_column].clip(lower=0)
    totals = group_df.groupby(["target", "method"])[value_column].transform("sum")
    group_df["importance_share"] = np.where(totals > 0, group_df[value_column] / totals, 0.0)
    return group_df
