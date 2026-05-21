from __future__ import annotations

import json
import math
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.feature_selection import mutual_info_classif

from llm_behavior_xai.config import (
    LLM_RESULTS_TRAIN_PROMPTS,
    LLM_RESULTS_VAL_PROMPTS,
    LLM_RESULTS_TEST_PROMPTS,
    XAI_MODELS_DIR,
    XAI_REPORTS_DIR,
    STYLE_PROFILES_REPORTS_DIR,
)
from llm_behavior_xai.model_training_and_profiles.build_numeric_training_features import build_feature_splits, \
    build_feature_frame


def eta_squared_by_group(values: pd.Series, labels: pd.Series) -> float:
    overall_mean = values.mean()
    total_variation = ((values - overall_mean) ** 2).sum()
    if total_variation == 0 or pd.isna(total_variation):
        return 0.0
    between_variation = 0.0
    for _, group_values in values.groupby(labels):
        between_variation += len(group_values) * (group_values.mean() - overall_mean) ** 2
    return float(between_variation / total_variation)


def load_final_splits(split_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {split: pd.read_csv(path) for split, path in split_paths.items()}


def main():
    FINAL_SPLIT_PATHS = {
        "train": LLM_RESULTS_TRAIN_PROMPTS,
        "val": LLM_RESULTS_VAL_PROMPTS,
        "test": LLM_RESULTS_TEST_PROMPTS,
    }

    TARGET_COLUMNS = ("model_key", "language")
    SELECTION_METRIC = "macro_f1"
    RANDOM_STATE = 42

    MODELS_DIR = XAI_MODELS_DIR
    XAI_DIR = XAI_REPORTS_DIR
    PROFILES_DIR = STYLE_PROFILES_REPORTS_DIR

    for directory in (
        MODELS_DIR,
        XAI_DIR,
        XAI_DIR / "predictions",
        XAI_DIR / "confusion_matrices",
        XAI_DIR / "shap",
        XAI_DIR / "features",
        XAI_DIR / "importance",
        XAI_DIR / "surrogate_trees",
        PROFILES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    final_splits = load_final_splits(FINAL_SPLIT_PATHS)

    split_overview = []
    for split, df in final_splits.items():
        split_overview.append(
            {
                "split": split,
                "rows": len(df),
                "models": df["model_key"].nunique(),
                "languages": df["language"].nunique(),
                "paraphrase_values": df["is_paraphrase"].nunique(),
            }
        )

    print()
    print("-- Dataset split overview: -----------------------------------------------------------")
    print(pd.DataFrame(split_overview))

    SOURCE_NUMERIC_COLUMNS = (
        "response_length",
        "generated_tokens",
        "perplexity",
        "avg_logprob",
        "sum_logprob",
    )

    PREDICTION_CONTEXT_COLUMNS = (
        "run_id",
        "prompt_id",
        "category",
        "prompt_column",
        "language",
        "is_paraphrase",
        "model_key",
        "provider",
        "prompt_text",
        "response",
    )

    WORD_RE = re.compile(r"\b[\w']+\b", flags=re.UNICODE)
    SENTENCE_RE = re.compile(r"[^.!?]+[.!?]*", flags=re.UNICODE)
    LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+")
    HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")

    FIRST_PERSON_PRONOUNS = {"i", "me", "my", "mine", "we", "us", "our", "ours", "ja", "mnie", "nas", "nam"}
    SECOND_PERSON_PRONOUNS = {"you", "your", "yours", "ty", "ci", "ciebie", "tobie", "wy", "was", "wam"}
    HEDGE_WORDS = {"maybe", "perhaps", "probably", "possibly", "likely", "might", "could", "may", "moze", "chyba"}
    NEGATION_WORDS = {"no", "not", "never", "none", "cannot", "can't", "nie", "nigdy"}

    feature_splits, feature_columns, fill_values = build_feature_splits(
        final_splits,
        SOURCE_NUMERIC_COLUMNS,
        WORD_RE,
        SENTENCE_RE,
        LIST_MARKER_RE,
        HEADING_RE,
        FIRST_PERSON_PRONOUNS,
        SECOND_PERSON_PRONOUNS,
        HEDGE_WORDS,
        NEGATION_WORDS,
    )

    for split, features in feature_splits.items():
        features.to_csv(XAI_DIR / "features" / f"{split}_features.csv", index=False)

    feature_list = pd.DataFrame({"feature": feature_columns})
    feature_list.to_csv(XAI_DIR / "features" / "feature_list.csv", index=False)
    with (XAI_DIR / "features" / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_columns": feature_columns,
                "fill_values": fill_values.to_dict(),
                "source": "data/processed/final",
                "targets": TARGET_COLUMNS,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"-- Feature count: {len(feature_columns)} -----------------------------------------------------------------")
    print(feature_list)

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
    feature_meanings.to_csv(XAI_DIR / "features" / "feature_descriptions.csv", index=False)

    print()
    print("-- Feature meanings: -----------------------------------------------------------------")
    print(feature_meanings)

    train_model_labels = final_splits["train"]["model_key"].astype(str)
    try:
        mutual_information = mutual_info_classif(
            feature_splits["train"],
            train_model_labels,
            discrete_features=False,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:
        print(f"Mutual information check failed: {exc}")
        mutual_information = np.zeros(len(feature_columns))

    feature_signal_rows = []
    for feature, mi_score in zip(feature_columns, mutual_information):
        values = feature_splits["train"][feature]
        overall_mean = values.mean()
        overall_std = values.std(ddof=0)
        per_model_means = values.groupby(train_model_labels).mean().to_dict()
        if overall_std == 0 or pd.isna(overall_std):
            max_abs_model_mean_z = 0.0
        else:
            max_abs_model_mean_z = max(
                abs((mean_value - overall_mean) / overall_std) for mean_value in per_model_means.values()
            )
        eta_squared = eta_squared_by_group(values, train_model_labels)
        if eta_squared >= 0.80 or max_abs_model_mean_z >= 2.5:
            flag = "very_strong_model_signal"
        elif eta_squared >= 0.50 or max_abs_model_mean_z >= 1.5:
            flag = "strong_model_signal"
        else:
            flag = ""
        feature_signal_rows.append(
            {
                "feature": feature,
                "eta_squared_model_key": eta_squared,
                "mutual_information_model_key": float(mi_score),
                "max_abs_model_mean_z": float(max_abs_model_mean_z),
                "unique_values": int(values.nunique()),
                "per_model_means": json.dumps({str(key): float(value) for key, value in per_model_means.items()}),
                "review_flag": flag,
            }
        )

    feature_model_signal_check = pd.DataFrame(feature_signal_rows).sort_values(
        ["eta_squared_model_key", "mutual_information_model_key"],
        ascending=False,
    )
    feature_model_signal_check.to_csv(XAI_DIR / "features" / "feature_model_signal_check.csv", index=False)

    print()
    print("-- Model's features signal check: ----------------------------------------------------")
    print(feature_model_signal_check.head(25))

    raw_feature_frames = []
    raw_context_frames = []

    for split, source_df in final_splits.items():
        raw_features = build_feature_frame(source_df, SOURCE_NUMERIC_COLUMNS, WORD_RE, SENTENCE_RE, LIST_MARKER_RE, HEADING_RE, FIRST_PERSON_PRONOUNS, SECOND_PERSON_PRONOUNS, HEDGE_WORDS, NEGATION_WORDS).reindex(columns=feature_columns)
        raw_features["split"] = split
        raw_feature_frames.append(raw_features)

        context = source_df[["model_key"]].copy()
        context["split"] = split
        raw_context_frames.append(context)

    raw_all_features = pd.concat(raw_feature_frames, ignore_index=True)
    raw_all_context = pd.concat(raw_context_frames, ignore_index=True)
    all_model_labels = raw_all_context["model_key"].astype(str)

    zero_nan_rows = []
    for feature in feature_columns:
        values = raw_all_features[feature]
        for model_key in sorted(all_model_labels.unique()):
            model_mask = all_model_labels == model_key
            model_values = values[model_mask]
            other_values = values[~model_mask]

            model_nan_ratio = float(model_values.isna().mean())
            other_nan_ratio = float(other_values.isna().mean())
            model_zero_ratio = float((model_values.fillna(np.nan) == 0).mean())
            other_zero_ratio = float((other_values.fillna(np.nan) == 0).mean())
            model_zero_or_nan_ratio = float((model_values.isna() | (model_values == 0)).mean())
            other_zero_or_nan_ratio = float((other_values.isna() | (other_values == 0)).mean())

            model_normal_values = model_values.dropna()
            model_normal_values = model_normal_values[model_normal_values != 0]
            other_normal_values = other_values.dropna()
            other_normal_values = other_normal_values[other_normal_values != 0]

            difference = model_zero_or_nan_ratio - other_zero_or_nan_ratio
            if model_zero_or_nan_ratio >= 0.80 and other_zero_or_nan_ratio <= 0.30:
                flag = "model_mostly_zero_or_nan"
            elif difference >= 0.50:
                flag = "model_more_zero_or_nan"
            elif difference <= -0.50:
                flag = "others_more_zero_or_nan"
            else:
                flag = ""

            zero_nan_rows.append(
                {
                    "feature": feature,
                    "model_key": model_key,
                    "model_zero_or_nan_ratio": model_zero_or_nan_ratio,
                    "other_zero_or_nan_ratio": other_zero_or_nan_ratio,
                    "difference": difference,
                    "model_nan_ratio": model_nan_ratio,
                    "other_nan_ratio": other_nan_ratio,
                    "model_zero_ratio": model_zero_ratio,
                    "other_zero_ratio": other_zero_ratio,
                    "model_nonzero_nonnull_mean": float(model_normal_values.mean())
                    if len(model_normal_values)
                    else np.nan,
                    "other_nonzero_nonnull_mean": float(other_normal_values.mean())
                    if len(other_normal_values)
                    else np.nan,
                    "review_flag": flag,
                }
            )

    feature_zero_nan_check = pd.DataFrame(zero_nan_rows).sort_values(
        ["review_flag", "difference"],
        ascending=[False, False],
    )
    feature_zero_nan_check.to_csv(XAI_DIR / "features" / "feature_zero_nan_by_model_check.csv", index=False)

    excluded_zero_nan_features = sorted(
        feature_zero_nan_check.loc[feature_zero_nan_check["review_flag"] != "", "feature"].unique()
    )
    pd.DataFrame({"excluded_feature": excluded_zero_nan_features}).to_csv(
        XAI_DIR / "features" / "excluded_zero_nan_features.csv",
        index=False,
    )

    if excluded_zero_nan_features:
        print("Excluding features with model-specific zero/NaN patterns:")
        for feature in excluded_zero_nan_features:
            print(f"- {feature}")
    else:
        print("No model-specific zero/NaN features were flagged for exclusion.")

    # From this point onward, all training uses only the filtered feature set.
    feature_columns = [feature for feature in feature_columns if feature not in excluded_zero_nan_features]
    feature_splits = {split: features[feature_columns].copy() for split, features in feature_splits.items()}
    feature_list = pd.DataFrame({"feature": feature_columns})
    feature_list.to_csv(XAI_DIR / "features" / "feature_list.csv", index=False)

    if "feature_meanings" in globals():
        feature_meanings = feature_meanings.copy()
        feature_meanings["used_for_training"] = feature_meanings["feature"].isin(feature_columns)
        feature_meanings["excluded_reason"] = np.where(
            feature_meanings["feature"].isin(excluded_zero_nan_features),
            "model-specific zero/NaN pattern",
            "",
        )
        feature_meanings.to_csv(XAI_DIR / "features" / "feature_descriptions.csv", index=False)

    with (XAI_DIR / "features" / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_columns": feature_columns,
                "excluded_zero_nan_features": excluded_zero_nan_features,
                "source": "data/processed/final",
                "targets": TARGET_COLUMNS,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"-- Remaining training feature count: {len(feature_columns)} ----------------------------------------------")
    print(feature_zero_nan_check[feature_zero_nan_check["review_flag"] != ""].head(30))

    validation_rows = []
    best_by_target = {}

    for target in TARGET_COLUMNS:
        encoder, encoded_targets = encode_target(target, final_splits)
        class_names = list(encoder.classes_)
        target_candidates = []

        for model_name, (base_estimator, param_grid) in build_model_candidates(RANDOM_STATE).items():
            for params in ParameterGrid(param_grid):
                model = fit_with_params(base_estimator, params)
                model.fit(feature_splits["train"], encoded_targets["train"])
                val_pred = model.predict(feature_splits["val"])
                row = metric_row(target, "val", model_name, params, encoded_targets["val"], val_pred, class_names)
                validation_rows.append(row)
                target_candidates.append((row[SELECTION_METRIC], row["balanced_accuracy"], model_name, params, model))

        target_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, best_balanced_acc, best_model_name, best_params, best_model = target_candidates[0]
        best_by_target[target] = {
            "encoder": encoder,
            "encoded_targets": encoded_targets,
            "class_names": class_names,
            "model_name": best_model_name,
            "params": best_params,
            "model": best_model,
            "val_score": best_score,
            "val_balanced_accuracy": best_balanced_acc,
        }

    validation_metrics = pd.DataFrame(validation_rows).sort_values(
        ["target", SELECTION_METRIC, "balanced_accuracy"],
        ascending=[True, False, False],
    )
    validation_metrics.to_csv(XAI_DIR / "validation_tuning_metrics.csv", index=False)

    print()
    print("-- Validation metrics: ---------------------------------------------------------------")
    print(validation_metrics.head(20))

    selected_rows = []
    for target, bundle in best_by_target.items():
        selected_rows.append(
            {
                "target": target,
                "selected_model": bundle["model_name"],
                "selected_params": json.dumps(bundle["params"], sort_keys=True),
                "validation_macro_f1": bundle["val_score"],
                "validation_balanced_accuracy": bundle["val_balanced_accuracy"],
            }
        )

    selected_models = pd.DataFrame(selected_rows)
    selected_models.to_csv(XAI_DIR / "selected_models.csv", index=False)
    
    print()
    print("-- Selected models: ------------------------------------------------------------------")
    print(selected_models)

    all_metrics = []
    for target, bundle in best_by_target.items():
        model = bundle["model"]
        class_names = bundle["class_names"]
        for split in ("train", "val", "test"):
            y_true = bundle["encoded_targets"][split]
            y_pred = model.predict(feature_splits[split])
            probabilities = model.predict_proba(feature_splits[split])
            all_metrics.append(
                metric_row(target, split, bundle["model_name"], bundle["params"], y_true, y_pred, class_names)
            )
            save_predictions(target, split, final_splits[split], y_true, y_pred, probabilities, class_names, best_by_target, XAI_DIR, PREDICTION_CONTEXT_COLUMNS)
            save_confusion_matrix(target, split, y_true, y_pred, class_names, XAI_DIR)

        joblib.dump(model, MODELS_DIR / f"{target}_best_model.joblib")
        with (MODELS_DIR / f"{target}_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "target": target,
                    "model_name": bundle["model_name"],
                    "params": bundle["params"],
                    "class_names": class_names,
                    "feature_columns": feature_columns,
                    "selection_metric": SELECTION_METRIC,
                    "selection_split": "val",
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(XAI_DIR / "all_metrics.csv", index=False)

    print()
    print("-- Final test metrics for selected models: -------------------------------------------")
    print(metrics_df[metrics_df["split"].isin(["val", "test"])][["target", "split", "model_name", "accuracy", "macro_f1", "balanced_accuracy", "params"]])

    importance_outputs = []
    permutation_outputs = []
    shap_outputs = []

    for target, bundle in best_by_target.items():
        model = bundle["model"]

        built_in = built_in_importance(target, model, feature_columns)
        built_in.to_csv(XAI_DIR / "importance" / f"{target}_built_in_importance.csv", index=False)
        plot_top_importance(
            built_in,
            "importance",
            f"Built-in importance for {target}",
            XAI_DIR / "importance" / f"{target}_built_in_importance.png",
        )
        importance_outputs.append(built_in.assign(method="built_in"))

        permutation_df = permutation_importance_frame(target, model, best_by_target, feature_splits, feature_columns, RANDOM_STATE)
        permutation_df.to_csv(XAI_DIR / "importance" / f"{target}_permutation_importance.csv", index=False)
        plot_top_importance(
            permutation_df,
            "importance_mean",
            f"Permutation importance for {target}",
            XAI_DIR / "importance" / f"{target}_permutation_importance.png",
        )
        permutation_outputs.append(permutation_df.assign(method="permutation"))

        shap_df = explain_with_shap(target, model, best_by_target, feature_splits, feature_columns, RANDOM_STATE)
        shap_df.to_csv(XAI_DIR / "shap" / f"{target}_shap_importance.csv", index=False)
        overall_shap = shap_df[shap_df["class_name"] == "__overall__"]
        plot_top_importance(
            overall_shap.rename(columns={"mean_abs_shap": "importance"}),
            "importance",
            f"SHAP importance for {target}",
            XAI_DIR / "shap" / f"{target}_shap_importance.png",
        )
        shap_outputs.append(shap_df)

    print()
    print("-- Importance outputs: ---------------------------------------------------------------")
    print(pd.concat(importance_outputs).head(20))

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
    feature_group_importance.to_csv(XAI_DIR / "importance" / "feature_group_importance.csv", index=False)

    for target in TARGET_COLUMNS:
        target_df = feature_group_importance[
            (feature_group_importance["target"] == target) & (feature_group_importance["method"] == "shap")
        ].sort_values("importance_share")
        if not target_df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(target_df["feature_group"], target_df["importance_share"])
            ax.set_title(f"Feature group importance for {target} (SHAP)")
            ax.set_xlabel("Share of importance")
            fig.tight_layout()
            fig.savefig(XAI_DIR / "importance" / f"{target}_feature_group_importance.png")
            plt.close(fig)

    print()
    print("-- Feature group importance: ---------------------------------------------------------")
    print(feature_group_importance.sort_values(["target", "method", "importance_share"], ascending=[True, True, False]))

    surrogate_metrics = pd.concat(
        [train_surrogate_tree(target) for target in TARGET_COLUMNS],
        ignore_index=True,
    )
    surrogate_metrics.to_csv(XAI_DIR / "surrogate_trees" / "surrogate_tree_metrics.csv", index=False)

    print()
    print("-- Surrogate tree metrics: -----------------------------------------------------------")
    print(surrogate_metrics)

    profile_data = combine_profile_data(final_splits, feature_splits)
    feature_summary = build_model_feature_summary(profile_data, feature_columns)
    effect_sizes = build_model_effect_sizes(profile_data, feature_columns)
    top_features = effect_sizes.groupby("model_key", group_keys=False).head(8).reset_index(drop=True)
    sensitivity = build_language_paraphrase_sensitivity(profile_data, feature_columns)

    model_key_shap_path = XAI_DIR / "shap" / "model_key_shap_importance.csv"
    shap_summary = pd.read_csv(model_key_shap_path) if model_key_shap_path.exists() else pd.DataFrame()
    if not shap_summary.empty:
        shap_summary = shap_summary[shap_summary["class_name"] != "__overall__"].copy()

    feature_summary.to_csv(PROFILES_DIR / "model_feature_summary.csv", index=False)
    effect_sizes.to_csv(PROFILES_DIR / "model_effect_sizes.csv", index=False)
    top_features.to_csv(PROFILES_DIR / "model_top_features.csv", index=False)
    sensitivity.to_csv(PROFILES_DIR / "language_paraphrase_sensitivity.csv", index=False)
    write_profile_markdown(top_features, shap_summary, PROFILES_DIR)

    print()
    print("-- Profiles: ------------------------------------------------------------------------------")
    print(top_features)


def combine_profile_data(final_splits, feature_splits) -> pd.DataFrame:
    records = []
    context_columns = ["prompt_id", "category", "language", "is_paraphrase", "model_key", "provider"]
    for split, source_df in final_splits.items():
        available_context = [column for column in context_columns if column in source_df.columns]
        split_df = pd.concat(
            [source_df[available_context].reset_index(drop=True), feature_splits[split].reset_index(drop=True)],
            axis=1,
        )
        split_df["split"] = split
        records.append(split_df)
    return pd.concat(records, ignore_index=True)


def build_model_feature_summary(profile_data: pd.DataFrame, feature_columns) -> pd.DataFrame:
    summary = profile_data.groupby("model_key")[feature_columns].agg(["mean", "median", "std"]).reset_index()
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns.to_flat_index()]
    return summary


def build_model_effect_sizes(profile_data: pd.DataFrame, feature_columns) -> pd.DataFrame:
    records = []
    for model_key in sorted(profile_data["model_key"].dropna().unique()):
        model_rows = profile_data[profile_data["model_key"] == model_key]
        other_rows = profile_data[profile_data["model_key"] != model_key]
        for feature in feature_columns:
            model_mean = float(model_rows[feature].mean())
            other_mean = float(other_rows[feature].mean())
            model_std = float(model_rows[feature].std(ddof=1))
            other_std = float(other_rows[feature].std(ddof=1))
            std_values = [value**2 for value in (model_std, other_std) if not np.isnan(value)]
            pooled_std = float(np.sqrt(np.mean(std_values))) if std_values else 0.0
            effect_size = 0.0 if pooled_std == 0 else (model_mean - other_mean) / pooled_std
            records.append(
                {
                    "model_key": model_key,
                    "feature": feature,
                    "model_mean": model_mean,
                    "other_models_mean": other_mean,
                    "effect_size": effect_size,
                    "abs_effect_size": abs(effect_size),
                    "direction": "higher" if effect_size >= 0 else "lower",
                }
            )
    return pd.DataFrame(records).sort_values(["model_key", "abs_effect_size"], ascending=[True, False])


def group_mean(rows: pd.DataFrame, group_column: str, group_value: object, feature: str) -> float:
    group_rows = rows[rows[group_column] == group_value]
    return 0.0 if group_rows.empty else float(group_rows[feature].mean())


def build_language_paraphrase_sensitivity(profile_data: pd.DataFrame, feature_columns) -> pd.DataFrame:
    records = []
    for model_key in sorted(profile_data["model_key"].dropna().unique()):
        model_rows = profile_data[profile_data["model_key"] == model_key]
        for feature in feature_columns:
            en_mean = group_mean(model_rows, "language", "en", feature)
            pl_mean = group_mean(model_rows, "language", "pl", feature)
            base_mean = group_mean(model_rows, "is_paraphrase", False, feature)
            paraphrase_mean = group_mean(model_rows, "is_paraphrase", True, feature)
            records.append(
                {
                    "model_key": model_key,
                    "feature": feature,
                    "en_mean": en_mean,
                    "pl_mean": pl_mean,
                    "language_difference_en_minus_pl": en_mean - pl_mean,
                    "base_prompt_mean": base_mean,
                    "paraphrase_mean": paraphrase_mean,
                    "paraphrase_difference_true_minus_false": paraphrase_mean - base_mean,
                }
            )
    return pd.DataFrame(records)


def write_profile_markdown(top_features: pd.DataFrame, shap_summary: pd.DataFrame, PROFILES_DIR):
    lines = [
        "# Style Profiles",
        "",
        "Profiles are generated from engineered features derived from data/processed/final responses.",
        "",
    ]
    for model_key in sorted(top_features["model_key"].unique()):
        lines.extend([f"## {model_key}", "", "Top descriptive differences:"])
        model_features = top_features[top_features["model_key"] == model_key]
        for _, row in model_features.iterrows():
            lines.append(
                f"- `{row['feature']}` is {row['direction']} than other models "
                f"(effect size: {row['effect_size']:.3f})."
            )
        model_shap = shap_summary[shap_summary["class_name"] == model_key].sort_values("rank").head(5)
        if not model_shap.empty:
            lines.extend(["", "Top SHAP features for this model-key classifier class:"])
            for _, row in model_shap.iterrows():
                lines.append(f"- `{row['feature']}` (mean absolute SHAP: {row['mean_abs_shap']:.6f}).")
        lines.append("")
    (PROFILES_DIR / "style_profiles.md").write_text("\n".join(lines), encoding="utf-8")
    
    

def train_surrogate_tree(target: str, best_by_target, feature_splits, feature_columns, random_state, MODELS_DIR, XAI_DIR, max_depth: int = 3, min_samples_leaf: int = 30) -> pd.DataFrame:
    main_model = best_by_target[target]["model"]
    class_names = best_by_target[target]["class_names"]
    pseudo_train = main_model.predict(feature_splits["train"])

    surrogate = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        class_weight="balanced",
    )
    surrogate.fit(feature_splits["train"], pseudo_train)

    rows = []
    for split in ("train", "val", "test"):
        main_predictions = main_model.predict(feature_splits[split])
        surrogate_predictions = surrogate.predict(feature_splits[split])
        true_labels = best_by_target[target]["encoded_targets"][split]
        rows.append(
            {
                "target": target,
                "split": split,
                "surrogate_max_depth": max_depth,
                "surrogate_min_samples_leaf": min_samples_leaf,
                "fidelity_accuracy": accuracy_score(main_predictions, surrogate_predictions),
                "fidelity_macro_f1": f1_score(
                    main_predictions,
                    surrogate_predictions,
                    average="macro",
                    zero_division=0,
                ),
                "surrogate_task_accuracy": accuracy_score(true_labels, surrogate_predictions),
                "surrogate_task_macro_f1": f1_score(
                    true_labels,
                    surrogate_predictions,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    rules = export_text(surrogate, feature_names=feature_columns, show_weights=True)
    (XAI_DIR / "surrogate_trees" / f"{target}_surrogate_tree_rules.txt").write_text(rules, encoding="utf-8")
    joblib.dump(surrogate, MODELS_DIR / f"{target}_surrogate_tree.joblib")

    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        surrogate,
        feature_names=feature_columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title(f"Surrogate decision tree for {target}")
    fig.tight_layout()
    fig.savefig(XAI_DIR / "surrogate_trees" / f"{target}_surrogate_tree.png")
    plt.close(fig)

    return pd.DataFrame(rows)


def estimator_from_model(model):
    return model.named_steps["model"] if isinstance(model, Pipeline) else model


def built_in_importance(target: str, model, feature_columns) -> pd.DataFrame:
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


def permutation_importance_frame(target: str, model, best_by_target, feature_splits, feature_columns, random_state) -> pd.DataFrame:
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


def shap_values_to_importance(target: str, values: np.ndarray, class_names: list[str], feature_columns) -> pd.DataFrame:
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


def fallback_shap_frame(target: str, model, reason: str, feature_columns) -> pd.DataFrame:
    fallback = built_in_importance(target, model, feature_columns)
    fallback = fallback.rename(columns={"importance": "mean_abs_shap"})
    fallback["class_name"] = "__overall__"
    fallback["rank"] = range(1, len(fallback) + 1)
    fallback["importance_kind"] = "fallback_importance"
    fallback["note"] = reason
    return fallback[["target", "class_name", "feature", "mean_abs_shap", "rank", "importance_kind", "note"]]


def explain_with_shap(target: str, model, best_by_target, feature_splits, feature_columns, random_state) -> pd.DataFrame:
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


def plot_top_importance(df: pd.DataFrame, value_column: str, title: str, output_path: Path, top_n: int = 20):
    top = df.sort_values(value_column, ascending=False).head(top_n).sort_values(value_column)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top[value_column])
    ax.set_title(title)
    ax.set_xlabel(value_column)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_predictions(target: str, split: str, source_df: pd.DataFrame, y_true, y_pred, probabilities, class_names, best_by_target, XAI_DIR, prediction_context_columns):
    prediction_df = prediction_context(source_df, prediction_context_columns)
    prediction_df["target"] = target
    prediction_df["split"] = split
    prediction_df["y_true"] = best_by_target[target]["encoder"].inverse_transform(y_true)
    prediction_df["y_pred"] = best_by_target[target]["encoder"].inverse_transform(y_pred)
    prediction_df["correct"] = prediction_df["y_true"] == prediction_df["y_pred"]
    prediction_df["prediction_confidence"] = probabilities.max(axis=1)
    for class_index, class_name in enumerate(class_names):
        prediction_df[f"probability_{safe_filename_part(class_name)}"] = probabilities[:, class_index]
    prediction_df.to_csv(XAI_DIR / "predictions" / f"{target}_{split}_predictions.csv", index=False)


def save_confusion_matrix(target: str, split: str, y_true, y_pred, class_names, XAI_DIR):
    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        XAI_DIR / "confusion_matrices" / f"{target}_{split}_confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(f"{target} confusion matrix ({split})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(labels, labels=class_names, rotation=45, ha="right")
    ax.set_yticks(labels, labels=class_names)
    for row_index in labels:
        for column_index in labels:
            ax.text(column_index, row_index, matrix[row_index, column_index], ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(XAI_DIR / "confusion_matrices" / f"{target}_{split}_confusion_matrix.png")
    plt.close(fig)
    

def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "missing"
    
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    
    return str(value)


def prediction_context(df: pd.DataFrame, prediction_context_columns) -> pd.DataFrame:
    columns = [column for column in prediction_context_columns if column in df.columns]
    return df[columns].copy()


def safe_filename_part(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower() or "class"


def encode_target(target: str, final_splits: dict[str, pd.DataFrame]) -> tuple[LabelEncoder, dict[str, np.ndarray]]:
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(final_splits["train"][target].map(normalize_label))
    y_val = encoder.transform(final_splits["val"][target].map(normalize_label))
    y_test = encoder.transform(final_splits["test"][target].map(normalize_label))

    return encoder, {"train": y_train, "val": y_val, "test": y_test}


def metric_row(target: str, split: str, model_name: str, params: dict[str, Any], y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    return {
        "target": target,
        "split": split,
        "model_name": model_name,
        "params": json.dumps(params, sort_keys=True),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "classification_report": json.dumps(
            classification_report(
                y_true,
                y_pred,
                labels=labels,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            ),
            ensure_ascii=False,
        ),
    }


def build_model_candidates(random_state: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=random_state)),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 5.0],
                "model__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=random_state),
            {
                "n_estimators": [200, 500],
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 3],
            },
        ),
        "extra_trees": (
            ExtraTreesClassifier(class_weight="balanced", n_jobs=-1, random_state=random_state),
            {
                "n_estimators": [200, 500],
                "max_depth": [None, 8, 16],
                "min_samples_leaf": [1, 3],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [2, 3],
            },
        ),
    }


def fit_with_params(estimator, params: dict[str, Any]):
    model = clone(estimator)
    model.set_params(**params)
    return model


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


if __name__ == "__main__":
    main()
