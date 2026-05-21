from __future__ import annotations

import json
from pathlib import Path
from re import Pattern
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from llm_behavior_xai.model_training_and_profiles.build_numeric_training_features import build_feature_frame


def eta_squared_by_group(values: pd.Series, labels: pd.Series) -> float:
    overall_mean = values.mean()
    total_variation = ((values - overall_mean) ** 2).sum()

    if total_variation == 0 or pd.isna(total_variation):
        return 0.0

    between_variation = 0.0

    for _, group_values in values.groupby(labels):
        between_variation += len(group_values) * (group_values.mean() - overall_mean) ** 2

    return float(between_variation / total_variation)


def check_feature_model_signal(
    random_state: int,
    xai_dir: Path,
    feature_columns: list[str],
    feature_splits: dict[str, pd.DataFrame],
    train_model_labels: pd.Series,
) -> pd.DataFrame:

    try:
        mutual_information = mutual_info_classif(
            feature_splits["train"],
            train_model_labels,
            discrete_features=False,
            random_state=random_state,
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
    feature_model_signal_check.to_csv(xai_dir / "features" / "feature_model_signal_check.csv", index=False)

    return feature_model_signal_check


def filter_dataset(
    target_columns: tuple[str, ...],
    xai_dir: Path,
    excluded_zero_nan_features: list[str],
    feature_columns: list[str],
    feature_meanings: pd.DataFrame,
    feature_splits: dict[str, pd.DataFrame],
) -> tuple[list[str], dict[str, pd.Series | pd.DataFrame | Any]]:

    feature_columns = [feature for feature in feature_columns if feature not in excluded_zero_nan_features]
    feature_splits = {split: features[feature_columns].copy() for split, features in feature_splits.items()}
    feature_list = pd.DataFrame({"feature": feature_columns})
    feature_list.to_csv(xai_dir / "features" / "feature_list.csv", index=False)

    if "feature_meanings" in globals():
        feature_meanings = feature_meanings.copy()
        feature_meanings["used_for_training"] = feature_meanings["feature"].isin(feature_columns)
        feature_meanings["excluded_reason"] = np.where(
            feature_meanings["feature"].isin(excluded_zero_nan_features),
            "model-specific zero/NaN pattern",
            "",
        )
        feature_meanings.to_csv(xai_dir / "features" / "feature_descriptions.csv", index=False)

    with (xai_dir / "features" / "feature_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "feature_columns": feature_columns,
                "excluded_zero_nan_features": excluded_zero_nan_features,
                "source": "data/processed/final",
                "targets": target_columns,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    return feature_columns, feature_splits


def exclude_model_specific_and_zero_nan_patterns(
    first_person_pronouns: set[Any],
    heading_re: Pattern[str],
    hedge_words: set[str],
    list_marker_re: Pattern[str],
    negation_words: set[str],
    second_person_pronouns: set[str],
    sentence_re: Pattern[str],
    source_numeric_columns: tuple[str, ...],
    word_re: Pattern[str],
    xai_dir: Path,
    feature_columns: list[str],
    final_splits: dict[str, pd.DataFrame],
) -> tuple[list, pd.DataFrame]:

    raw_feature_frames = []
    raw_context_frames = []

    for split, source_df in final_splits.items():
        raw_features = build_feature_frame(
            source_df,
            source_numeric_columns,
            word_re,
            sentence_re,
            list_marker_re,
            heading_re,
            first_person_pronouns,
            second_person_pronouns,
            hedge_words,
            negation_words,
        ).reindex(columns=feature_columns)
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
    feature_zero_nan_check.to_csv(xai_dir / "features" / "feature_zero_nan_by_model_check.csv", index=False)

    excluded_zero_nan_features = sorted(
        feature_zero_nan_check.loc[feature_zero_nan_check["review_flag"] != "", "feature"].unique()
    )
    pd.DataFrame({"excluded_feature": excluded_zero_nan_features}).to_csv(
        xai_dir / "features" / "excluded_zero_nan_features.csv",
        index=False,
    )

    if excluded_zero_nan_features:
        print("Excluding features with model-specific zero/NaN patterns:")
        for feature in excluded_zero_nan_features:
            print(f"- {feature}")
    else:
        print("No model-specific zero/NaN features were flagged for exclusion.")
    return excluded_zero_nan_features, feature_zero_nan_check
