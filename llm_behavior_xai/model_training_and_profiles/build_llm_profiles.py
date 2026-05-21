from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_llm_profiles(
    profiles_dir: Path,
    xai_dir: Path,
    feature_columns: list[str],
    feature_splits: dict[str, pd.DataFrame],
    final_splits: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    profile_data = combine_profile_data(final_splits, feature_splits)
    feature_summary = build_model_feature_summary(profile_data, feature_columns)
    effect_sizes = build_model_effect_sizes(profile_data, feature_columns)
    top_features = effect_sizes.groupby("model_key", group_keys=False).head(8).reset_index(drop=True)
    sensitivity = build_language_paraphrase_sensitivity(profile_data, feature_columns)

    model_key_shap_path = xai_dir / "shap" / "model_key_shap_importance.csv"
    shap_summary = pd.read_csv(model_key_shap_path) if model_key_shap_path.exists() else pd.DataFrame()

    if not shap_summary.empty:
        shap_summary = shap_summary[shap_summary["class_name"] != "__overall__"].copy()

    feature_summary.to_csv(profiles_dir / "model_feature_summary.csv", index=False)
    effect_sizes.to_csv(profiles_dir / "model_effect_sizes.csv", index=False)
    top_features.to_csv(profiles_dir / "model_top_features.csv", index=False)
    sensitivity.to_csv(profiles_dir / "language_paraphrase_sensitivity.csv", index=False)
    write_profile_markdown(top_features, shap_summary, profiles_dir)

    return top_features


def combine_profile_data(
    final_splits: dict[str, pd.DataFrame], feature_splits: dict[str, pd.DataFrame]
) -> pd.DataFrame:

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


def build_model_feature_summary(profile_data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    summary = profile_data.groupby("model_key")[feature_columns].agg(["mean", "median", "std"]).reset_index()
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns.to_flat_index()]

    return summary


def build_model_effect_sizes(profile_data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
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


def build_language_paraphrase_sensitivity(profile_data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
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


def write_profile_markdown(top_features: pd.DataFrame, shap_summary: pd.DataFrame, profiles_dir: Path) -> None:
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
                f"- `{row['feature']}` is {row['direction']} than other models (effect size: {row['effect_size']:.3f})."
            )

        model_shap = shap_summary[shap_summary["class_name"] == model_key].sort_values("rank").head(5)

        if not model_shap.empty:
            lines.extend(["", "Top SHAP features for this model-key classifier class:"])

            for _, row in model_shap.iterrows():
                lines.append(f"- `{row['feature']}` (mean absolute SHAP: {row['mean_abs_shap']:.6f}).")

        lines.append("")

    (profiles_dir / "style_profiles.md").write_text("\n".join(lines), encoding="utf-8")
