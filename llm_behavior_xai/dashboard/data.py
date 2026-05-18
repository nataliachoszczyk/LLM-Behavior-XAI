from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    LLM_RESULTS_TEST_PROMPTS,
    LLM_RESULTS_TRAIN_PROMPTS,
    LLM_RESULTS_VAL_PROMPTS,
    STYLE_PROFILES_REPORTS_DIR,
    XAI_REPORTS_DIR,
)


FINAL_SPLIT_PATHS = {
    "train": LLM_RESULTS_TRAIN_PROMPTS,
    "val": LLM_RESULTS_VAL_PROMPTS,
    "test": LLM_RESULTS_TEST_PROMPTS,
}


def load_final_results(split_paths: dict[str, Path] | None = None) -> pd.DataFrame:
    paths = split_paths or FINAL_SPLIT_PATHS
    frames = []
    for split, path in paths.items():
        split_df = pd.read_csv(path)
        split_df["split"] = split
        frames.append(split_df)
    return pd.concat(frames, ignore_index=True)


def filter_results(
    results: pd.DataFrame,
    splits: list[str] | None = None,
    models: list[str] | None = None,
    languages: list[str] | None = None,
    paraphrase_values: list[bool] | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    filtered = results.copy()
    if splits:
        filtered = filtered[filtered["split"].isin(splits)]
    if models and "model_key" in filtered.columns:
        filtered = filtered[filtered["model_key"].isin(models)]
    if languages and "language" in filtered.columns:
        filtered = filtered[filtered["language"].isin(languages)]
    if paraphrase_values and "is_paraphrase" in filtered.columns:
        filtered = filtered[filtered["is_paraphrase"].isin(paraphrase_values)]
    if categories and "category" in filtered.columns:
        filtered = filtered[filtered["category"].isin(categories)]
    return filtered


def load_xai_metrics(reports_dir: Path = XAI_REPORTS_DIR) -> pd.DataFrame:
    path = reports_dir / "all_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_predictions(target: str, split: str, reports_dir: Path = XAI_REPORTS_DIR) -> pd.DataFrame:
    path = reports_dir / "predictions" / f"{target}_{split}_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_shap_importance(target: str, reports_dir: Path = XAI_REPORTS_DIR) -> pd.DataFrame:
    path = reports_dir / "shap" / f"{target}_shap_importance.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_style_profiles(reports_dir: Path = STYLE_PROFILES_REPORTS_DIR) -> dict[str, pd.DataFrame]:
    files = {
        "top_features": reports_dir / "model_top_features.csv",
        "effect_sizes": reports_dir / "model_effect_sizes.csv",
        "sensitivity": reports_dir / "language_paraphrase_sensitivity.csv",
        "feature_summary": reports_dir / "model_feature_summary.csv",
    }
    return {name: pd.read_csv(path) if path.exists() else pd.DataFrame() for name, path in files.items()}
