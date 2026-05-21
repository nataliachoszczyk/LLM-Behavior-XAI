from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Any

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import confusion_matrix

from llm_behavior_xai.model_training_and_profiles.train_and_validate_models import (
    metric_row,
    prediction_context,
    safe_filename_part,
)


def calculate_final_metrics(
    MODELS_DIR: Path,
    PREDICTION_CONTEXT_COLUMNS: tuple[
        Literal["run_id"],
        Literal["prompt_id"],
        Literal["category"],
        Literal["prompt_column"],
        Literal["language"],
        Literal["is_paraphrase"],
        Literal["model_key"],
        Literal["provider"],
        Literal["prompt_text"],
        Literal["response"],
    ],
    SELECTION_METRIC: str,
    XAI_DIR: Path,
    best_by_target: DataFrame,
    feature_columns: list[str],
    feature_splits: dict[str, Series | DataFrame | Any],
    final_splits: dict[str, DataFrame],
) -> DataFrame:
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
            save_predictions(
                target,
                split,
                final_splits[split],
                y_true,
                y_pred,
                probabilities,
                class_names,
                best_by_target,
                XAI_DIR,
                PREDICTION_CONTEXT_COLUMNS,
            )
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
    return metrics_df


def save_predictions(
    target: str,
    split: str,
    source_df: pd.DataFrame,
    y_true,
    y_pred,
    probabilities,
    class_names,
    best_by_target,
    XAI_DIR,
    prediction_context_columns,
):
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
