from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Any

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn import clone
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_and_validate_models(
    RANDOM_STATE: int,
    SELECTION_METRIC: str,
    TARGET_COLUMNS: tuple[Literal["model_key"], Literal["language"]],
    XAI_DIR: Path,
    feature_splits: dict[str, Series | DataFrame | Any],
    final_splits: dict[str, DataFrame],
) -> tuple[DataFrame, dict[Any, Any]]:
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
    return best_by_target, validation_metrics


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
