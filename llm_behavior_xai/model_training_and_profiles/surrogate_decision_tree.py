from __future__ import annotations

from pathlib import Path
from typing import Literal, Any

import joblib
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree


def train_surrogate_tree(
    target: str,
    best_by_target,
    feature_splits,
    feature_columns,
    random_state,
    MODELS_DIR,
    XAI_DIR,
    max_depth: int = 3,
    min_samples_leaf: int = 30,
) -> pd.DataFrame:
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


def calculate_surrogate_decision_tree_metrics(
    MODELS_DIR: Path,
    RANDOM_STATE: int,
    TARGET_COLUMNS: tuple[Literal["model_key"], Literal["language"]],
    XAI_DIR: Path,
    best_by_target: DataFrame,
    feature_columns: list[str],
    feature_splits: dict[str, Series | DataFrame | Any],
) -> DataFrame:
    surrogate_metrics = pd.concat(
        [
            train_surrogate_tree(
                target, best_by_target, feature_splits, feature_columns, RANDOM_STATE, MODELS_DIR, XAI_DIR
            )
            for target in TARGET_COLUMNS
        ],
        ignore_index=True,
    )
    surrogate_metrics.to_csv(XAI_DIR / "surrogate_trees" / "surrogate_tree_metrics.csv", index=False)
    return surrogate_metrics
