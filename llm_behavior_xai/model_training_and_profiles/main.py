from __future__ import annotations

import json
import re

import matplotlib

from llm_behavior_xai.model_training_and_profiles.build_llm_profiles import build_llm_profiles
from llm_behavior_xai.model_training_and_profiles.explainability import (
    calculate_outputs_importances,
    calculate_feature_group_importance,
)
from llm_behavior_xai.model_training_and_profiles.metrics import calculate_final_metrics
from llm_behavior_xai.model_training_and_profiles.split_datasets import load_final_splits, create_split_overview
from llm_behavior_xai.model_training_and_profiles.surrogate_decision_tree import (
    calculate_surrogate_decision_tree_metrics,
)
from llm_behavior_xai.model_training_and_profiles.train_and_validate_models import (
    train_and_validate_models,
)
from llm_behavior_xai.model_training_and_profiles.verify_training_features import (
    check_feature_model_signal,
    filter_dataset,
    exclude_model_spefic_and_zero_nan_patterns,
)

matplotlib.use("Agg")
import pandas as pd

from llm_behavior_xai.config import (
    LLM_RESULTS_TRAIN_PROMPTS,
    LLM_RESULTS_VAL_PROMPTS,
    LLM_RESULTS_TEST_PROMPTS,
    XAI_MODELS_DIR,
    XAI_REPORTS_DIR,
    STYLE_PROFILES_REPORTS_DIR,
)
from llm_behavior_xai.model_training_and_profiles.build_numeric_training_features import (
    build_feature_splits,
    create_feature_descriptions,
    create_feature_list,
)


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

    final_splits = load_final_splits(FINAL_SPLIT_PATHS)
    create_split_overview(final_splits)

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

    feature_list = create_feature_list(TARGET_COLUMNS, XAI_DIR, feature_columns, fill_values)

    print()
    print(f"-- Feature count: {len(feature_columns)} -----------------------------------------------------------------")
    print(feature_list)

    feature_meanings = create_feature_descriptions(XAI_DIR, feature_columns)

    print()
    print("-- Feature meanings: -----------------------------------------------------------------")
    print(feature_meanings)

    train_model_labels = final_splits["train"]["model_key"].astype(str)
    feature_model_signal_check = check_feature_model_signal(
        RANDOM_STATE, XAI_DIR, feature_columns, feature_splits, train_model_labels
    )

    print()
    print("-- Model's features signal check: ----------------------------------------------------")
    print(feature_model_signal_check.head(25))

    excluded_zero_nan_features, feature_zero_nan_check = exclude_model_spefic_and_zero_nan_patterns(
        FIRST_PERSON_PRONOUNS,
        HEADING_RE,
        HEDGE_WORDS,
        LIST_MARKER_RE,
        NEGATION_WORDS,
        SECOND_PERSON_PRONOUNS,
        SENTENCE_RE,
        SOURCE_NUMERIC_COLUMNS,
        WORD_RE,
        XAI_DIR,
        feature_columns,
        final_splits,
    )

    feature_columns, feature_splits = filter_dataset(
        TARGET_COLUMNS, XAI_DIR, excluded_zero_nan_features, feature_columns, feature_meanings, feature_splits
    )

    print()
    print(f"-- Remaining training feature count: {len(feature_columns)} ----------------------------------------------")
    print(feature_zero_nan_check[feature_zero_nan_check["review_flag"] != ""].head(30))

    best_by_target, validation_metrics = train_and_validate_models(
        RANDOM_STATE, SELECTION_METRIC, TARGET_COLUMNS, XAI_DIR, feature_splits, final_splits
    )

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

    metrics_df = calculate_final_metrics(
        MODELS_DIR,
        PREDICTION_CONTEXT_COLUMNS,
        SELECTION_METRIC,
        XAI_DIR,
        best_by_target,
        feature_columns,
        feature_splits,
        final_splits,
    )

    print()
    print("-- Final test metrics for selected models: -------------------------------------------")
    print(
        metrics_df[metrics_df["split"].isin(["val", "test"])][
            ["target", "split", "model_name", "accuracy", "macro_f1", "balanced_accuracy", "params"]
        ]
    )

    importance_outputs, permutation_outputs, shap_outputs = calculate_outputs_importances(
        RANDOM_STATE, XAI_DIR, best_by_target, feature_columns, feature_splits
    )

    print()
    print("-- Importance outputs: ---------------------------------------------------------------")
    print(pd.concat(importance_outputs).head(20))

    feature_group_importance = calculate_feature_group_importance(
        TARGET_COLUMNS, XAI_DIR, importance_outputs, permutation_outputs, shap_outputs
    )

    print()
    print("-- Feature group importance: ---------------------------------------------------------")
    print(feature_group_importance.sort_values(["target", "method", "importance_share"], ascending=[True, True, False]))

    surrogate_metrics = calculate_surrogate_decision_tree_metrics(
        MODELS_DIR, RANDOM_STATE, TARGET_COLUMNS, XAI_DIR, best_by_target, feature_columns, feature_splits
    )

    print()
    print("-- Surrogate tree metrics: -----------------------------------------------------------")
    print(surrogate_metrics)

    top_features = build_llm_profiles(PROFILES_DIR, XAI_DIR, feature_columns, feature_splits, final_splits)

    print()
    print("-- Profiles: ------------------------------------------------------------------------------")
    print(top_features)


if __name__ == "__main__":
    main()
