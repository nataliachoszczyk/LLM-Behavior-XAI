import pandas as pd

from llm_behavior_xai.dashboard.data import (
    filter_results,
    get_surrogate_tree_plot_path,
    load_feature_group_importance,
    load_final_results,
    load_predictions,
    load_shap_importance,
    load_style_profiles,
    load_surrogate_tree_metrics,
    load_surrogate_tree_rules,
    load_xai_metrics,
)


def test_dashboard_loaders_read_generated_artifacts(tmp_path):
    split_paths = {}
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}.csv"
        pd.DataFrame(
            {
                "prompt_id": [f"{split}_1"],
                "category": ["explain"],
                "language": ["en"],
                "is_paraphrase": [False],
                "model_key": ["model_a"],
                "response": ["Answer"],
                "response_length": [6],
            }
        ).to_csv(path, index=False)
        split_paths[split] = path

    reports_dir = tmp_path / "xai"
    (reports_dir / "predictions").mkdir(parents=True)
    (reports_dir / "shap").mkdir(parents=True)
    (reports_dir / "importance").mkdir(parents=True)
    (reports_dir / "surrogate_trees").mkdir(parents=True)
    pd.DataFrame({"target": ["model_key"], "split": ["test"], "macro_f1": [1.0]}).to_csv(
        reports_dir / "all_metrics.csv", index=False
    )
    pd.DataFrame({"target": ["model_key"], "split": ["test"], "correct": [True]}).to_csv(
        reports_dir / "predictions" / "model_key_test_predictions.csv", index=False
    )
    pd.DataFrame({"target": ["model_key"], "class_name": ["__overall__"], "feature": ["text_word_count"]}).to_csv(
        reports_dir / "shap" / "model_key_shap_importance.csv", index=False
    )
    pd.DataFrame(
        {
            "target": ["model_key"],
            "method": ["shap"],
            "feature_group": ["length_and_structure"],
            "importance_share": [1.0],
        }
    ).to_csv(reports_dir / "importance" / "feature_group_importance.csv", index=False)
    pd.DataFrame({"target": ["model_key"], "split": ["test"], "fidelity_accuracy": [0.9]}).to_csv(
        reports_dir / "surrogate_trees" / "surrogate_tree_metrics.csv", index=False
    )
    (reports_dir / "surrogate_trees" / "model_key_surrogate_tree_rules.txt").write_text(
        "text_word_count <= 10",
        encoding="utf-8",
    )

    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    pd.DataFrame({"model_key": ["model_a"], "feature": ["text_word_count"], "effect_size": [1.0]}).to_csv(
        profiles_dir / "model_top_features.csv", index=False
    )

    results = load_final_results(split_paths)
    filtered = filter_results(results, splits=["test"], models=["model_a"], languages=["en"])

    assert len(results) == 3
    assert len(filtered) == 1
    assert not load_xai_metrics(reports_dir).empty
    assert not load_predictions("model_key", "test", reports_dir).empty
    assert not load_shap_importance("model_key", reports_dir).empty
    assert not load_feature_group_importance(reports_dir).empty
    assert not load_surrogate_tree_metrics(reports_dir).empty
    assert load_surrogate_tree_rules("model_key", reports_dir)
    assert get_surrogate_tree_plot_path("model_key", reports_dir).name == "model_key_surrogate_tree.png"
    assert not load_style_profiles(profiles_dir)["top_features"].empty
