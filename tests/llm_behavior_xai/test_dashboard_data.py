import pandas as pd

from llm_behavior_xai.dashboard.data import (
    filter_results,
    load_final_results,
    load_predictions,
    load_shap_importance,
    load_style_profiles,
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
    pd.DataFrame({"target": ["model_key"], "split": ["test"], "macro_f1": [1.0]}).to_csv(
        reports_dir / "all_metrics.csv", index=False
    )
    pd.DataFrame({"target": ["model_key"], "split": ["test"], "correct": [True]}).to_csv(
        reports_dir / "predictions" / "model_key_test_predictions.csv", index=False
    )
    pd.DataFrame({"target": ["model_key"], "class_name": ["__overall__"], "feature": ["text_word_count"]}).to_csv(
        reports_dir / "shap" / "model_key_shap_importance.csv", index=False
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
    assert not load_style_profiles(profiles_dir)["top_features"].empty

