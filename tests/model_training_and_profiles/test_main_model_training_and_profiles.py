from unittest.mock import patch

import pandas as pd

from llm_behavior_xai.model_training_and_profiles.main import main


class TestMain:
    @patch("llm_behavior_xai.model_training_and_profiles.main.build_llm_profiles")
    @patch("llm_behavior_xai.model_training_and_profiles.main.calculate_surrogate_decision_tree_metrics")
    @patch("llm_behavior_xai.model_training_and_profiles.main.calculate_feature_group_importance")
    @patch("llm_behavior_xai.model_training_and_profiles.main.calculate_outputs_importances")
    @patch("llm_behavior_xai.model_training_and_profiles.main.calculate_final_metrics")
    @patch("llm_behavior_xai.model_training_and_profiles.main.train_and_validate_models")
    @patch("llm_behavior_xai.model_training_and_profiles.main.filter_dataset")
    @patch("llm_behavior_xai.model_training_and_profiles.main.exclude_model_specific_and_zero_nan_patterns")
    @patch("llm_behavior_xai.model_training_and_profiles.main.check_feature_model_signal")
    @patch("llm_behavior_xai.model_training_and_profiles.main.create_feature_descriptions")
    @patch("llm_behavior_xai.model_training_and_profiles.main.create_feature_list")
    @patch("llm_behavior_xai.model_training_and_profiles.main.build_feature_splits")
    @patch("llm_behavior_xai.model_training_and_profiles.main.create_split_overview")
    @patch("llm_behavior_xai.model_training_and_profiles.main.load_final_splits")
    def test_main_executes_pipeline(
        self,
        mock_load_final_splits,
        mock_create_split_overview,
        mock_build_feature_splits,
        mock_create_feature_list,
        mock_create_feature_descriptions,
        mock_check_feature_model_signal,
        mock_exclude_model_specific_and_zero_nan_patterns,
        mock_filter_dataset,
        mock_train_and_validate_models,
        mock_calculate_final_metrics,
        mock_calculate_outputs_importances,
        mock_calculate_feature_group_importance,
        mock_calculate_surrogate_decision_tree_metrics,
        mock_build_llm_profiles,
    ):
        final_splits = {
            "train": pd.DataFrame({"model_key": ["gpt"]}),
            "val": pd.DataFrame({"model_key": ["gpt"]}),
            "test": pd.DataFrame({"model_key": ["gpt"]}),
        }

        feature_splits = {
            "train": pd.DataFrame({"feature": [1]}),
            "val": pd.DataFrame({"feature": [2]}),
            "test": pd.DataFrame({"feature": [3]}),
        }

        feature_columns = ["feature_1", "feature_2"]
        fill_values = {"feature_1": 0}

        mock_load_final_splits.return_value = final_splits
        mock_create_split_overview.return_value = {"train": {"rows": 1}}

        mock_build_feature_splits.return_value = (
            feature_splits,
            feature_columns,
            fill_values,
        )

        mock_create_feature_list.return_value = pd.DataFrame({"feature": feature_columns})

        mock_create_feature_descriptions.return_value = pd.DataFrame(
            {
                "feature": feature_columns,
                "description": ["desc1", "desc2"],
            }
        )

        mock_check_feature_model_signal.return_value = pd.DataFrame({"feature": feature_columns})

        mock_exclude_model_specific_and_zero_nan_patterns.return_value = (
            [],
            pd.DataFrame({"review_flag": [""]}),
        )

        mock_filter_dataset.return_value = (
            feature_columns,
            feature_splits,
        )

        best_by_target = {
            "model_key": {
                "model_name": "RandomForest",
                "params": {"n_estimators": 10},
                "val_score": 0.9,
                "val_balanced_accuracy": 0.88,
            }
        }

        validation_metrics = pd.DataFrame({"macro_f1": [0.9]})

        mock_train_and_validate_models.return_value = (
            best_by_target,
            validation_metrics,
        )

        mock_calculate_final_metrics.return_value = pd.DataFrame(
            {
                "split": ["test"],
                "target": ["model_key"],
                "accuracy": [0.9],
                "macro_f1": [0.89],
                "balanced_accuracy": [0.88],
                "params": ["{}"],
                "model_name": ["RandomForest"],
            }
        )

        mock_calculate_outputs_importances.return_value = (
            [pd.DataFrame({"importance": [1]})],
            [pd.DataFrame({"importance": [1]})],
            [pd.DataFrame({"importance": [1]})],
        )

        mock_calculate_feature_group_importance.return_value = pd.DataFrame(
            {
                "target": ["model_key"],
                "method": ["shap"],
                "importance_share": [0.5],
            }
        )

        mock_calculate_surrogate_decision_tree_metrics.return_value = pd.DataFrame({"score": [0.8]})

        mock_build_llm_profiles.return_value = pd.DataFrame({"feature": ["feature_1"]})

        with patch("llm_behavior_xai.model_training_and_profiles.main.pd.DataFrame.to_csv"):
            main()

        mock_load_final_splits.assert_called_once()
        mock_create_split_overview.assert_called_once()
        mock_build_feature_splits.assert_called_once()
        mock_create_feature_list.assert_called_once()
        mock_create_feature_descriptions.assert_called_once()
        mock_check_feature_model_signal.assert_called_once()
        mock_exclude_model_specific_and_zero_nan_patterns.assert_called_once()
        mock_filter_dataset.assert_called_once()
        mock_train_and_validate_models.assert_called_once()
        mock_calculate_final_metrics.assert_called_once()
        mock_calculate_outputs_importances.assert_called_once()
        mock_calculate_feature_group_importance.assert_called_once()
        mock_calculate_surrogate_decision_tree_metrics.assert_called_once()
        mock_build_llm_profiles.assert_called_once()
