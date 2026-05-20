from unittest.mock import patch

import pandas as pd

from llm_behavior_xai.llm_response_merger_and_validator.main import main


class TestMain:
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_summary_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_dataframe_report")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.analyze_df")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.clean_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.load_and_merge_datasets")
    def test_main_with_non_empty_datasets(
        self,
        mock_load_and_merge,
        mock_clean,
        mock_analyze,
        mock_print_report,
        mock_print_summary,
    ):
        train_df = pd.DataFrame(
            {
                "prompt_id": ["p1", "p2"],
                "response": ["R1", "R2"],
            }
        )
        val_df = pd.DataFrame(
            {
                "prompt_id": ["p3"],
                "response": ["R3"],
            }
        )
        test_df = pd.DataFrame(
            {
                "prompt_id": ["p4"],
                "response": ["R4"],
            }
        )

        mock_load_and_merge.return_value = {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

        cleaned_train = train_df.copy()
        cleaned_val = val_df.copy()
        cleaned_test = test_df.copy()

        mock_clean.side_effect = [
            (cleaned_train, 0, 1),
            (cleaned_val, 0, 0),
            (cleaned_test, 1, 0),
        ]

        report_train = {"name": "train", "total_rows": 2}
        report_val = {"name": "val", "total_rows": 1}
        report_test = {"name": "test", "total_rows": 1}

        mock_analyze.side_effect = [report_train, report_val, report_test]

        with patch.object(pd.DataFrame, "to_csv"):
            main()

        mock_load_and_merge.assert_called_once()

        assert mock_clean.call_count == 3
        assert mock_analyze.call_count == 3
        assert mock_print_report.call_count == 3

        mock_print_summary.assert_called_once()
        call_args = mock_print_summary.call_args
        reports = call_args[0][1]

        assert len(reports) == 3
        assert "train" in reports
        assert "val" in reports
        assert "test" in reports

    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_summary_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_dataframe_report")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.analyze_df")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.clean_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.load_and_merge_datasets")
    def test_main_skips_empty_datasets(
        self,
        mock_load_and_merge,
        mock_clean,
        mock_analyze,
        mock_print_report,
        mock_print_summary,
        capsys,
    ):
        train_df = pd.DataFrame(
            {
                "prompt_id": ["p1"],
                "response": ["R1"],
            }
        )
        val_df = pd.DataFrame()

        mock_load_and_merge.return_value = {
            "train": train_df,
            "val": val_df,
        }

        cleaned_train = train_df.copy()
        mock_clean.return_value = (cleaned_train, 0, 0)

        report_train = {"name": "train", "total_rows": 1}
        mock_analyze.return_value = report_train

        with patch.object(pd.DataFrame, "to_csv"):
            main()

        assert mock_clean.call_count == 1
        assert mock_analyze.call_count == 1

        captured = capsys.readouterr()

        assert "Skipping analysis for empty val" in captured.out

    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_summary_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_dataframe_report")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.analyze_df")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.clean_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.load_and_merge_datasets")
    def test_main_processes_clean_results(
        self,
        mock_load_and_merge,
        mock_clean,
        mock_analyze,
        mock_print_report,
        mock_print_summary,
        capsys,
    ):
        train_df = pd.DataFrame(
            {
                "prompt_id": ["p1", "p2", "p3"],
                "response": ["R1", "R2", "R3"],
            }
        )

        mock_load_and_merge.return_value = {"train": train_df}

        cleaned_train = pd.DataFrame(
            {
                "prompt_id": ["p1"],
                "response": ["R1"],
            }
        )

        mock_clean.return_value = (cleaned_train, 5, 3)

        report_train = {"name": "train", "total_rows": 1}
        mock_analyze.return_value = report_train

        with patch.object(pd.DataFrame, "to_csv"):
            main()

        captured = capsys.readouterr()

        assert "removed error rows=3" in captured.out
        assert "removed duplicate rows=5" in captured.out
        assert "rows=1" in captured.out

    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_summary_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_dataframe_report")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.analyze_df")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.clean_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.load_and_merge_datasets")
    def test_main_saves_to_correct_paths(
        self,
        mock_load_and_merge,
        mock_clean,
        mock_analyze,
        mock_print_report,
        mock_print_summary,
    ):
        train_df = pd.DataFrame({"prompt_id": ["p1"], "response": ["R1"]})
        val_df = pd.DataFrame({"prompt_id": ["p2"], "response": ["R2"]})

        mock_load_and_merge.return_value = {
            "train": train_df,
            "val": val_df,
        }

        cleaned_train = train_df.copy()
        cleaned_val = val_df.copy()

        mock_clean.side_effect = [
            (cleaned_train, 0, 0),
            (cleaned_val, 0, 0),
        ]

        mock_analyze.side_effect = [
            {"name": "train", "total_rows": 1},
            {"name": "val", "total_rows": 1},
        ]

        with patch.object(pd.DataFrame, "to_csv") as mock_to_csv:
            main()

        assert mock_to_csv.call_count == 2

        call_args_list = mock_to_csv.call_args_list
        first_call_path = str(call_args_list[0][0][0])
        second_call_path = str(call_args_list[1][0][0])

        assert "llm_results_train_full.csv" in first_call_path
        assert "llm_results_val_full.csv" in second_call_path

    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_summary_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_dataframe_report")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.analyze_df")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.clean_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.load_and_merge_datasets")
    def test_main_collects_output_paths(
        self,
        mock_load_and_merge,
        mock_clean,
        mock_analyze,
        mock_print_report,
        mock_print_summary,
    ):
        train_df = pd.DataFrame({"prompt_id": ["p1"], "response": ["R1"]})
        test_df = pd.DataFrame({"prompt_id": ["p2"], "response": ["R2"]})

        mock_load_and_merge.return_value = {
            "train": train_df,
            "test": test_df,
        }

        mock_clean.side_effect = [
            (train_df.copy(), 0, 0),
            (test_df.copy(), 0, 0),
        ]

        mock_analyze.side_effect = [
            {"name": "train", "total_rows": 1},
            {"name": "test", "total_rows": 1},
        ]

        with patch.object(pd.DataFrame, "to_csv"):
            main()

        mock_print_summary.assert_called_once()
        call_args = mock_print_summary.call_args
        output_paths = call_args[0][0]

        assert len(output_paths) == 2
        assert "train" in output_paths
        assert "test" in output_paths

    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_summary_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_dataframe_report")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.analyze_df")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.clean_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.load_and_merge_datasets")
    def test_main_calls_print_report_with_correct_args(
        self,
        mock_load_and_merge,
        mock_clean,
        mock_analyze,
        mock_print_report,
        mock_print_summary,
    ):
        train_df = pd.DataFrame({"prompt_id": ["p1"], "response": ["R1"]})

        mock_load_and_merge.return_value = {"train": train_df}

        cleaned_train = train_df.copy()
        mock_clean.return_value = (cleaned_train, 0, 0)

        report_train = {"name": "train", "total_rows": 1, "error_count": 0}
        mock_analyze.return_value = report_train

        with patch.object(pd.DataFrame, "to_csv"):
            main()

        mock_print_report.assert_called_once_with("train", report_train)

    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_summary_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.print_dataframe_report")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.analyze_df")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.clean_dataframe")
    @patch("llm_behavior_xai.llm_response_merger_and_validator.main.load_and_merge_datasets")
    def test_main_with_all_empty_datasets(
        self,
        mock_load_and_merge,
        mock_clean,
        mock_analyze,
        mock_print_report,
        mock_print_summary,
        capsys,
    ):
        mock_load_and_merge.return_value = {
            "train": pd.DataFrame(),
            "val": pd.DataFrame(),
            "test": pd.DataFrame(),
        }

        with patch.object(pd.DataFrame, "to_csv"):
            main()

        mock_clean.assert_not_called()
        mock_analyze.assert_not_called()
        captured = capsys.readouterr()

        assert "Skipping analysis for empty train" in captured.out
        assert "Skipping analysis for empty val" in captured.out
        assert "Skipping analysis for empty test" in captured.out

        mock_print_summary.assert_called_once_with({}, {})
