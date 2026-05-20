import pandas as pd
import pytest

from llm_behavior_xai.llm_response_merger_and_validator.validate_dataset import (
    find_column,
    analyze_df,
    normalize_lang,
    normalize_paraphrase,
    normalize_paraphrase_from_variant,
    normalize_paraphrase_from_text,
    print_dataframe_report,
    print_summary_dataframe,
)


class TestFindColumn:
    def test_find_column_matches_and_none(self):
        df = pd.DataFrame(columns=["Prompt_ID", "response_text", "Language", "other"])

        assert find_column(df, ["prompt_id", "id"]) == "Prompt_ID"
        assert find_column(df, ["response"]) == "response_text"
        assert find_column(df, ["does_not_exist"]) is None


class TestNormalizeLang:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("pl", "pl"),
            ("PL", "pl"),
            ("polish", "pl"),
            ("en-US", "en"),
            ("english", "en"),
            ("fr", None),
            (None, None),
        ],
    )
    def test_normalize_lang(self, value, expected):
        assert normalize_lang(value) == expected


class TestNormalizeParaphrase:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", "paraphrase"),
            ("True", "paraphrase"),
            ("para", "paraphrase"),
            ("yes", "paraphrase"),
            ("no", "no_paraphrase"),
            ("0", "no_paraphrase"),
        ],
    )
    def test_normalize_paraphrase(self, value, expected):
        assert normalize_paraphrase(value) == expected


class TestNormalizeParaphraseFromVariantAndText:
    def test_normalize_paraphrase_from_variant_and_text(self):
        assert normalize_paraphrase_from_variant("paraphrase_mode") == "paraphrase"
        assert normalize_paraphrase_from_variant("no_variant") == "no_paraphrase"

        assert normalize_paraphrase_from_text("this is a paraphrase example") == "paraphrase"
        assert normalize_paraphrase_from_text("completely different text") == "no_paraphrase"


class TestAnalyzeDf:
    def test_analyze_df_error_detection_with_nan_and_empty_strings(self):
        rows = [
            {"prompt_id": "p1", "response": "R1"},
            {"prompt_id": "p2", "response": None},
            {"prompt_id": "p3", "response": "   "},
            {"prompt_id": "p4", "response": ""},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="error_test")

        assert report["error_count"] == 3

    def test_analyze_df_error_detection_no_response_column(self):
        rows = [
            {"prompt_id": "p1"},
            {"prompt_id": "p2"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="no_response")

        assert report["error_count"] == 0

    def test_analyze_df_unique_ids_from_prompt_id_col(self):
        rows = [
            {"prompt_id": "p1", "response": "R1"},
            {"prompt_id": "p1", "response": "R2"},
            {"prompt_id": "p2", "response": "R3"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="id_col_test")

        assert report["unique_prompt_ids"] == 2

    def test_analyze_df_unique_ids_from_prompt_col_when_no_id_col(self):
        rows = [
            {"prompt": "Q1", "response": "R1"},
            {"prompt": "Q1", "response": "R2"},
            {"prompt": "Q2", "response": "R3"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="prompt_col_test")

        assert report["unique_prompt_ids"] == 2

    def test_analyze_df_duplicate_rows_detection(self):
        rows = [
            {"prompt_id": "p1", "response": "R1"},
            {"prompt_id": "p1", "response": "R1"},
            {"prompt_id": "p1", "response": "R1"},
            {"prompt_id": "p1", "response": "R2"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="dup_test")

        assert report["duplicate_rows_same_prompt_response"] == 2

    def test_analyze_df_variants_and_duplicates_and_errors(self):
        rows = [
            {"prompt_id": "p1", "prompt": "Q1", "response": "R1", "lang": "en", "paraphrase": "no"},
            {"prompt_id": "p1", "prompt": "Q1", "response": "R1_para", "lang": "en", "paraphrase": "para"},
            {"prompt_id": "p1", "prompt": "Q1", "response": "R1_pl", "lang": "pl", "paraphrase": "no"},
            {"prompt_id": "p1", "prompt": "Q1", "response": "R1_pl_para", "lang": "pl", "paraphrase": "para"},
            {"prompt_id": "p2", "prompt": "Q2", "response": "R2", "lang": "en", "paraphrase": "no"},
            {"prompt_id": "p2", "prompt": "Q2", "response": "R2", "lang": "en", "paraphrase": "no"},
            {
                "prompt_id": "p3",
                "prompt": "Q3",
                "response": "",
                "lang": "en",
                "paraphrase": "no",
                "error_msg": "timeout",
            },
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="testset")

        assert report["name"] == "testset"
        assert report["total_rows"] == len(df)
        assert report["error_count"] == 1
        assert report["unique_prompt_ids"] == 3
        assert report["duplicate_rows_same_prompt_response"] == 1
        assert report["prompts_with_multiple_unique_responses"] == 1
        assert report["prompt_ids_with_full_variant_set"] == 1

        missing = {item["prompt_id"] for item in report["missing_variants_by_prompt_id"]}

        assert "p2" in missing

    def test_analyze_df_paraphrase_normalization_from_paraphrase_col(self):
        rows = [
            {"prompt_id": "p1", "response": "R1", "lang": "en", "paraphrase": "true"},
            {"prompt_id": "p1", "response": "R2", "lang": "en", "paraphrase": "no"},
            {"prompt_id": "p1", "response": "R3", "lang": "pl", "paraphrase": "1"},
            {"prompt_id": "p1", "response": "R4", "lang": "pl", "paraphrase": "0"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="para_col_test")

        assert report["prompt_ids_with_full_variant_set"] == 1
        assert len(report["missing_variants_by_prompt_id"]) == 0

    def test_analyze_df_paraphrase_normalization_from_variant_col(self):
        rows = [
            {"prompt_id": "p1", "response": "R1", "lang": "en", "variant": "paraphrase"},
            {"prompt_id": "p1", "response": "R2", "lang": "en", "variant": "standard"},
            {"prompt_id": "p1", "response": "R3", "lang": "pl", "variant": "paraphrase_mode"},
            {"prompt_id": "p1", "response": "R4", "lang": "pl", "variant": "normal"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="variant_col_test")

        assert report["prompt_ids_with_full_variant_set"] == 1
        assert len(report["missing_variants_by_prompt_id"]) == 0

    def test_analyze_df_paraphrase_normalization_from_response_text(self):
        rows = [
            {"prompt_id": "p1", "response": "This is a paraphrase", "lang": "en"},
            {"prompt_id": "p1", "response": "Original text", "lang": "en"},
            {"prompt_id": "p1", "response": "Paraphrased version", "lang": "pl"},
            {"prompt_id": "p1", "response": "Standard response", "lang": "pl"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="response_text_test")

        assert report["prompt_ids_with_full_variant_set"] == 1
        assert len(report["missing_variants_by_prompt_id"]) == 0

    def test_analyze_df_full_variant_set_counting(self):
        rows = [
            {"prompt_id": "p1", "response": "R1", "lang": "en", "paraphrase": "no"},
            {"prompt_id": "p1", "response": "R2", "lang": "en", "paraphrase": "yes"},
            {"prompt_id": "p1", "response": "R3", "lang": "pl", "paraphrase": "no"},
            {"prompt_id": "p1", "response": "R4", "lang": "pl", "paraphrase": "yes"},
            {"prompt_id": "p2", "response": "R5", "lang": "en", "paraphrase": "no"},
            {"prompt_id": "p2", "response": "R6", "lang": "en", "paraphrase": "yes"},
            {"prompt_id": "p3", "response": "R7", "lang": "en", "paraphrase": "no"},
            {"prompt_id": "p3", "response": "R8", "lang": "en", "paraphrase": "yes"},
            {"prompt_id": "p3", "response": "R9", "lang": "pl", "paraphrase": "no"},
            {"prompt_id": "p3", "response": "R10", "lang": "pl", "paraphrase": "yes"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="full_variant_test")

        assert report["prompt_ids_with_full_variant_set"] == 2

        missing = {item["prompt_id"] for item in report["missing_variants_by_prompt_id"]}

        assert "p2" in missing
        assert len(missing) == 1

    def test_analyze_df_unique_prompt_ids_none_when_no_id_or_prompt_col(self):
        rows = [
            {"response": "R1"},
            {"response": "R2"},
        ]

        df = pd.DataFrame(rows)
        report = analyze_df(df, name="no_id_test")

        assert report["unique_prompt_ids"] is None


class TestPrintDataFrameReport:
    def test_print_dataframe_report(self, capsys):
        report = {"total_rows": 2, "name": "a", "error_count": 1}
        print_dataframe_report("testset", report)
        captured = capsys.readouterr()

        assert "Report for testset:" in captured.out
        assert "error_count" in captured.out

    def test_print_dataframe_report_with_empty_missing_variants(self, capsys):
        report = {
            "total_rows": 5,
            "error_count": 0,
            "missing_variants_by_prompt_id": [],
        }
        print_dataframe_report("testset", report)
        captured = capsys.readouterr()

        assert "Report for testset:" in captured.out
        assert "missing_variants_by_prompt_id: []" in captured.out

    def test_print_dataframe_report_with_missing_variants(self, capsys):
        report = {
            "total_rows": 5,
            "error_count": 0,
            "missing_variants_by_prompt_id": [
                {"prompt_id": "p1", "missing": ["EN without paraphrase", "PL with paraphrase"]},
                {"prompt_id": "p2", "missing": ["PL without paraphrase"]},
            ],
        }
        print_dataframe_report("testset", report)
        captured = capsys.readouterr()

        assert "Report for testset:" in captured.out
        assert "missing_variants_by_prompt_id:" in captured.out
        assert "prompt_id=p1:" in captured.out
        assert "prompt_id=p2:" in captured.out
        assert "EN without paraphrase" in captured.out
        assert "PL with paraphrase" in captured.out
        assert "PL without paraphrase" in captured.out


class TestPrintSummaryTable:
    def test_print_summary_dataframe(self, capsys):
        reports = {"a": {"total_rows": 2, "name": "a"}, "b": {"total_rows": 3, "name": "b"}}
        output_paths = {"summary": "/tmp/out.csv"}

        print_summary_dataframe(output_paths, reports)
        captured = capsys.readouterr()

        assert "Summary table:" in captured.out
        assert "/tmp/out.csv" in captured.out

    def test_print_summary_dataframe_empty_reports(self, capsys):
        reports = {}
        output_paths = {"summary": "/tmp/out.csv"}

        print_summary_dataframe(output_paths, reports)
        captured = capsys.readouterr()

        assert captured.out == ""

    def test_print_summary_dataframe_multiple_output_paths(self, capsys):
        reports = {
            "train": {"total_rows": 100, "name": "train"},
            "test": {"total_rows": 20, "name": "test"},
            "val": {"total_rows": 30, "name": "val"},
        }
        output_paths = {
            "train_report": "/tmp/train.csv",
            "test_report": "/tmp/test.csv",
            "val_report": "/tmp/val.csv",
        }

        print_summary_dataframe(output_paths, reports)
        captured = capsys.readouterr()

        assert "Summary table:" in captured.out
        assert "Saved files:" in captured.out
        assert "/tmp/train.csv" in captured.out
        assert "/tmp/test.csv" in captured.out
        assert "/tmp/val.csv" in captured.out
