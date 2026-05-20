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
    def test_analyze_df_variants_and_duplicates_and_errors(self, capsys):
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


class TestPrintDataFrameReport:
    def test_print_dataframe_report(self, capsys):
        report = {"total_rows": 2, "name": "a", "error_count": 1}
        print_dataframe_report("testset", report)
        captured = capsys.readouterr()

        assert "Report for testset:" in captured.out
        assert "error_count" in captured.out


class TestPrintSummaryTable:
    def test_print_summary_dataframe(self, capsys):
        reports = {"a": {"total_rows": 2, "name": "a"}, "b": {"total_rows": 3, "name": "b"}}
        output_paths = {"summary": "/tmp/out.csv"}

        print_summary_dataframe(output_paths, reports)
        captured = capsys.readouterr()

        assert "Summary table:" in captured.out
        assert "/tmp/out.csv" in captured.out
