import pandas as pd

from llm_behavior_xai.llm_response_merger_and_validator.clean_dataset import (
    clean_dataframe,
    deduplicate_by_prompt_and_response,
    get_error_mask,
)


class TestDeduplicateByPromptAndResponse:
    def test_removes_duplicate_rows_keeping_last(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q1", "q2"],
                "response": ["a1", "a1", "a2"],
                "extra": [1, 2, 3],
            }
        )
        result = deduplicate_by_prompt_and_response(df)

        assert len(result) == 2
        assert result[result["prompt"] == "q1"]["extra"].values[0] == 2

    def test_returns_copy_when_no_matching_columns(self):
        df = pd.DataFrame({"col_x": [1, 2], "col_y": [3, 4]})
        result = deduplicate_by_prompt_and_response(df)

        assert len(result) == 2
        assert result is not df

    def test_no_duplicates_unchanged(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q2"],
                "response": ["a1", "a2"],
            }
        )
        result = deduplicate_by_prompt_and_response(df)

        assert len(result) == 2

    def test_alternate_column_names_instruction_answer(self):
        df = pd.DataFrame(
            {
                "instruction": ["q1", "q1"],
                "answer": ["a1", "a1"],
            }
        )
        result = deduplicate_by_prompt_and_response(df)

        assert len(result) == 1

    def test_same_prompt_different_responses_kept(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q1"],
                "response": ["a1", "a2"],
            }
        )
        result = deduplicate_by_prompt_and_response(df)

        assert len(result) == 2

    def test_returns_dataframe(self):
        df = pd.DataFrame({"prompt": ["q1"], "response": ["a1"]})
        result = deduplicate_by_prompt_and_response(df)

        assert isinstance(result, pd.DataFrame)


class TestGetErrorMask:
    def test_error_column_notna_marks_error(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q2", "q3"],
                "response": ["a1", "a2", "a3"],
                "error": [None, "some error", None],
            }
        )
        mask = get_error_mask(df)

        assert mask.tolist() == [False, True, False]

    def test_no_error_col_nan_response_is_error(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q2"],
                "response": ["a1", None],
            }
        )
        mask = get_error_mask(df)

        assert mask.tolist() == [False, True]

    def test_no_error_col_empty_string_response_is_error(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q2"],
                "response": ["a1", "   "],
            }
        )
        mask = get_error_mask(df)

        assert mask.tolist() == [False, True]

    def test_no_matching_columns_returns_all_false(self):
        df = pd.DataFrame({"col_x": [1, 2], "col_y": [3, 4]})
        mask = get_error_mask(df)

        assert mask.tolist() == [False, False]

    def test_returns_series(self):
        df = pd.DataFrame({"response": ["a1"]})
        mask = get_error_mask(df)

        assert isinstance(mask, pd.Series)

    def test_all_valid_rows_no_errors(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q2"],
                "response": ["a1", "a2"],
            }
        )
        mask = get_error_mask(df)

        assert not mask.any()


class TestCleanDataframe:
    def test_removes_error_rows(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q2", "q3"],
                "response": ["a1", None, "a3"],
            }
        )
        deduped, removed_duplicates, removed_errors = clean_dataframe(df)

        assert removed_errors == 1
        assert len(deduped) == 2

    def test_removes_duplicate_rows(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q1", "q2"],
                "response": ["a1", "a1", "a2"],
            }
        )
        deduped, removed_duplicates, removed_errors = clean_dataframe(df)

        assert removed_duplicates == 1
        assert len(deduped) == 2

    def test_returns_tuple_of_three(self):
        df = pd.DataFrame({"prompt": ["q1"], "response": ["a1"]})
        result = clean_dataframe(df)

        assert len(result) == 3

    def test_removes_errors_before_deduplication(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q1", "q2"],
                "response": [None, "a1", "a2"],
            }
        )
        deduped, removed_duplicates, removed_errors = clean_dataframe(df)

        assert removed_errors == 1
        assert removed_duplicates == 0
        assert len(deduped) == 2

    def test_returns_dataframe(self):
        df = pd.DataFrame({"prompt": ["q1"], "response": ["a1"]})
        deduped, _, _ = clean_dataframe(df)

        assert isinstance(deduped, pd.DataFrame)

    def test_no_issues_returns_zero_counts(self):
        df = pd.DataFrame(
            {
                "prompt": ["q1", "q2"],
                "response": ["a1", "a2"],
            }
        )
        deduped, removed_duplicates, removed_errors = clean_dataframe(df)

        assert removed_errors == 0
        assert removed_duplicates == 0
        assert len(deduped) == 2
