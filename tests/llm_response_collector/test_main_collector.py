from pathlib import Path
from unittest.mock import patch

import pytest

from llm_response_collector.main import main


TRAIN = Path("/data/train_prompts.csv")
VAL = Path("/data/val_prompts.csv")
TEST = Path("/data/test_prompts.csv")
TRAIN_RESULTS = Path("/out/train.csv")
VAL_RESULTS = Path("/out/val.csv")
TEST_RESULTS = Path("/out/test.csv")
TMP_DIR = Path("/tmp/responses")
TRAIN_BASE = Path("/out/llm_results_train.csv")
VAL_BASE = Path("/out/llm_results_val.csv")
TEST_BASE = Path("/out/llm_results_test.csv")


@pytest.fixture(autouse=True)
def patch_config():
    with (
        patch("llm_response_collector.main.TRAIN_PROMPTS", TRAIN),
        patch("llm_response_collector.main.VAL_PROMPTS", VAL),
        patch("llm_response_collector.main.TEST_PROMPTS", TEST),
        patch("llm_response_collector.main.LLM_RESULTS_TRAIN_PROMPTS", TRAIN_RESULTS),
        patch("llm_response_collector.main.LLM_RESULTS_VAL_PROMPTS", VAL_RESULTS),
        patch("llm_response_collector.main.LLM_RESULTS_TEST_PROMPTS", TEST_RESULTS),
        patch("llm_response_collector.main.TEMPORARY_RESPONSES_DIR", TMP_DIR),
        patch("llm_response_collector.main.LLM_RESULTS_TRAIN_PROMPTS_BASE_PATH", TRAIN_BASE),
        patch("llm_response_collector.main.LLM_RESULTS_VAL_PROMPTS_BASE_PATH", VAL_BASE),
        patch("llm_response_collector.main.LLM_RESULTS_TEST_PROMPTS_BASE_PATH", TEST_BASE),
    ):
        yield


class TestMainFullMode:
    @pytest.fixture(autouse=True)
    def set_full_mode(self):
        with patch("llm_response_collector.main.RUN_MODE", "full"):
            yield

    def test_calls_collector_pipeline_three_times(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00"),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        assert mock_pipeline.call_count == 3

    def test_passes_train_prompts_path(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00"),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        prompts_paths = [c.args[0] for c in mock_pipeline.call_args_list]

        assert TRAIN in prompts_paths

    def test_passes_val_prompts_path(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00"),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        prompts_paths = [c.args[0] for c in mock_pipeline.call_args_list]

        assert VAL in prompts_paths

    def test_passes_test_prompts_path(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00"),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        prompts_paths = [c.args[0] for c in mock_pipeline.call_args_list]

        assert TEST in prompts_paths

    def test_results_paths_are_inside_tmp_dir(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00"),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        results_paths = [c.args[1] for c in mock_pipeline.call_args_list]

        assert all(str(p).startswith(str(TMP_DIR)) for p in results_paths)

    def test_results_paths_contain_timestamp(self):
        timestamp = "2026-05-18_12-00-00"
        with (
            patch("llm_response_collector.main.get_timestamp", return_value=timestamp),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        results_paths = [c.args[1] for c in mock_pipeline.call_args_list]

        assert all(timestamp in str(p) for p in results_paths)

    def test_results_paths_contain_base_stem(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00"),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        results_paths = [str(c.args[1]) for c in mock_pipeline.call_args_list]

        assert any(TRAIN_BASE.stem in p for p in results_paths)
        assert any(VAL_BASE.stem in p for p in results_paths)
        assert any(TEST_BASE.stem in p for p in results_paths)

    def test_results_paths_are_csv_files(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00"),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        results_paths = [c.args[1] for c in mock_pipeline.call_args_list]

        assert all(str(p).endswith(".csv") for p in results_paths)

    def test_get_timestamp_called_once(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00") as mock_ts,
            patch("llm_response_collector.main.collector_pipeline"),
        ):
            main()

        mock_ts.assert_called_once()

    def test_all_three_results_paths_are_distinct(self):
        with (
            patch("llm_response_collector.main.get_timestamp", return_value="2026-05-18_12-00-00"),
            patch("llm_response_collector.main.collector_pipeline") as mock_pipeline,
        ):
            main()

        results_paths = [c.args[1] for c in mock_pipeline.call_args_list]

        assert len(set(str(p) for p in results_paths)) == 3


class TestMainNonFullMode:
    @pytest.fixture(autouse=True)
    def set_non_full_mode(self):
        with patch("llm_response_collector.main.RUN_MODE", "test"):
            yield

    def test_calls_collector_pipeline_three_times(self):
        with patch("llm_response_collector.main.collector_pipeline") as mock_pipeline:
            main()

        assert mock_pipeline.call_count == 3

    def test_uses_fixed_train_results_path(self):
        with patch("llm_response_collector.main.collector_pipeline") as mock_pipeline:
            main()

        calls = {c.args[0]: c.args[1] for c in mock_pipeline.call_args_list}

        assert calls[TRAIN] == TRAIN_RESULTS

    def test_uses_fixed_val_results_path(self):
        with patch("llm_response_collector.main.collector_pipeline") as mock_pipeline:
            main()

        calls = {c.args[0]: c.args[1] for c in mock_pipeline.call_args_list}

        assert calls[VAL] == VAL_RESULTS

    def test_uses_fixed_test_results_path(self):
        with patch("llm_response_collector.main.collector_pipeline") as mock_pipeline:
            main()

        calls = {c.args[0]: c.args[1] for c in mock_pipeline.call_args_list}

        assert calls[TEST] == TEST_RESULTS

    def test_does_not_call_get_timestamp(self):
        with (
            patch("llm_response_collector.main.get_timestamp") as mock_ts,
            patch("llm_response_collector.main.collector_pipeline"),
        ):
            main()

        mock_ts.assert_not_called()

    def test_pipeline_called_in_order_train_val_test(self):
        with patch("llm_response_collector.main.collector_pipeline") as mock_pipeline:
            main()

        prompts_paths = [c.args[0] for c in mock_pipeline.call_args_list]

        assert prompts_paths == [TRAIN, VAL, TEST]
