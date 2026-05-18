import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from llm_response_collector.pipeline import (
    validate_prompts,
    clean_invalid_responses,
    has_valid_response,
    build_parameters_string,
    flush_checkpoint,
    load_models_config,
    load_prompts,
    get_timestamp,
    model_pipeline,
    ask_gemini_till_exhaustion,
    collector_pipeline,
    run_pipeline,
)


class TestValidatePrompts:
    def test_passes_when_all_required_columns_present(self):
        df = pd.DataFrame(columns=["prompt_id", "category", "prompt_en", "prompt_pl"])
        validate_prompts(df, ["prompt_en", "prompt_pl"])

    def test_raises_when_prompt_id_missing(self):
        df = pd.DataFrame(columns=["category", "prompt_en"])

        with pytest.raises(ValueError, match="prompt_id"):
            validate_prompts(df, ["prompt_en"])

    def test_raises_when_category_missing(self):
        df = pd.DataFrame(columns=["prompt_id", "prompt_en"])

        with pytest.raises(ValueError, match="category"):
            validate_prompts(df, ["prompt_en"])

    def test_raises_when_prompt_column_missing(self):
        df = pd.DataFrame(columns=["prompt_id", "category"])

        with pytest.raises(ValueError, match="prompt_en"):
            validate_prompts(df, ["prompt_en"])

    def test_error_message_lists_missing_columns(self):
        df = pd.DataFrame(columns=["prompt_id"])

        with pytest.raises(ValueError, match="Missing required columns"):
            validate_prompts(df, ["prompt_en", "prompt_pl"])

    def test_empty_prompt_columns_list_passes(self):
        df = pd.DataFrame(columns=["prompt_id", "category"])
        validate_prompts(df, [])


class TestCleanInvalidResponses:
    def test_keeps_rows_with_response_and_no_error(self):
        df = pd.DataFrame({"response": ["text"], "error": [None]})
        result = clean_invalid_responses(df)

        assert len(result) == 1

    def test_removes_rows_with_none_response(self):
        df = pd.DataFrame({"response": [None, "text"], "error": [None, None]})
        result = clean_invalid_responses(df)

        assert len(result) == 1
        assert result.iloc[0]["response"] == "text"

    def test_removes_rows_with_error_present(self):
        df = pd.DataFrame({"response": ["text", "other"], "error": ["some error", None]})
        result = clean_invalid_responses(df)

        assert len(result) == 1
        assert result.iloc[0]["response"] == "other"

    def test_removes_rows_with_both_response_and_error(self):
        df = pd.DataFrame({"response": ["text"], "error": ["err"]})
        result = clean_invalid_responses(df)

        assert len(result) == 0

    def test_resets_index(self):
        df = pd.DataFrame({"response": [None, "text", None], "error": [None, None, None]})
        result = clean_invalid_responses(df)

        assert list(result.index) == [0]

    def test_returns_dataframe(self):
        df = pd.DataFrame({"response": ["a"], "error": [None]})
        result = clean_invalid_responses(df)

        assert isinstance(result, pd.DataFrame)

    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame({"response": [], "error": []})
        result = clean_invalid_responses(df)

        assert len(result) == 0


class TestHasValidResponse:
    @pytest.fixture
    def base_df(self):
        return pd.DataFrame(
            {
                "prompt_id": ["1"],
                "category": ["definition"],
                "prompt_text": ["What is AI?"],
                "language": ["en"],
                "is_paraphrase": [False],
                "model_key": ["gpt4"],
                "response": ["AI is..."],
                "error": [None],
            }
        )

    def test_returns_true_for_exact_match(self, base_df):
        result = has_valid_response(base_df, "1", "definition", "What is AI?", "en", False, "gpt4")

        assert result is np.True_

    def test_returns_false_when_prompt_id_differs(self, base_df):
        result = has_valid_response(base_df, "99", "definition", "What is AI?", "en", False, "gpt4")

        assert result is np.False_

    def test_returns_false_when_model_key_differs(self, base_df):
        result = has_valid_response(base_df, "1", "definition", "What is AI?", "en", False, "claude")

        assert result is np.False_

    def test_returns_false_when_response_is_none(self):
        df = pd.DataFrame(
            {
                "prompt_id": ["1"],
                "category": ["definition"],
                "prompt_text": ["What is AI?"],
                "language": ["en"],
                "is_paraphrase": [False],
                "model_key": ["gpt4"],
                "response": [None],
                "error": [None],
            }
        )
        result = has_valid_response(df, "1", "definition", "What is AI?", "en", False, "gpt4")

        assert result is np.False_

    def test_returns_false_when_error_is_not_none(self):
        df = pd.DataFrame(
            {
                "prompt_id": ["1"],
                "category": ["definition"],
                "prompt_text": ["What is AI?"],
                "language": ["en"],
                "is_paraphrase": [False],
                "model_key": ["gpt4"],
                "response": ["AI is..."],
                "error": ["timeout"],
            }
        )
        result = has_valid_response(df, "1", "definition", "What is AI?", "en", False, "gpt4")

        assert result is np.False_

    def test_returns_false_for_empty_dataframe(self):
        df = pd.DataFrame(
            columns=[
                "prompt_id",
                "category",
                "prompt_text",
                "language",
                "is_paraphrase",
                "model_key",
                "response",
                "error",
            ]
        )
        result = has_valid_response(df, "1", "definition", "What is AI?", "en", False, "gpt4")

        assert result is np.False_


class TestBuildParametersString:
    def test_includes_temperature(self):
        result = build_parameters_string({"temperature": 0.3})

        assert "temperature=0.3" in result

    def test_includes_top_p(self):
        result = build_parameters_string({"top_p": 0.9})

        assert "top_p=0.9" in result

    def test_includes_max_tokens(self):
        result = build_parameters_string({"max_tokens": 512})

        assert "max_tokens=512" in result

    def test_includes_max_new_tokens(self):
        result = build_parameters_string({"max_new_tokens": 1024})

        assert "max_new_tokens=1024" in result

    def test_includes_repetition_penalty(self):
        result = build_parameters_string({"repetition_penalty": 1.1})

        assert "repetition_penalty=1.1" in result

    def test_skips_absent_keys(self):
        result = build_parameters_string({"temperature": 0.3})

        assert "top_p" not in result
        assert "max_tokens" not in result

    def test_respects_ordering(self):
        cfg = {"repetition_penalty": 1.1, "temperature": 0.3, "top_p": 0.9}
        result = build_parameters_string(cfg)
        temp_pos = result.index("temperature")
        top_p_pos = result.index("top_p")
        rep_pos = result.index("repetition_penalty")

        assert temp_pos < top_p_pos < rep_pos

    def test_empty_config_returns_empty_string(self):
        result = build_parameters_string({})

        assert result == ""

    def test_parts_joined_by_comma_space(self):
        result = build_parameters_string({"temperature": 0.3, "top_p": 0.9})

        assert ", " in result

    def test_ignores_unknown_keys(self):
        result = build_parameters_string({"unknown_key": 42})

        assert result == ""


class TestFlushCheckpoint:
    def test_does_nothing_when_rows_empty(self, tmp_path):
        output = tmp_path / "out.csv"
        flush_checkpoint([], output)

        assert not output.exists()

    def test_creates_file_when_it_does_not_exist(self, tmp_path):
        output = tmp_path / "out.csv"
        flush_checkpoint([{"a": 1}], output)

        assert output.exists()

    def test_writes_header_when_file_does_not_exist(self, tmp_path):
        output = tmp_path / "out.csv"
        flush_checkpoint([{"col": "val"}], output)
        content = output.read_text(encoding="utf-8-sig")

        assert "col" in content

    def test_appends_rows_to_existing_file(self, tmp_path):
        output = tmp_path / "out.csv"
        flush_checkpoint([{"col": "first"}], output)
        flush_checkpoint([{"col": "second"}], output)
        content = output.read_text(encoding="utf-8-sig")

        assert "first" in content
        assert "second" in content

    def test_no_duplicate_header_on_append(self, tmp_path):
        output = tmp_path / "out.csv"
        flush_checkpoint([{"col": "first"}], output)
        flush_checkpoint([{"col": "second"}], output)
        content = output.read_text(encoding="utf-8-sig")

        assert content.count("col") == 1

    def test_accepts_string_path(self, tmp_path):
        output = str(tmp_path / "out.csv")
        flush_checkpoint([{"a": 1}], output)

        assert Path(output).exists()


class TestLoadModelsConfig:
    def test_adds_parameters_key_to_each_model(self):
        models_config = {
            "model-a": {"provider": "X", "temperature": 0.3, "top_p": 0.9},
            "model-b": {"provider": "Y", "temperature": 0.5},
        }

        with patch("llm_response_collector.pipeline.MODELS_CONFIG", models_config):
            load_models_config()

        assert "parameters" in models_config["model-a"]
        assert "parameters" in models_config["model-b"]

    def test_parameters_value_is_string(self):
        models_config = {"model-a": {"provider": "X", "temperature": 0.3}}

        with patch("llm_response_collector.pipeline.MODELS_CONFIG", models_config):
            load_models_config()

        assert isinstance(models_config["model-a"]["parameters"], str)

    def test_prints_loaded_message(self, capsys):
        with patch("llm_response_collector.pipeline.MODELS_CONFIG", {}):
            load_models_config()

        assert "Model configuration loaded" in capsys.readouterr().out


class TestLoadPrompts:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "prompt_id": [" 1 ", "2"],
                "category": ["definition", ""],
                "prompt_en": ["q1", "q2"],
                "prompt_pl": ["p1", "p2"],
                "paraphrase_en": ["q1p", "q2p"],
                "paraphrase_pl": ["p1p", "p2p"],
            }
        )

    def test_returns_dataframe(self, sample_df, tmp_path):
        prompts_path = tmp_path / "prompts.csv"

        with (
            patch("llm_response_collector.pipeline.read_prompts", return_value=sample_df),
            patch("llm_response_collector.pipeline.validate_prompts"),
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["prompt_en", "prompt_pl"]),
        ):
            result = load_prompts(prompts_path)

        assert isinstance(result, pd.DataFrame)

    def test_strips_whitespace_from_prompt_id(self, sample_df, tmp_path):
        prompts_path = tmp_path / "prompts.csv"

        with (
            patch("llm_response_collector.pipeline.read_prompts", return_value=sample_df),
            patch("llm_response_collector.pipeline.validate_prompts"),
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["prompt_en", "prompt_pl"]),
        ):
            result = load_prompts(prompts_path)

        assert result.iloc[0]["prompt_id"] == "1"

    def test_empty_category_replaced_with_unknown(self, sample_df, tmp_path):
        prompts_path = tmp_path / "prompts.csv"

        with (
            patch("llm_response_collector.pipeline.read_prompts", return_value=sample_df),
            patch("llm_response_collector.pipeline.validate_prompts"),
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["prompt_en", "prompt_pl"]),
        ):
            result = load_prompts(prompts_path)

        assert result.iloc[1]["category"] == "unknown"

    def test_calls_validate_prompts(self, sample_df, tmp_path):
        prompts_path = tmp_path / "prompts.csv"

        with (
            patch("llm_response_collector.pipeline.read_prompts", return_value=sample_df),
            patch("llm_response_collector.pipeline.validate_prompts") as mock_validate,
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["prompt_en"]),
        ):
            load_prompts(prompts_path)

        mock_validate.assert_called_once()


class TestGetTimestamp:
    def test_returns_string(self):
        assert isinstance(get_timestamp(), str)

    def test_matches_expected_format(self):
        ts = get_timestamp()

        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", ts)


class TestAskGeminiTillExhaustion:
    @pytest.fixture(autouse=True)
    def patch_config(self):
        with (
            patch("llm_response_collector.pipeline.GEMINI_API_KEYS", ["key-0", "key-1", "key-2"]),
            patch("llm_response_collector.pipeline.MODELS_CONFIG", {"gemini-flash-latest": {"provider": "Google"}}),
        ):
            yield

    def test_returns_tuple_of_four(self):
        with patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.5, {})):
            result = ask_gemini_till_exhaustion([], 0, MagicMock(), "gemini-flash-latest", "prompt", Path("/tmp/out"))

        assert len(result) == 4

    def test_returns_response_on_success(self):
        with patch("llm_response_collector.pipeline.query_model", return_value=("answer", None, 1.0, {})):
            elapsed, error, stats, response = ask_gemini_till_exhaustion(
                [], 0, MagicMock(), "gemini-flash-latest", "prompt", Path("/tmp/out")
            )

        assert response == "answer"
        assert error is None

    def test_switches_api_key_on_error(self):
        with (
            patch(
                "llm_response_collector.pipeline.query_model",
                side_effect=[
                    (None, "quota exceeded", 0.0, {}),
                    ("ok", None, 1.0, {}),
                ],
            ),
            patch("llm_response_collector.pipeline.get_gemini_client") as mock_get_client,
        ):
            ask_gemini_till_exhaustion([], 0, MagicMock(), "gemini-flash-latest", "prompt", Path("/tmp/out"))

        mock_get_client.assert_called_once_with("key-1")

    def test_raises_when_all_keys_exhausted(self, tmp_path):
        output = tmp_path / "out.csv"

        with patch(
            "llm_response_collector.pipeline.query_model",
            return_value=(None, "quota exceeded", 0.0, {}),
        ):
            with pytest.raises(RuntimeError, match="exhausted"):
                ask_gemini_till_exhaustion([], 0, MagicMock(), "gemini-flash-latest", "prompt", output)

    def test_flushes_checkpoint_before_raising(self, tmp_path):
        output = tmp_path / "out.csv"
        rows = [{"col": "val"}]

        with patch(
            "llm_response_collector.pipeline.query_model",
            return_value=(None, "quota exceeded", 0.0, {}),
        ):
            with pytest.raises(RuntimeError):
                ask_gemini_till_exhaustion(rows, 0, MagicMock(), "gemini-flash-latest", "prompt", output)

        assert output.exists()


MOCK_MODELS_CONFIG = {
    "test-model": {
        "provider": "Groq API",
        "model_id": "test-model-id",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 512,
        "type": "api",
        "parameters": "temperature=0.3",
    }
}


class TestModelPipeline:
    @pytest.fixture(autouse=True)
    def patch_config(self):
        with (
            patch("llm_response_collector.pipeline.MODELS_CONFIG", MOCK_MODELS_CONFIG),
            patch("llm_response_collector.pipeline.CHECKPOINT_EVERY", 10),
            patch("llm_response_collector.pipeline.time.sleep"),
        ):
            yield

    def _call(self, results=None, checkpoint_buffer=None, results_path=None, **kwargs):
        if results is None:
            results = []
        if checkpoint_buffer is None:
            checkpoint_buffer = []
        if results_path is None:
            results_path = Path("/tmp/nonexistent_results.csv")

        defaults = dict(
            model_key="test-model",
            call_count=1,
            total_calls=10,
            run_id=1,
            prompt_id="1",
            prompt_col="prompt_en",
            category="definition",
            prompt_text="What is AI?",
            lang="en",
            is_paraphrase=False,
            results_path=results_path,
            results=results,
            checkpoint_buffer=checkpoint_buffer,
            gemini_api_key_index=0,
            gemini_client=MagicMock(),
            groq_client=MagicMock(),
            local_models={},
        )
        defaults.update(kwargs)

        return model_pipeline(**defaults)

    def test_returns_list(self):
        with patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.0, {})):
            result = self._call()

        assert isinstance(result, list)

    def test_appends_row_to_results(self):
        results = []

        with patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.0, {})):
            self._call(results=results)

        assert len(results) == 1

    def test_appends_row_to_checkpoint_buffer(self):
        buf = []

        with patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.0, {})):
            result = self._call(checkpoint_buffer=buf)

        assert len(result) == 1

    def test_row_contains_expected_keys(self):
        results = []

        with patch("llm_response_collector.pipeline.query_model", return_value=("response text", None, 1.2, {})):
            self._call(results=results)

        row = results[0]

        for key in ["prompt_id", "model_key", "response", "error", "elapsed_seconds", "timestamp"]:
            assert key in row

    def test_row_response_length_calculated(self):
        results = []

        with patch("llm_response_collector.pipeline.query_model", return_value=("hello", None, 1.0, {})):
            self._call(results=results)

        assert results[0]["response_length"] == 5

    def test_row_error_empty_string_when_no_error(self):
        results = []

        with patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.0, {})):
            self._call(results=results)

        assert results[0]["error"] == ""

    def test_row_response_empty_string_when_none(self):
        results = []

        with patch("llm_response_collector.pipeline.query_model", return_value=(None, "timeout", 0.5, {})):
            self._call(results=results)

        assert results[0]["response"] == ""

    def test_skips_when_valid_response_exists(self, tmp_path):
        results_path = tmp_path / "results.csv"
        results_path.write_text("col\n", encoding="utf-8")
        results = []

        with (
            patch("llm_response_collector.pipeline.read_llm_results", return_value=MagicMock()),
            patch("llm_response_collector.pipeline.has_valid_response", return_value=True),
            patch("llm_response_collector.pipeline.query_model") as mock_query,
        ):
            self._call(results=results, results_path=results_path)

        mock_query.assert_not_called()
        assert len(results) == 0

    def test_flushes_when_checkpoint_buffer_full(self, tmp_path):
        results_path = tmp_path / "results.csv"
        buf = [{"col": "val"}] * 9

        with (
            patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.0, {})),
            patch("llm_response_collector.pipeline.CHECKPOINT_EVERY", 10),
            patch("llm_response_collector.pipeline.flush_checkpoint") as mock_flush,
        ):
            result = self._call(checkpoint_buffer=buf, results_path=results_path)

        mock_flush.assert_called_once()

        assert result == []

    def test_gemini_model_calls_ask_gemini_till_exhaustion(self):
        gemini_config = {
            "gemini-flash-latest": {
                "provider": "Google Gemini API",
                "model_id": "gemini-flash-latest",
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 512,
                "type": "api",
                "parameters": "temperature=0.3",
            }
        }

        with (
            patch("llm_response_collector.pipeline.MODELS_CONFIG", gemini_config),
            patch(
                "llm_response_collector.pipeline.ask_gemini_till_exhaustion",
                return_value=(1.0, None, {}, "gemini response"),
            ) as mock_ask,
        ):
            self._call(model_key="gemini-flash-latest")

        mock_ask.assert_called_once()

    def test_non_gemini_model_calls_query_model(self):
        with patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.0, {})) as mock_query:
            self._call(model_key="test-model")

        mock_query.assert_called_once()

    def test_api_model_sleeps_after_query(self):
        with (
            patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.0, {})),
            patch("llm_response_collector.pipeline.time.sleep") as mock_sleep,
        ):
            self._call()

        mock_sleep.assert_called_once_with(1)

    def test_logprob_stats_stored_in_row(self):
        stats = {
            "logprob_available": True,
            "sum_logprob": -5.0,
            "avg_logprob": -1.0,
            "generated_tokens": 5,
            "perplexity": 2.7,
        }
        results = []

        with patch("llm_response_collector.pipeline.query_model", return_value=("text", None, 1.0, stats)):
            self._call(results=results)

        row = results[0]

        assert row["logprob_available"] is True
        assert row["avg_logprob"] == -1.0
        assert row["generated_tokens"] == 5


class TestCollectorPipeline:
    @pytest.fixture
    def mock_prompts_df(self):
        return pd.DataFrame(
            {
                "prompt_id": ["1"],
                "category": ["definition"],
                "prompt_en": ["q"],
                "prompt_pl": ["p"],
                "paraphrase_en": ["q2"],
                "paraphrase_pl": ["p2"],
            }
        )

    def test_prints_message_and_returns_when_prompts_file_missing(self, tmp_path, capsys):
        missing = tmp_path / "missing.csv"
        results_path = tmp_path / "results.csv"
        collector_pipeline(missing, results_path)

        assert "not found" in capsys.readouterr().out

    def test_does_not_call_load_clients_when_prompts_missing(self, tmp_path):
        missing = tmp_path / "missing.csv"
        results_path = tmp_path / "results.csv"

        with patch("llm_response_collector.pipeline.load_clients") as mock_lc:
            collector_pipeline(missing, results_path)

        mock_lc.assert_not_called()

    def test_calls_load_clients_when_prompts_present(self, tmp_path, mock_prompts_df):
        prompts_path = tmp_path / "prompts.csv"
        prompts_path.write_text("x\n")
        results_path = tmp_path / "results.csv"

        with (
            patch("llm_response_collector.pipeline.load_prompts", return_value=mock_prompts_df),
            patch("llm_response_collector.pipeline.load_models_config"),
            patch(
                "llm_response_collector.pipeline.load_clients", return_value=(0, MagicMock(), MagicMock(), {})
            ) as mock_lc,
            patch("llm_response_collector.pipeline.run_pipeline", return_value=[]),
            patch("llm_response_collector.pipeline.flush_checkpoint"),
            patch("llm_response_collector.pipeline.N_RUNS", 1),
            patch("llm_response_collector.pipeline.MODELS_CONFIG", {}),
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", []),
        ):
            collector_pipeline(prompts_path, results_path)

        mock_lc.assert_called_once()

    def test_cleans_existing_results_if_file_present(self, tmp_path, mock_prompts_df):
        prompts_path = tmp_path / "prompts.csv"
        prompts_path.write_text("x\n")
        results_path = tmp_path / "results.csv"
        results_path.write_text("col\n")

        existing_df = pd.DataFrame({"response": ["text"], "error": [None]})

        with (
            patch("llm_response_collector.pipeline.load_prompts", return_value=mock_prompts_df),
            patch("llm_response_collector.pipeline.load_models_config"),
            patch("llm_response_collector.pipeline.load_clients", return_value=(0, MagicMock(), MagicMock(), {})),
            patch("llm_response_collector.pipeline.run_pipeline", return_value=[]),
            patch("llm_response_collector.pipeline.flush_checkpoint"),
            patch("llm_response_collector.pipeline.N_RUNS", 1),
            patch("llm_response_collector.pipeline.MODELS_CONFIG", {}),
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", []),
            patch("llm_response_collector.pipeline.read_llm_results", return_value=existing_df) as mock_read,
            patch("llm_response_collector.pipeline.save_results") as mock_save,
        ):
            collector_pipeline(prompts_path, results_path)

        mock_read.assert_called_once()
        mock_save.assert_called_once()

    def test_calls_run_pipeline_once_per_n_runs(self, tmp_path, mock_prompts_df):
        prompts_path = tmp_path / "prompts.csv"
        prompts_path.write_text("x\n")
        results_path = tmp_path / "results.csv"

        with (
            patch("llm_response_collector.pipeline.load_prompts", return_value=mock_prompts_df),
            patch("llm_response_collector.pipeline.load_models_config"),
            patch("llm_response_collector.pipeline.load_clients", return_value=(0, MagicMock(), MagicMock(), {})),
            patch("llm_response_collector.pipeline.run_pipeline", return_value=[]) as mock_run,
            patch("llm_response_collector.pipeline.flush_checkpoint"),
            patch("llm_response_collector.pipeline.N_RUNS", 3),
            patch("llm_response_collector.pipeline.MODELS_CONFIG", {}),
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", []),
        ):
            collector_pipeline(prompts_path, results_path)

        assert mock_run.call_count == 3


class TestRunPipeline:
    @pytest.fixture
    def df_prompts(self):
        return pd.DataFrame(
            {
                "prompt_id": ["1", "2"],
                "category": ["definition", "explanation"],
                "prompt_en": ["q1", "q2"],
                "prompt_pl": ["p1", "p2"],
                "paraphrase_en": ["q1p", "q2p"],
                "paraphrase_pl": ["p1p", "p2p"],
            }
        )

    def test_returns_checkpoint_buffer(self, df_prompts):
        models_config = {"model-a": {}}

        with (
            patch("llm_response_collector.pipeline.MODELS_CONFIG", models_config),
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["prompt_en"]),
            patch("llm_response_collector.pipeline.model_pipeline", return_value=[]),
        ):
            result = run_pipeline(0, [], df_prompts, 0, MagicMock(), MagicMock(), {}, [], Path("/tmp/out"), 1, 10)

        assert isinstance(result, list)

    def test_calls_model_pipeline_for_each_prompt_col_and_model(self, df_prompts):
        models_config = {"model-a": {}, "model-b": {}}
        prompt_columns = ["prompt_en", "prompt_pl"]

        with (
            patch("llm_response_collector.pipeline.MODELS_CONFIG", models_config),
            patch("llm_response_collector.pipeline.PROMPT_COLUMNS", prompt_columns),
            patch("llm_response_collector.pipeline.model_pipeline", return_value=[]) as mock_mp,
        ):
            run_pipeline(0, [], df_prompts, 0, MagicMock(), MagicMock(), {}, [], Path("/tmp/out"), 1, 10)

        expected_calls = len(df_prompts) * len(prompt_columns) * len(models_config)

        assert mock_mp.call_count == expected_calls

    #
    # def test_language_is_en_for_en_column(self, df_prompts):
    #     with (
    #         patch("llm_response_collector.pipeline.MODELS_CONFIG", {"m": {}}),
    #         patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["prompt_en"]),
    #         patch("llm_response_collector.pipeline.model_pipeline", return_value=[]) as mock_mp,
    #     ):
    #         run_pipeline(0, [], df_prompts, 0, MagicMock(), MagicMock(), {}, [], Path("/tmp/out"), 1, 10)
    #
    #     lang_args = [c.kwargs["lang"] for c in mock_mp.call_args_list]
    #
    #     assert all(lang == "en" for lang in lang_args)
    #
    # def test_language_is_pl_for_pl_column(self, df_prompts):
    #     with (
    #         patch("llm_response_collector.pipeline.MODELS_CONFIG", {"m": {}}),
    #         patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["prompt_pl"]),
    #         patch("llm_response_collector.pipeline.model_pipeline", return_value=[]) as mock_mp,
    #     ):
    #         run_pipeline(0, [], df_prompts, 0, MagicMock(), MagicMock(), {}, [], Path("/tmp/out"), 1, 10)
    #
    #     lang_args = [c.kwargs["lang"] for c in mock_mp.call_args_list]
    #     assert all(lang == "pl" for lang in lang_args)
    #
    # def test_is_paraphrase_true_for_paraphrase_column(self, df_prompts):
    #     with (
    #         patch("llm_response_collector.pipeline.MODELS_CONFIG", {"m": {}}),
    #         patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["paraphrase_en"]),
    #         patch("llm_response_collector.pipeline.model_pipeline", return_value=[]) as mock_mp,
    #     ):
    #         run_pipeline(0, [], df_prompts, 0, MagicMock(), MagicMock(), {}, [], Path("/tmp/out"), 1, 10)
    #
    #     paraphrase_args = [c.kwargs["is_paraphrase"] for c in mock_mp.call_args_list]
    #     assert all(is_par is True for is_par in paraphrase_args)
    #
    # def test_is_paraphrase_false_for_prompt_column(self, df_prompts):
    #     with (
    #         patch("llm_response_collector.pipeline.MODELS_CONFIG", {"m": {}}),
    #         patch("llm_response_collector.pipeline.PROMPT_COLUMNS", ["prompt_en"]),
    #         patch("llm_response_collector.pipeline.model_pipeline", return_value=[]) as mock_mp,
    #     ):
    #         run_pipeline(0, [], df_prompts, 0, MagicMock(), MagicMock(), {}, [], Path("/tmp/out"), 1, 10)
    #
    #     paraphrase_args = [c.kwargs["is_paraphrase"] for c in mock_mp.call_args_list]
    #     assert all(is_par is False for is_par in paraphrase_args)
