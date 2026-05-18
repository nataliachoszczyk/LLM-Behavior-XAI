import math
from unittest.mock import MagicMock, patch

import pytest

from llm_response_collector.query_llm import (
    summarize_logprobs,
    query_gemini,
    query_groq,
    query_local_hf,
    query_model,
)


class TestSummarizeLogprobs:
    def test_empty_list_returns_logprob_unavailable(self):
        result = summarize_logprobs([])

        assert result["logprob_available"] is False

    def test_empty_list_returns_none_fields(self):
        result = summarize_logprobs([])

        assert result["sum_logprob"] is None
        assert result["avg_logprob"] is None
        assert result["perplexity"] is None

    def test_empty_list_returns_zero_generated_tokens(self):
        result = summarize_logprobs([])

        assert result["generated_tokens"] == 0

    def test_non_empty_sets_logprob_available_true(self):
        result = summarize_logprobs([-1.0, -2.0])

        assert result["logprob_available"] is True

    def test_sum_logprob_is_correct(self):
        result = summarize_logprobs([-1.0, -2.0, -3.0])

        assert result["sum_logprob"] == pytest.approx(-6.0, abs=1e-5)

    def test_avg_logprob_is_correct(self):
        result = summarize_logprobs([-1.0, -3.0])

        assert result["avg_logprob"] == pytest.approx(-2.0, abs=1e-5)

    def test_generated_tokens_count_is_correct(self):
        result = summarize_logprobs([-0.5, -1.0, -1.5])

        assert result["generated_tokens"] == 3

    def test_perplexity_is_exp_of_negative_avg(self):
        logprobs = [-1.0, -3.0]
        avg = sum(logprobs) / len(logprobs)
        expected_ppl = math.exp(-avg)

        result = summarize_logprobs(logprobs)

        assert result["perplexity"] == pytest.approx(expected_ppl, abs=1e-4)

    def test_single_logprob(self):
        result = summarize_logprobs([-2.0])

        assert result["sum_logprob"] == pytest.approx(-2.0, abs=1e-5)
        assert result["avg_logprob"] == pytest.approx(-2.0, abs=1e-5)
        assert result["generated_tokens"] == 1

    def test_returns_all_expected_keys(self):
        result = summarize_logprobs([-1.0])

        assert set(result.keys()) == {
            "logprob_available",
            "sum_logprob",
            "avg_logprob",
            "generated_tokens",
            "perplexity",
        }


class TestQueryGemini:
    @pytest.fixture
    def config(self):
        return {
            "temperature": 0.3,
            "max_tokens": 512,
            "top_p": 0.9,
            "model_id": "gemini-flash-latest",
        }

    @pytest.fixture
    def gemini_client(self):
        client = MagicMock()
        response = MagicMock()
        response.text = "This is the answer."
        response.candidates = [MagicMock(finish_reason="STOP")]
        client.models.generate_content.return_value = response
        return client

    def test_returns_text_on_success(self, config, gemini_client):
        text, error, _ = query_gemini("prompt", config, gemini_client)

        assert text == "This is the answer."
        assert error is None

    def test_returns_logprob_stats_dict(self, config, gemini_client):
        _, _, stats = query_gemini("prompt", config, gemini_client)

        assert isinstance(stats, dict)
        assert "logprob_available" in stats

    def test_empty_response_text_returns_none(self, config, gemini_client):
        gemini_client.models.generate_content.return_value.text = ""

        text, error, _ = query_gemini("prompt", config, gemini_client)

        assert text is None
        assert error is not None

    def test_none_candidates_returns_error(self, config, gemini_client):
        gemini_client.models.generate_content.return_value.candidates = None

        text, error, _ = query_gemini("prompt", config, gemini_client)

        assert text is None
        assert error is not None

    def test_non_stop_finish_reason_still_returns_text(self, config, gemini_client, capsys):
        gemini_client.models.generate_content.return_value.candidates[0].finish_reason = "MAX_TOKENS"

        text, error, _ = query_gemini("prompt", config, gemini_client)

        assert text == "This is the answer."
        assert error is None
        assert "finish_reason" in capsys.readouterr().out

    def test_not_found_error_tries_next_model_candidate(self, config):
        client = MagicMock()
        success_response = MagicMock()
        success_response.text = "ok"
        success_response.candidates = [MagicMock(finish_reason="STOP")]

        client.models.generate_content.side_effect = [
            Exception("NOT_FOUND: model not found"),
            success_response,
        ]

        text, error, _ = query_gemini("prompt", config, client)

        assert text == "ok"
        assert error is None
        assert client.models.generate_content.call_count == 2

    def test_non_not_found_error_returns_immediately(self, config):
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("PERMISSION_DENIED")

        text, error, _ = query_gemini("prompt", config, client)

        assert text is None
        assert "PERMISSION_DENIED" in error
        assert client.models.generate_content.call_count == 1

    def test_all_candidates_fail_returns_last_error(self, config):
        client = MagicMock()
        client.models.generate_content.side_effect = Exception("NOT_FOUND: model not found")

        text, error, _ = query_gemini("prompt", config, client)

        assert text is None
        assert error is not None

    def test_model_id_with_models_prefix_is_deduplicated(self, config):
        config["model_id"] = "models/gemini-flash-latest"
        client = MagicMock()
        response = MagicMock()
        response.text = "hello"
        response.candidates = [MagicMock(finish_reason="STOP")]
        client.models.generate_content.return_value = response

        text, _, _ = query_gemini("prompt", config, client)

        assert text == "hello"
        first_call_model = client.models.generate_content.call_args_list[0][1]["model"]
        assert first_call_model == "models/gemini-flash-latest"

    def test_strips_whitespace_from_response(self, config, gemini_client):
        gemini_client.models.generate_content.return_value.text = "  answer  "

        text, _, _ = query_gemini("prompt", config, gemini_client)

        assert text == "answer"


class TestQueryGroq:
    @pytest.fixture
    def config(self):
        return {
            "model_id": "llama-3.1-8b-instant",
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 512,
        }

    @pytest.fixture
    def groq_client(self):
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = "Groq answer"
        choice.logprobs = None
        completion = MagicMock()
        completion.choices = [choice]
        client.chat.completions.create.return_value = completion
        return client

    def test_returns_response_text(self, config, groq_client):
        text, error, _ = query_groq("prompt", config, groq_client)

        assert text == "Groq answer"
        assert error is None

    def test_returns_logprob_stats_dict(self, config, groq_client):
        _, _, stats = query_groq("prompt", config, groq_client)

        assert isinstance(stats, dict)
        assert "logprob_available" in stats

    def test_strips_whitespace_from_response(self, config, groq_client):
        groq_client.chat.completions.create.return_value.choices[0].message.content = "  answer  "

        text, _, _ = query_groq("prompt", config, groq_client)

        assert text == "answer"

    def test_logprob_error_retries_without_logprobs(self, config, groq_client):
        choice = MagicMock()
        choice.message.content = "fallback"
        choice.logprobs = None
        fallback_completion = MagicMock()
        fallback_completion.choices = [choice]

        groq_client.chat.completions.create.side_effect = [
            Exception("logprob not supported"),
            fallback_completion,
        ]

        text, error, _ = query_groq("prompt", config, groq_client)

        assert text == "fallback"
        assert error is None
        assert groq_client.chat.completions.create.call_count == 2

    def test_logprob_error_retry_omits_logprob_keys(self, config, groq_client):
        fallback_completion = MagicMock()
        fallback_completion.choices[0].message.content = "ok"
        fallback_completion.choices[0].logprobs = None

        groq_client.chat.completions.create.side_effect = [
            Exception("logprob not supported"),
            fallback_completion,
        ]

        query_groq("prompt", config, groq_client)

        second_call_kwargs = groq_client.chat.completions.create.call_args_list[1][1]
        assert "logprobs" not in second_call_kwargs
        assert "top_logprobs" not in second_call_kwargs

    def test_non_logprob_error_is_propagated(self, config, groq_client):
        groq_client.chat.completions.create.side_effect = Exception("network error")

        text, error, _ = query_groq("prompt", config, groq_client)

        assert text is None
        assert "network error" in error

    def test_with_logprobs_in_response(self, config, groq_client):
        lp_item = MagicMock()
        lp_item.logprob = -1.5
        choice = groq_client.chat.completions.create.return_value.choices[0]
        choice.logprobs = MagicMock()
        choice.logprobs.content = [lp_item]

        _, _, stats = query_groq("prompt", config, groq_client)

        assert stats["logprob_available"] is True
        assert stats["generated_tokens"] == 1

    def test_logprob_item_with_none_value_is_skipped(self, config, groq_client):
        lp_item = MagicMock()
        lp_item.logprob = None
        choice = groq_client.chat.completions.create.return_value.choices[0]
        choice.logprobs = MagicMock()
        choice.logprobs.content = [lp_item]

        _, _, stats = query_groq("prompt", config, groq_client)

        assert stats["logprob_available"] is False

    def test_request_includes_logprobs_true(self, config, groq_client):
        query_groq("prompt", config, groq_client)

        call_kwargs = groq_client.chat.completions.create.call_args_list[0][1]
        assert call_kwargs["logprobs"] is True

    def test_request_includes_user_message(self, config, groq_client):
        query_groq("my prompt", config, groq_client)

        call_kwargs = groq_client.chat.completions.create.call_args_list[0][1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "my prompt"}]


class TestQueryLocalHf:
    @pytest.fixture
    def config(self):
        return {
            "max_new_tokens": 128,
            "temperature": 0.3,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        }

    @pytest.fixture
    def local_bundle(self):
        import torch

        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<s>prompt</s>"
        tokenizer.eos_token_id = 2
        tokenizer.decode.return_value = "local model answer"

        input_ids = torch.tensor([[1, 2, 3]])
        tokenizer.return_value = {"input_ids": input_ids}
        tokenizer.side_effect = None

        model = MagicMock()
        model.device = "cpu"

        generated_ids = torch.tensor([4, 5])
        score_0 = torch.zeros(1, 100)
        score_1 = torch.zeros(1, 100)
        score_0[0, 4] = 1.0
        score_1[0, 5] = 1.0

        generated_output = MagicMock()
        generated_output.sequences = [torch.cat([input_ids[0], generated_ids])]
        generated_output.scores = [score_0, score_1]
        model.generate.return_value = generated_output

        return {"model": model, "tokenizer": tokenizer}

    def test_returns_response_text(self, config, local_bundle):
        local_models = {"mistral-7b-hf": local_bundle}

        text, error, _ = query_local_hf("prompt", "mistral-7b-hf", config, local_models)

        assert error is None
        assert isinstance(text, str)

    def test_returns_logprob_stats(self, config, local_bundle):
        local_models = {"mistral-7b-hf": local_bundle}

        _, _, stats = query_local_hf("prompt", "mistral-7b-hf", config, local_models)

        assert isinstance(stats, dict)
        assert "logprob_available" in stats

    def test_missing_model_key_returns_error(self, config):
        text, error, _ = query_local_hf("prompt", "unknown-model", config, {})

        assert text is None
        assert error is not None

    def test_none_local_models_defaults_to_empty_dict(self, config):
        text, error, _ = query_local_hf("prompt", "some-model", config, None)

        assert text is None
        assert error is not None

    def test_apply_chat_template_called_with_user_message(self, config, local_bundle):
        local_models = {"mistral-7b-hf": local_bundle}

        query_local_hf("hello world", "mistral-7b-hf", config, local_models)

        local_bundle["tokenizer"].apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "hello world"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def test_model_generate_called_with_config_params(self, config, local_bundle):
        local_models = {"mistral-7b-hf": local_bundle}

        query_local_hf("prompt", "mistral-7b-hf", config, local_models)

        call_kwargs = local_bundle["model"].generate.call_args[1]
        assert call_kwargs["max_new_tokens"] == config["max_new_tokens"]
        assert call_kwargs["temperature"] == config["temperature"]
        assert call_kwargs["do_sample"] is True
        assert call_kwargs["return_dict_in_generate"] is True
        assert call_kwargs["output_scores"] is True

    def test_response_text_is_stripped(self, config, local_bundle):
        local_bundle["tokenizer"].decode.return_value = "  answer  "
        local_models = {"mistral-7b-hf": local_bundle}

        text, _, _ = query_local_hf("prompt", "mistral-7b-hf", config, local_models)

        assert text == "answer"


class TestQueryModel:
    @pytest.fixture(autouse=True)
    def patch_models_config(self):
        models_config = {
            "gemini-flash-latest": {
                "provider": "Google Gemini API",
                "model_id": "gemini-flash-latest",
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 512,
                "type": "api",
            },
            "llama-3.1-8b-groq": {
                "provider": "Groq API",
                "model_id": "llama-3.1-8b-instant",
                "temperature": 0.3,
                "top_p": 0.9,
                "max_tokens": 512,
                "type": "api",
            },
            "mistral-7b-hf": {
                "provider": "HuggingFace (local GPU)",
                "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
                "temperature": 0.3,
                "top_p": 0.9,
                "max_new_tokens": 512,
                "repetition_penalty": 1.1,
                "type": "local",
            },
        }
        with patch("llm_response_collector.query_llm.MODELS_CONFIG", models_config):
            yield

    def test_returns_four_element_tuple(self):
        from google import genai

        client = MagicMock(spec=genai.Client)
        with patch("llm_response_collector.query_llm.query_gemini", return_value=("text", None, {})):
            result = query_model("gemini-flash-latest", "prompt", client=client)

        assert len(result) == 4

    def test_unknown_model_returns_error(self):
        with patch("llm_response_collector.query_llm.MODELS_CONFIG", {"unknown": {"provider": "X", "type": "x"}}):
            _, error, _, _ = query_model("unknown", "prompt")

        assert error is not None
        assert "Unknown model" in error

    def test_gemini_with_wrong_client_type_returns_error(self):
        from groq import Groq

        wrong_client = MagicMock(spec=Groq)
        _, error, elapsed, _ = query_model("gemini-flash-latest", "prompt", client=wrong_client)

        assert error == "Client not initialized"
        assert elapsed == 0

    def test_groq_with_wrong_client_type_returns_error(self):
        from google import genai

        wrong_client = MagicMock(spec=genai.Client)
        _, error, elapsed, _ = query_model("llama-3.1-8b-groq", "prompt", client=wrong_client)

        assert error == "Client not initialized"
        assert elapsed == 0

    def test_elapsed_time_is_non_negative(self):
        from google import genai

        client = MagicMock(spec=genai.Client)
        with patch("llm_response_collector.query_llm.query_gemini", return_value=("text", None, {})):
            _, _, elapsed, _ = query_model("gemini-flash-latest", "prompt", client=client)

        assert elapsed >= 0

    def test_local_models_defaults_to_empty_dict_when_none(self):
        with patch("llm_response_collector.query_llm.query_local_hf", return_value=(None, "err", {})):
            _, error, _, _ = query_model("mistral-7b-hf", "prompt", local_models=None)

        assert error is not None

    def test_gemini_dispatches_to_query_gemini_with_real_client(self):
        from google import genai

        client = object.__new__(genai.Client)

        with patch("llm_response_collector.query_llm.query_gemini", return_value=("answer", None, {})) as mock_gemini:
            response, error, _, _ = query_model("gemini-flash-latest", "my prompt", client=client)

        mock_gemini.assert_called_once()

        assert mock_gemini.call_args[0][0] == "my prompt"
        assert response == "answer"
        assert error is None

    def test_groq_dispatches_to_query_groq_with_real_client(self):
        from groq import Groq

        client = object.__new__(Groq)

        with patch("llm_response_collector.query_llm.query_groq", return_value=("groq answer", None, {})) as mock_groq:
            response, error, _, _ = query_model("llama-3.1-8b-groq", "my prompt", client=client)

        mock_groq.assert_called_once()

        assert mock_groq.call_args[0][0] == "my prompt"
        assert response == "groq answer"
        assert error is None

    def test_local_model_dispatches_to_query_local_hf_when_in_local_models(self):
        local_models = {"mistral-7b-hf": MagicMock()}

        with patch(
            "llm_response_collector.query_llm.query_local_hf", return_value=("local answer", None, {})
        ) as mock_local:
            response, error, _, _ = query_model("mistral-7b-hf", "my prompt", local_models=local_models)

        mock_local.assert_called_once()

        assert response == "local answer"
        assert error is None
