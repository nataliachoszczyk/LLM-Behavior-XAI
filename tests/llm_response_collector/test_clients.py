from unittest.mock import MagicMock, patch

import pytest

from llm_behavior_xai.llm_response_collector.clients import (
    get_gemini_client,
    get_groq_client,
    load_local_model,
    load_clients,
)


class TestGetGeminiClient:
    @patch("llm_behavior_xai.llm_response_collector.clients.genai.Client")
    def test_returns_genai_client(self, mock_client_cls):
        mock_instance = MagicMock()
        mock_client_cls.return_value = mock_instance

        result = get_gemini_client("test-api-key")

        assert result is mock_instance

    @patch("llm_behavior_xai.llm_response_collector.clients.genai.Client")
    def test_passes_api_key_to_client(self, mock_client_cls):
        get_gemini_client("my-api-key")

        mock_client_cls.assert_called_once_with(api_key="my-api-key")

    @patch("llm_behavior_xai.llm_response_collector.clients.genai.Client")
    def test_accepts_none_api_key(self, mock_client_cls):
        get_gemini_client(None)

        mock_client_cls.assert_called_once_with(api_key=None)

    @patch("llm_behavior_xai.llm_response_collector.clients.genai.Client")
    def test_prints_initialized_message(self, mock_client_cls, capsys):
        get_gemini_client("key")

        captured = capsys.readouterr()
        assert "Gemini initialized" in captured.out


class TestGetGroqClient:
    @patch("llm_behavior_xai.llm_response_collector.clients.Groq")
    def test_returns_groq_client(self, mock_groq_cls):
        mock_instance = MagicMock()
        mock_groq_cls.return_value = mock_instance

        result = get_groq_client("test-api-key")

        assert result is mock_instance

    @patch("llm_behavior_xai.llm_response_collector.clients.Groq")
    def test_passes_api_key_to_client(self, mock_groq_cls):
        get_groq_client("my-groq-key")

        mock_groq_cls.assert_called_once_with(api_key="my-groq-key")

    @patch("llm_behavior_xai.llm_response_collector.clients.Groq")
    def test_accepts_none_api_key(self, mock_groq_cls):
        get_groq_client(None)

        mock_groq_cls.assert_called_once_with(api_key=None)

    @patch("llm_behavior_xai.llm_response_collector.clients.Groq")
    def test_prints_initialized_message(self, mock_groq_cls, capsys):
        get_groq_client("key")

        captured = capsys.readouterr()
        assert "Groq initialized" in captured.out


class TestLoadLocalModel:
    @pytest.fixture(autouse=True)
    def mock_transformers(self):
        with (
            patch("llm_behavior_xai.llm_response_collector.clients.BitsAndBytesConfig") as mock_bnb,
            patch("llm_behavior_xai.llm_response_collector.clients.AutoTokenizer") as mock_tokenizer_cls,
            patch("llm_behavior_xai.llm_response_collector.clients.AutoConfig") as mock_config_cls,
            patch("llm_behavior_xai.llm_response_collector.clients.AutoModelForCausalLM") as mock_model_cls,
        ):
            self.mock_bnb = mock_bnb
            self.mock_tokenizer = MagicMock()
            self.mock_config = MagicMock()
            self.mock_model = MagicMock()
            mock_tokenizer_cls.from_pretrained.return_value = self.mock_tokenizer
            mock_config_cls.from_pretrained.return_value = self.mock_config
            mock_model_cls.from_pretrained.return_value = self.mock_model
            yield

    def test_returns_dict_with_model_and_tokenizer(self):
        result = load_local_model("some/model-id", "mistral-7b-hf")

        assert isinstance(result, dict)
        assert "model" in result
        assert "tokenizer" in result

    def test_returned_model_is_from_pretrained(self):
        result = load_local_model("some/model-id", "mistral-7b-hf")

        assert result["model"] is self.mock_model

    def test_returned_tokenizer_is_from_pretrained(self):
        result = load_local_model("some/model-id", "mistral-7b-hf")

        assert result["tokenizer"] is self.mock_tokenizer

    def test_model_eval_is_called(self):
        load_local_model("some/model-id", "mistral-7b-hf")

        self.mock_model.eval.assert_called_once()

    def test_trust_remote_code_true_for_non_phi_model(self):
        with patch("llm_behavior_xai.llm_response_collector.clients.AutoTokenizer") as mock_tok:
            with patch("llm_behavior_xai.llm_response_collector.clients.AutoConfig") as mock_cfg:
                with patch("llm_behavior_xai.llm_response_collector.clients.AutoModelForCausalLM") as mock_mdl:
                    mock_cfg.from_pretrained.return_value = MagicMock()
                    mock_mdl.from_pretrained.return_value = MagicMock()
                    load_local_model("some/model-id", "mistral-7b-hf")

                    mock_tok.from_pretrained.assert_called_once_with(
                        "some/model-id", token=None, trust_remote_code=True
                    )

    def test_trust_remote_code_false_for_phi_model(self):
        with patch("llm_behavior_xai.llm_response_collector.clients.AutoTokenizer") as mock_tok:
            with patch("llm_behavior_xai.llm_response_collector.clients.AutoConfig") as mock_cfg:
                with patch("llm_behavior_xai.llm_response_collector.clients.AutoModelForCausalLM") as mock_mdl:
                    mock_config_instance = MagicMock()
                    mock_config_instance.rope_parameters = {}
                    mock_cfg.from_pretrained.return_value = mock_config_instance
                    mock_mdl.from_pretrained.return_value = MagicMock()
                    load_local_model("microsoft/Phi-3-mini", "phi-3-mini-hf")

                    mock_tok.from_pretrained.assert_called_once_with(
                        "microsoft/Phi-3-mini", token=None, trust_remote_code=False
                    )

    def test_phi_model_sets_rope_type_default(self):
        with patch("llm_behavior_xai.llm_response_collector.clients.AutoTokenizer"):
            with patch("llm_behavior_xai.llm_response_collector.clients.AutoConfig") as mock_cfg:
                with patch("llm_behavior_xai.llm_response_collector.clients.AutoModelForCausalLM") as mock_mdl:
                    mock_config_instance = MagicMock()
                    mock_config_instance.rope_parameters = {}
                    mock_cfg.from_pretrained.return_value = mock_config_instance
                    mock_mdl.from_pretrained.return_value = MagicMock()

                    load_local_model("microsoft/Phi-3-mini", "phi-3-mini-hf")

                    assert mock_config_instance.rope_parameters["rope_type"] == "default"

    def test_phi_model_sets_attn_implementation_eager(self):
        with patch("llm_behavior_xai.llm_response_collector.clients.AutoTokenizer"):
            with patch("llm_behavior_xai.llm_response_collector.clients.AutoConfig") as mock_cfg:
                with patch("llm_behavior_xai.llm_response_collector.clients.AutoModelForCausalLM") as mock_mdl:
                    mock_config_instance = MagicMock()
                    mock_config_instance.rope_parameters = {}
                    mock_cfg.from_pretrained.return_value = mock_config_instance
                    mock_mdl.from_pretrained.return_value = MagicMock()

                    load_local_model("microsoft/Phi-3-mini", "phi-3-mini-hf")

                    assert mock_config_instance._attn_implementation == "eager"

    def test_phi_model_initializes_rope_parameters_when_not_dict(self):
        with patch("llm_behavior_xai.llm_response_collector.clients.AutoTokenizer"):
            with patch("llm_behavior_xai.llm_response_collector.clients.AutoConfig") as mock_cfg:
                with patch("llm_behavior_xai.llm_response_collector.clients.AutoModelForCausalLM") as mock_mdl:
                    mock_config_instance = MagicMock(spec=[])
                    mock_config_instance.rope_parameters = "not-a-dict"
                    mock_cfg.from_pretrained.return_value = mock_config_instance
                    mock_mdl.from_pretrained.return_value = MagicMock()

                    load_local_model("microsoft/Phi-3-mini", "phi-3-mini-hf")

                    assert mock_config_instance.rope_parameters["rope_type"] == "default"

    def test_passes_hf_token_to_pretrained_calls(self):
        with patch("llm_behavior_xai.llm_response_collector.clients.AutoTokenizer") as mock_tok:
            with patch("llm_behavior_xai.llm_response_collector.clients.AutoConfig") as mock_cfg:
                with patch("llm_behavior_xai.llm_response_collector.clients.AutoModelForCausalLM") as mock_mdl:
                    mock_cfg.from_pretrained.return_value = MagicMock()
                    mock_mdl.from_pretrained.return_value = MagicMock()

                    load_local_model("some/model-id", "mistral-7b-hf", hf_token="hf-secret")

                    mock_tok.from_pretrained.assert_called_once_with(
                        "some/model-id", token="hf-secret", trust_remote_code=True
                    )
                    mock_cfg.from_pretrained.assert_called_once_with(
                        "some/model-id", token="hf-secret", trust_remote_code=True
                    )

    def test_prints_loading_and_loaded_messages(self, capsys):
        load_local_model("some/model-id", "mistral-7b-hf")

        captured = capsys.readouterr()
        assert "some/model-id" in captured.out
        assert "mistral-7b-hf" in captured.out


class TestLoadClients:
    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        with (
            patch("llm_behavior_xai.llm_response_collector.clients.get_gemini_client") as mock_gemini,
            patch("llm_behavior_xai.llm_response_collector.clients.get_groq_client") as mock_groq,
            patch("llm_behavior_xai.llm_response_collector.clients.load_local_model") as mock_local,
            patch("llm_behavior_xai.llm_response_collector.clients.GEMINI_API_KEYS", ["key-0", "key-1"]),
            patch("llm_behavior_xai.llm_response_collector.clients.GROQ_API_KEY", "groq-key"),
            patch(
                "llm_behavior_xai.llm_response_collector.clients.MODELS_CONFIG",
                {
                    "mistral-7b-hf": {"model_id": "mistralai/Mistral-7B-Instruct-v0.3"},
                    "phi-3-mini-hf": {"model_id": "microsoft/Phi-3-mini-4k-instruct"},
                },
            ),
        ):
            self.mock_gemini = mock_gemini
            self.mock_groq = mock_groq
            self.mock_local = mock_local
            self.mock_gemini.return_value = MagicMock(name="gemini_client")
            self.mock_groq.return_value = MagicMock(name="groq_client")
            self.mock_local.return_value = {"model": MagicMock(), "tokenizer": MagicMock()}
            yield

    def test_returns_four_element_tuple(self):
        result = load_clients()

        assert len(result) == 4

    def test_first_element_is_zero_index(self):
        gemini_api_key_index, *_ = load_clients()

        assert gemini_api_key_index == 0

    def test_second_element_is_gemini_client(self):
        _, gemini_client, _, _ = load_clients()

        assert gemini_client is self.mock_gemini.return_value

    def test_third_element_is_groq_client(self):
        _, _, groq_client, _ = load_clients()

        assert groq_client is self.mock_groq.return_value

    def test_gemini_client_initialized_with_first_key(self):
        load_clients()

        self.mock_gemini.assert_called_once_with("key-0")

    def test_groq_client_initialized_with_groq_key(self):
        load_clients()

        self.mock_groq.assert_called_once_with("groq-key")

    def test_local_models_dict_contains_both_models(self):
        _, _, _, local_models = load_clients()

        assert "mistral-7b-hf" in local_models
        assert "phi-3-mini-hf" in local_models

    def test_load_local_model_called_for_each_local_model(self):
        load_clients()

        assert self.mock_local.call_count == 2
        self.mock_local.assert_any_call("mistralai/Mistral-7B-Instruct-v0.3", "mistral-7b-hf")
        self.mock_local.assert_any_call("microsoft/Phi-3-mini-4k-instruct", "phi-3-mini-hf")
