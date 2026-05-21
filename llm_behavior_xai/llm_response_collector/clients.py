from typing import Any

import torch
from google.genai import Client

from groq import Groq
from google import genai
from transformers import BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from llm_behavior_xai.config import GEMINI_API_KEYS, GROQ_API_KEY, MODELS_CONFIG


def get_gemini_client(gemini_api_key: str | None) -> genai.Client:
    gemini_client = genai.Client(api_key=gemini_api_key)
    print("Gemini initialized")

    return gemini_client


def get_groq_client(groq_api_key: str | None) -> Groq:
    groq_client = Groq(api_key=groq_api_key)
    print("Groq initialized")

    return groq_client


def load_local_model(model_id: str, model_key: str, hf_token: str | None = None) -> dict[str, Any]:
    print(f"⏳ Downloading model: {model_id} ...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    use_remote_code = model_key != "phi-3-mini-hf"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=use_remote_code,
    )

    config = AutoConfig.from_pretrained(
        model_id,
        token=hf_token,
        trust_remote_code=use_remote_code,
    )
    if model_key == "phi-3-mini-hf":
        rope_parameters = getattr(config, "rope_parameters", None)
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        rope_parameters["rope_type"] = "default"
        config.rope_parameters = rope_parameters
        config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=use_remote_code,
    )
    model.eval()

    print(f"✅ Model {model_key} loaded!")
    return {"model": model, "tokenizer": tokenizer}


def load_clients() -> tuple[int, Client, Groq, dict[Any, Any]]:
    gemini_api_key_index = 0
    gemini_client = get_gemini_client(GEMINI_API_KEYS[gemini_api_key_index])

    groq_client = get_groq_client(GROQ_API_KEY)

    local_models = {}
    local_models["mistral-7b-hf"] = load_local_model(str(MODELS_CONFIG["mistral-7b-hf"]["model_id"]), "mistral-7b-hf")
    local_models["phi-3-mini-hf"] = load_local_model(str(MODELS_CONFIG["phi-3-mini-hf"]["model_id"]), "phi-3-mini-hf")

    return gemini_api_key_index, gemini_client, groq_client, local_models
