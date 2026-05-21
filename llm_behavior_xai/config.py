import os

from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*args, **kwargs):
        return False


load_dotenv()

GEMINI_API_KEY_1 = os.getenv("GEMINI_API_KEY_1")
GEMINI_API_KEY_2 = os.getenv("GEMINI_API_KEY_2")
GEMINI_API_KEY_3 = os.getenv("GEMINI_API_KEY_3")
GEMINI_API_KEY_4 = os.getenv("GEMINI_API_KEY_4")
GEMINI_API_KEY_5 = os.getenv("GEMINI_API_KEY_5")
GEMINI_API_KEY_6 = os.getenv("GEMINI_API_KEY_6")
GEMINI_API_KEY_7 = os.getenv("GEMINI_API_KEY_7")
GEMINI_API_KEY_8 = os.getenv("GEMINI_API_KEY_8")
GEMINI_API_KEY_9 = os.getenv("GEMINI_API_KEY_9")
GEMINI_API_KEY_10 = os.getenv("GEMINI_API_KEY_10")
GEMINI_API_KEY_11 = os.getenv("GEMINI_API_KEY_11")
GEMINI_API_KEY_12 = os.getenv("GEMINI_API_KEY_12")
GEMINI_API_KEY_13 = os.getenv("GEMINI_API_KEY_13")
GEMINI_API_KEY_14 = os.getenv("GEMINI_API_KEY_14")
GEMINI_API_KEY_15 = os.getenv("GEMINI_API_KEY_15")
GEMINI_API_KEY_16 = os.getenv("GEMINI_API_KEY_16")
GEMINI_API_KEY_17 = os.getenv("GEMINI_API_KEY_17")
GEMINI_API_KEY_18 = os.getenv("GEMINI_API_KEY_18")
GEMINI_API_KEY_19 = os.getenv("GEMINI_API_KEY_19")
GEMINI_API_KEY_20 = os.getenv("GEMINI_API_KEY_20")

GEMINI_API_KEYS = [
    GEMINI_API_KEY_1,
    GEMINI_API_KEY_2,
    GEMINI_API_KEY_3,
    GEMINI_API_KEY_4,
    GEMINI_API_KEY_5,
    GEMINI_API_KEY_6,
    GEMINI_API_KEY_7,
    GEMINI_API_KEY_8,
    GEMINI_API_KEY_9,
    GEMINI_API_KEY_10,
    GEMINI_API_KEY_11,
    GEMINI_API_KEY_12,
    GEMINI_API_KEY_13,
    GEMINI_API_KEY_14,
    GEMINI_API_KEY_15,
    GEMINI_API_KEY_16,
    GEMINI_API_KEY_17,
    GEMINI_API_KEY_18,
    GEMINI_API_KEY_19,
    GEMINI_API_KEY_20,
]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

PROMPT_COLUMNS = ["prompt_en", "prompt_pl", "paraphrase_en", "paraphrase_pl"]
N_RUNS = 1
CHECKPOINT_EVERY = 10
RUN_MODE = "full"
TEST_HEAD = 2

GENERATION_DEFAULTS = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_new_tokens": 3072,
    "repetition_penalty": 1.1,
}

MODELS_CONFIG: dict[str, dict[str, str | float | int]] = {
    "gemini-flash-latest": {
        "provider": "Google Gemini API",
        "model_id": "gemini-flash-latest",
        "temperature": GENERATION_DEFAULTS["temperature"],
        "top_p": GENERATION_DEFAULTS["top_p"],
        "max_tokens": GENERATION_DEFAULTS["max_new_tokens"],
        "type": "api",
    },
    "llama-3.1-8b-groq": {
        "provider": "Groq API",
        "model_id": "llama-3.1-8b-instant",
        "temperature": GENERATION_DEFAULTS["temperature"],
        "top_p": GENERATION_DEFAULTS["top_p"],
        "max_tokens": GENERATION_DEFAULTS["max_new_tokens"],
        "type": "api",
    },
    "mistral-7b-hf": {
        "provider": "HuggingFace (local GPU)",
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "temperature": GENERATION_DEFAULTS["temperature"],
        "top_p": GENERATION_DEFAULTS["top_p"],
        "max_new_tokens": GENERATION_DEFAULTS["max_new_tokens"],
        "repetition_penalty": GENERATION_DEFAULTS["repetition_penalty"],
        "type": "local",
    },
    "phi-3-mini-hf": {
        "provider": "HuggingFace (local GPU)",
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "temperature": GENERATION_DEFAULTS["temperature"],
        "top_p": GENERATION_DEFAULTS["top_p"],
        "max_new_tokens": GENERATION_DEFAULTS["max_new_tokens"],
        "repetition_penalty": GENERATION_DEFAULTS["repetition_penalty"],
        "type": "local",
    },
}

PROJ_ROOT = Path(__file__).resolve().parents[0]

DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"

PROMPTS_DIR = DATA_DIR / "prompts"
TEST_PROMPTS = PROMPTS_DIR / "test_prompts.csv"
TRAIN_PROMPTS = PROMPTS_DIR / "train_prompts.csv"
VAL_PROMPTS = PROMPTS_DIR / "valid_prompts.csv"

RAW_DATA_DIR = DATA_DIR / "raw"
EXAMPLE_PROMPTS = RAW_DATA_DIR / "prompts.csv"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

FINAL_RESPONSES_DIR = PROCESSED_DATA_DIR / "final"

LLM_RESULTS_TRAIN_PROMPTS = FINAL_RESPONSES_DIR / "llm_results_train_full.csv"
LLM_RESULTS_VAL_PROMPTS = FINAL_RESPONSES_DIR / "llm_results_val_full.csv"
LLM_RESULTS_TEST_PROMPTS = FINAL_RESPONSES_DIR / "llm_results_test_full.csv"

TEMPORARY_RESPONSES_DIR = PROCESSED_DATA_DIR / "tmp"

LLM_RESULTS_TEST_PROMPTS_1 = TEMPORARY_RESPONSES_DIR / "llm_results_20260423_test.csv"
LLM_RESULTS_VAL_PROMPTS_1 = TEMPORARY_RESPONSES_DIR / "llm_results_20260509_valid.csv"
LLM_RESULTS_TRAIN_PROMPTS_1 = TEMPORARY_RESPONSES_DIR / "llm_results_20260511_train.csv"
LLM_RESULTS_TRAIN_PROMPTS_1_64 = TEMPORARY_RESPONSES_DIR / "llm_results_20260510_train_1-64.csv"
LLM_RESULTS_TRAIN_PROMPTS_65_120 = TEMPORARY_RESPONSES_DIR / "llm_results_20260510_train_65-120.csv"
LLM_RESULTS_TRAIN_PROMPTS_GROQ = TEMPORARY_RESPONSES_DIR / "llm_results_20260511_train_groq.csv"
LLM_RESULTS_TRAIN_PROMPTS_GROQ_LAST_2 = TEMPORARY_RESPONSES_DIR / "llm_results_20260511_train_last_2_prompts_groq.csv"
LLM_RESULTS_TRAIN_PROMPTS_GEMINI = TEMPORARY_RESPONSES_DIR / "llm_results_train_2026-05-08_23-34-10.csv"
LLM_RESULTS_VAL_PROMPTS_GEMINI = TEMPORARY_RESPONSES_DIR / "llm_results_val_2026-05-10_23-21-44.csv"
LLM_RESULTS_TEST_PROMPTS_GEMINI = TEMPORARY_RESPONSES_DIR / "llm_results_test_2026-05-11_18-13-46.csv"

LLM_RESULTS_TRAIN_PROMPTS_BASE_PATH = TEMPORARY_RESPONSES_DIR / "llm_results_train.csv"
LLM_RESULTS_VAL_PROMPTS_BASE_PATH = TEMPORARY_RESPONSES_DIR / "llm_results_val.csv"
LLM_RESULTS_TEST_PROMPTS_BASE_PATH = TEMPORARY_RESPONSES_DIR / "llm_results_test.csv"

RESPONSE_FEATURES_DIR = DATA_DIR / "response_features"

LLM_RESPONSES_TRAIN_FEATURES = RESPONSE_FEATURES_DIR / "llm_responses_train_features.csv"
LLM_RESPONSES_VAL_FEATURES = RESPONSE_FEATURES_DIR / "llm_responses_val_features.csv"
LLM_RESPONSES_TEST_FEATURES = RESPONSE_FEATURES_DIR / "llm_responses_test_features.csv"

LLM_RESPONSES_TRAIN_FEATURES_PLOTS_DIR = RESPONSE_FEATURES_DIR / "train_features_plots"
LLM_RESPONSES_VAL_FEATURES_PLOTS_DIR = RESPONSE_FEATURES_DIR / "val_features_plots"
LLM_RESPONSES_TEST_FEATURES_PLOTS_DIR = RESPONSE_FEATURES_DIR / "test_features_plots"

MODELS_DIR = PROJ_ROOT / "models"
XAI_MODELS_DIR = MODELS_DIR / "xai"

REPORTS_DIR = PROJ_ROOT / "reports"
XAI_REPORTS_DIR = REPORTS_DIR / "xai"
STYLE_PROFILES_REPORTS_DIR = REPORTS_DIR / "style_profiles"
