from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[0]

DATA_DIR = PROJ_ROOT / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
LLM_RESULTS_EXAMPLE_PROMPTS = PROCESSED_DATA_DIR / "llm_results.csv"
LLM_RESULTS_EXAMPLE_PROMPTS_1 = PROCESSED_DATA_DIR / "llm_results (1).csv"
LLM_RESULTS_EXAMPLE_PROMPTS_2 = PROCESSED_DATA_DIR / "llm_results (2).csv"
LLM_RESULTS_TEST_PROMPTS = PROCESSED_DATA_DIR / "llm_results_20260423_test.csv"

PROMPTS_DIR = DATA_DIR / "prompts"
TEST_PROMPTS = PROMPTS_DIR / "test_prompts.csv"
TRAIN_PROMPTS = PROMPTS_DIR / "train_prompts.csv"
VAL_PROMPTS = PROMPTS_DIR / "valid_prompts.csv"

RAW_DATA_DIR = DATA_DIR / "raw"
EXAMPLE_PROMPTS = RAW_DATA_DIR / "prompts.csv"
