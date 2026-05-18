from pathlib import Path

from config import (
    TEST_PROMPTS,
    TRAIN_PROMPTS,
    VAL_PROMPTS,
    LLM_RESULTS_TRAIN_PROMPTS_BASE_PATH,
    TEMPORARY_RESPONSES_DIR,
    LLM_RESULTS_VAL_PROMPTS_BASE_PATH,
    LLM_RESULTS_TEST_PROMPTS_BASE_PATH,
    RUN_MODE,
    LLM_RESULTS_TRAIN_PROMPTS,
    LLM_RESULTS_VAL_PROMPTS,
    LLM_RESULTS_TEST_PROMPTS,
)
from llm_behavior_xai.llm_response_collector.pipeline import get_timestamp, collector_pipeline


def main() -> None:
    if RUN_MODE == "full":
        current_timestamp = get_timestamp()

        llm_results_paths = [
            (
                TRAIN_PROMPTS,
                TEMPORARY_RESPONSES_DIR
                / Path(f"{str(LLM_RESULTS_TRAIN_PROMPTS_BASE_PATH.stem)}_{current_timestamp}.csv"),
            ),
            (
                VAL_PROMPTS,
                TEMPORARY_RESPONSES_DIR
                / Path(f"{str(LLM_RESULTS_VAL_PROMPTS_BASE_PATH.stem)}_{current_timestamp}.csv"),
            ),
            (
                TEST_PROMPTS,
                TEMPORARY_RESPONSES_DIR
                / Path(f"{str(LLM_RESULTS_TEST_PROMPTS_BASE_PATH.stem)}_{current_timestamp}.csv"),
            ),
        ]
    else:
        llm_results_paths = [
            (TRAIN_PROMPTS, LLM_RESULTS_TRAIN_PROMPTS),
            (VAL_PROMPTS, LLM_RESULTS_VAL_PROMPTS),
            (TEST_PROMPTS, LLM_RESULTS_TEST_PROMPTS),
        ]

    for prompts_path, results_path in llm_results_paths:
        collector_pipeline(prompts_path, results_path)


if __name__ == "__main__":
    main()
