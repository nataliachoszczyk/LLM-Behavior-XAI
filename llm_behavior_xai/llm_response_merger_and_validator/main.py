from llm_behavior_xai.config import (
    LLM_RESULTS_TEST_PROMPTS_1,
    LLM_RESULTS_VAL_PROMPTS_1,
    LLM_RESULTS_TRAIN_PROMPTS_1,
    LLM_RESULTS_TRAIN_PROMPTS_1_64,
    LLM_RESULTS_TRAIN_PROMPTS_65_120,
    LLM_RESULTS_TRAIN_PROMPTS_GROQ,
    LLM_RESULTS_TRAIN_PROMPTS_GROQ_LAST_2,
    LLM_RESULTS_TRAIN_PROMPTS_GEMINI,
    LLM_RESULTS_VAL_PROMPTS_GEMINI,
    LLM_RESULTS_TEST_PROMPTS_GEMINI,
    FINAL_RESPONSES_DIR,
)
from llm_behavior_xai.llm_response_merger_and_validator.clean_dataset import clean_dataframe
from llm_behavior_xai.llm_response_merger_and_validator.merge_datasets import load_and_merge_datasets
from llm_behavior_xai.llm_response_merger_and_validator.validate_dataset import (
    analyze_df,
    print_dataframe_report,
    print_summary_dataframe,
)


def main() -> None:
    temporary_datasets = [
        (
            "train",
            [
                LLM_RESULTS_TRAIN_PROMPTS_1,
                LLM_RESULTS_TRAIN_PROMPTS_1_64,
                LLM_RESULTS_TRAIN_PROMPTS_65_120,
                LLM_RESULTS_TRAIN_PROMPTS_GROQ,
                LLM_RESULTS_TRAIN_PROMPTS_GROQ_LAST_2,
                LLM_RESULTS_TRAIN_PROMPTS_GEMINI,
            ],
        ),
        ("val", [LLM_RESULTS_VAL_PROMPTS_1, LLM_RESULTS_VAL_PROMPTS_GEMINI]),
        ("test", [LLM_RESULTS_TEST_PROMPTS_1, LLM_RESULTS_TEST_PROMPTS_GEMINI]),
    ]

    merged_dfs = load_and_merge_datasets(temporary_datasets)

    reports = {}
    output_paths = {}

    for name, df in merged_dfs.items():
        if df.empty:
            print(f"Skipping analysis for empty {name}")
            continue

        deduped, removed_duplicates, removed_errors = clean_dataframe(df)

        out_path = FINAL_RESPONSES_DIR / f"llm_results_{name}_full.csv"
        deduped.to_csv(out_path, index=False)

        output_paths[name] = out_path

        print(
            f"{name}: removed error rows={removed_errors}, removed duplicate rows={removed_duplicates},",
            f"saved={out_path.name}, rows={len(deduped)}",
        )

        r = analyze_df(deduped, name)
        reports[name] = r
        print_dataframe_report(name, r)

    print_summary_dataframe(output_paths, reports)


if __name__ == "__main__":
    main()
