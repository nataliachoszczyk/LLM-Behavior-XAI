import os
import time
import datetime

import pandas as pd

from config import (
    TEST_PROMPTS,
    TRAIN_PROMPTS,
    VAL_PROMPTS,
    MODELS_CONFIG,
    GEMINI_API_KEYS,
    LLM_RESULTS_TRAIN_PROMPTS,
    LLM_RESULTS_VAL_PROMPTS,
    LLM_RESULTS_TEST_PROMPTS,
)
from file_utils import read_prompts, read_llm_results, save_results
from llm_response_collector.llm_clients import get_gemini_client
from llm_response_collector.query_llm import query_model


def get_timestamp() -> str:
    ct = datetime.datetime.now()

    return ct.strftime("%Y-%m-%d_%H-%M-%S")


def validate_prompts(df_prompts: pd.DataFrame, prompt_columns: list[str]) -> None:
    required_columns = ["prompt_id", "category", *prompt_columns]
    missing_columns = [col for col in required_columns if col not in df_prompts.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in prompts file: {missing_columns}. "
            f"Available columns: {list(df_prompts.columns)}"
        )


def clean_invalid_responses(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["response"].notna() & df["error"].isna()

    return df[mask].reset_index(drop=True)


def has_valid_response(df: pd.DataFrame, prompt_id, category, prompt_text, lang, is_paraphrase, model_key: str) -> bool:
    mask = (
        (df["prompt_id"] == prompt_id)
        & (df["category"] == category)
        & (df["prompt_text"] == prompt_text)
        & (df["language"] == lang)
        & (df["is_paraphrase"] == is_paraphrase)
        & (df["model_key"] == model_key)
        & df["response"].notna()
        & df["error"].isna()
    )
    return mask.any()


def build_parameters_string(cfg):
    """Create a stable, human-readable parameter summary from config fields."""
    ordered_keys = [
        "temperature",
        "top_p",
        "max_tokens",
        "max_new_tokens",
        "repetition_penalty",
    ]

    parts = []

    for key in ordered_keys:
        value = cfg.get(key, None)

        if value is not None:
            parts.append(f"{key}={value}")

    return ", ".join(parts)


def flush_checkpoint(rows, output_path):
    """Append buffered rows to CSV checkpoint file."""
    if not rows:
        return

    chunk_df = pd.DataFrame(rows)
    write_header = not (os.path.exists(output_path))
    chunk_df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )


def main():
    PROMPT_COLUMNS = ["prompt_en", "prompt_pl", "paraphrase_en", "paraphrase_pl"]
    N_RUNS = 1
    CHECKPOINT_EVERY = 10

    # current_timestamp = get_timestamp()

    llm_results_paths = [
        (TRAIN_PROMPTS, LLM_RESULTS_TRAIN_PROMPTS),
        (VAL_PROMPTS, LLM_RESULTS_VAL_PROMPTS),
        (TEST_PROMPTS, LLM_RESULTS_TEST_PROMPTS),
    ]

    for prompts_path, results_path in llm_results_paths:
        df_prompts = read_prompts(prompts_path)
        validate_prompts(df_prompts, PROMPT_COLUMNS)

        df_prompts = df_prompts.copy()
        df_prompts["prompt_id"] = df_prompts["prompt_id"].astype(str).str.strip()
        df_prompts["category"] = df_prompts["category"].fillna("unknown").astype(str).str.strip()
        df_prompts.loc[df_prompts["category"] == "", "category"] = "unknown"

        print(f"Loaded {len(df_prompts)} prompts from {prompts_path}")
        print("Source columns available for output: prompt_id, category")

        if results_path.exists():
            results_df = read_llm_results(results_path)
            results_df = clean_invalid_responses(results_df)
            save_results(results_df, results_path)
            print(f"Existing results loaded from {results_path} | {len(results_df)} valid responses retained")

        for cfg in MODELS_CONFIG.values():
            cfg["parameters"] = build_parameters_string(cfg)

        print("Model configuration loaded")

        for name, cfg in MODELS_CONFIG.items():
            print(f"{name} ({cfg['provider']}) | {cfg['parameters']}")

        gemini_api_key_index = 0
        gemini_client = get_gemini_client(GEMINI_API_KEYS[gemini_api_key_index])

        results = []
        checkpoint_buffer = []

        total_calls = len(df_prompts) * len(MODELS_CONFIG) * len(PROMPT_COLUMNS) * N_RUNS
        call_count = 0

        print(
            f"Pipeline start | {len(df_prompts)} prompts × {len(MODELS_CONFIG)} models × "
            f"{len(PROMPT_COLUMNS)} variants × {N_RUNS} runs = {total_calls} calls\n"
        )

        for run_id in range(1, N_RUNS + 1):
            print(f"\n🔁 Run {run_id}/{N_RUNS}")

            for _, row in df_prompts.iterrows():
                prompt_id = str(row["prompt_id"]).strip()
                category = str(row["category"]).strip() or "unknown"

                for prompt_col in PROMPT_COLUMNS:
                    prompt_text = row[prompt_col]
                    lang = "en" if "_en" in prompt_col else "pl"
                    is_paraphrase = "paraphrase" in prompt_col

                    call_count += 1

                    model_key = "gemini-flash-latest"
                    cfg = MODELS_CONFIG[model_key]

                    print(
                        f"[{call_count}/{total_calls}] run={run_id} "
                        f"prompt_id={prompt_id} | {prompt_col} | {model_key} ...",
                        end=" ",
                    )

                    if results_path.exists():
                        results_df = read_llm_results(results_path)

                        if has_valid_response(
                            results_df, prompt_id, category, prompt_text, lang, is_paraphrase, model_key
                        ):
                            continue

                    ask_model = True

                    while ask_model:
                        response, error, elapsed, logprob_stats = query_model(model_key, prompt_text, gemini_client)

                        if error:
                            print(f"Error querying model {model_key}: {error}")

                            gemini_api_key_index += 1
                            if gemini_api_key_index >= len(GEMINI_API_KEYS):
                                flush_checkpoint(checkpoint_buffer, results_path)

                                if checkpoint_buffer:
                                    print(f"💾 Final checkpoint saved: +{len(checkpoint_buffer)} rows")

                                raise RuntimeError("All Gemini API keys have been exhausted. Stopping pipeline.")

                            print(f"Switching to next Gemini API key (index {gemini_api_key_index}) and retrying...")
                            gemini_client = get_gemini_client(GEMINI_API_KEYS[gemini_api_key_index])
                        else:
                            ask_model = False

                    row_out = {
                        "run_id": run_id,
                        "prompt_id": prompt_id,
                        "category": category,
                        "prompt_column": prompt_col,
                        "language": lang,
                        "is_paraphrase": is_paraphrase,
                        "prompt_text": prompt_text,
                        "model_key": model_key,
                        "model_id": cfg["model_id"],
                        "provider": cfg["provider"],
                        "model_parameters": cfg["parameters"],
                        "temperature": cfg["temperature"],
                        "top_p": cfg.get("top_p", None),
                        "max_tokens": cfg.get("max_tokens", None),
                        "max_new_tokens": cfg.get("max_new_tokens", None),
                        "repetition_penalty": cfg.get("repetition_penalty", None),
                        "response": response if response else "",
                        "error": error if error else "",
                        "response_length": len(response) if response else 0,
                        "elapsed_seconds": elapsed,
                        "logprob_available": logprob_stats.get("logprob_available", False),
                        "sum_logprob": logprob_stats.get("sum_logprob", None),
                        "avg_logprob": logprob_stats.get("avg_logprob", None),
                        "generated_tokens": logprob_stats.get("generated_tokens", 0),
                        "perplexity": logprob_stats.get("perplexity", None),
                        "timestamp": get_timestamp(),
                    }

                    results.append(row_out)
                    checkpoint_buffer.append(row_out)

                    if response:
                        base = f"✅ ({elapsed}s, {len(response)} chars)"
                        if logprob_stats.get("logprob_available"):
                            base += f" | avg_logprob={logprob_stats.get('avg_logprob')}"
                        print(base)
                    else:
                        print(f"❌ {error}")

                    if len(checkpoint_buffer) >= CHECKPOINT_EVERY:
                        flush_checkpoint(checkpoint_buffer, results_path)
                        print(f"💾 Checkpoint saved: +{len(checkpoint_buffer)} rows")
                        checkpoint_buffer = []

                    if cfg["type"] == "api":
                        time.sleep(1)

        flush_checkpoint(checkpoint_buffer, results_path)

        if checkpoint_buffer:
            print(f"💾 Final checkpoint saved: +{len(checkpoint_buffer)} rows")

        print(f"\n✅ Pipeline complete! Collected {len(results)} results.")
        print(f"✅ Incremental CSV saved to: {results_path}")


if __name__ == "__main__":
    main()
