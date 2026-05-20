from typing import List, Any

import pandas as pd


def find_column(df: pd.DataFrame, keywords: List[str]) -> str | None:
    for k in keywords:
        for col in df.columns:
            if k.lower() in col.lower():
                return col
    return None


def analyze_df(df: pd.DataFrame, name: str) -> dict[str, Any]:
    report: dict[str, Any] = {"name": name}

    prompt_id_col = find_column(df, ["prompt_id", "promptid", "prompt id", "prompt_uuid", "prompt_uuid_id"])
    prompt_col = find_column(df, ["prompt", "instruction", "input", "query"])
    resp_col = find_column(df, ["response", "answer", "output", "text"])
    lang_col = find_column(df, ["lang", "language", "locale"])
    paraphrase_col = find_column(df, ["paraphrase", "is_paraphrase", "has_paraphrase", "with_paraphrase"])
    variant_col = find_column(df, ["variant", "type", "response_type", "mode", "form"])

    report["total_rows"] = len(df)

    err_cols = [c for c in df.columns if "error" in c.lower()]

    if err_cols:
        is_error = df[err_cols].notna().any(axis=1)
        report["error_columns"] = err_cols
    else:
        if resp_col is not None:
            is_error = df[resp_col].isna() | (df[resp_col].astype(str).str.strip() == "")
        else:
            is_error = pd.Series([False] * len(df))

    report["error_count"] = int(is_error.sum())

    if prompt_id_col is not None:
        report["unique_prompt_ids"] = int(df[prompt_id_col].nunique())
    elif prompt_col is not None:
        report["unique_prompt_ids"] = int(df[prompt_col].nunique())
    else:
        report["unique_prompt_ids"] = None

    prompt_key_col = prompt_id_col or prompt_col

    if prompt_key_col is not None and resp_col is not None:
        grp = df[[prompt_key_col, resp_col]].dropna()
        duplicate_mask = grp.duplicated(keep="first")
        report["duplicate_rows_same_prompt_response"] = int(duplicate_mask.sum())
        counts = grp.drop_duplicates().groupby(prompt_key_col)[resp_col].nunique()
        report["prompts_with_multiple_unique_responses"] = int((counts > 1).sum())
        report["unique_responses_per_prompt_stats"] = counts.describe().to_dict() if len(counts) else {}
    else:
        report["duplicate_rows_same_prompt_response"] = None
        report["prompts_with_multiple_unique_responses"] = None
        report["unique_responses_per_prompt_stats"] = None

    report["prompt_ids_with_full_variant_set"] = None
    report["missing_variants_by_prompt_id"] = []

    if prompt_key_col is not None and (lang_col is not None or paraphrase_col is not None or variant_col is not None):
        working = df.copy()
        working["_lang_norm"] = working[lang_col].map(normalize_lang) if lang_col is not None else None

        if paraphrase_col is not None:
            working["_para_norm"] = working[paraphrase_col].map(normalize_paraphrase)
        elif variant_col is not None:
            working["_para_norm"] = working[variant_col].map(normalize_paraphrase_from_variant)
        else:
            response_source = resp_col if resp_col is not None else prompt_col
            working["_para_norm"] = working[response_source].map(normalize_paraphrase_from_text)

        working = working.dropna(subset=["_lang_norm"])
        working = working.drop_duplicates(subset=[prompt_key_col, "_lang_norm", "_para_norm"])

        expected = {
            ("pl", "no_paraphrase"): "PL without paraphrase",
            ("pl", "paraphrase"): "PL with paraphrase",
            ("en", "no_paraphrase"): "EN without paraphrase",
            ("en", "paraphrase"): "EN with paraphrase",
        }

        missing_rows = []
        full_count = 0

        for prompt_id, group in working.groupby(prompt_key_col):
            present = set(zip(group["_lang_norm"], group["_para_norm"]))
            missing = [label for key, label in expected.items() if key not in present]

            if missing:
                missing_rows.append({"prompt_id": prompt_id, "missing": missing})
            else:
                full_count += 1

        report["prompt_ids_with_full_variant_set"] = int(full_count)
        report["missing_variants_by_prompt_id"] = missing_rows
    else:
        report["prompt_ids_with_full_variant_set"] = None
        report["missing_variants_by_prompt_id"] = None

    return report


def normalize_lang(x: str) -> str | None:
    s = str(x).lower()

    if s.startswith("pl") or "pol" in s:
        return "pl"

    if s.startswith("en") or "eng" in s:
        return "en"

    return None


def normalize_paraphrase(x: str) -> str | None:
    s = str(x).lower()

    if s in {"1", "true", "yes", "y", "t"} or "para" in s:
        return "paraphrase"

    return "no_paraphrase"


def normalize_paraphrase_from_variant(x: str) -> str | None:
    s = str(x).lower()

    if "para" in s:
        return "paraphrase"

    return "no_paraphrase"


def normalize_paraphrase_from_text(x: str) -> str | None:
    s = str(x).lower()

    if "para" in s:
        return "paraphrase"

    return "no_paraphrase"


def print_dataframe_report(name: str, r: dict) -> None:
    print(f"Report for {name}:")

    for k, v in r.items():
        if k == "missing_variants_by_prompt_id" and isinstance(v, list):
            if not v:
                print("  missing_variants_by_prompt_id: []")
            else:
                print("  missing_variants_by_prompt_id:")

                for item in v:
                    print(f"    prompt_id={item['prompt_id']}: missing {', '.join(item['missing'])}")

            continue

        print(f"  {k}: {v}")

    print()


def print_summary_dataframe(output_paths: dict[Any, Any], reports: dict[Any, Any]) -> None:
    if reports:
        df_summary = pd.DataFrame.from_dict(reports, orient="index")

        print("Summary table:")
        print(df_summary.to_string())
        print("Saved files:")

        for name, path in output_paths.items():
            print(f"  {name}: {path}")
