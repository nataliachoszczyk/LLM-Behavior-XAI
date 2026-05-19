import pandas as pd
from pandas import DataFrame

from llm_behavior_xai.llm_response_merger_and_validator.validate_dataset import find_column


def deduplicate_by_prompt_and_response(df: pd.DataFrame) -> pd.DataFrame:
    prompt_col = find_column(df, ["prompt", "instruction", "input", "query"])
    resp_col = find_column(df, ["response", "answer", "output", "text"])

    if prompt_col is None or resp_col is None:
        return df.copy()

    dup_mask = df[[prompt_col, resp_col]].duplicated(keep="last")

    return df.loc[~dup_mask].copy()


def get_error_mask(df: pd.DataFrame) -> pd.Series:
    resp_col = find_column(df, ["response", "answer", "output", "text"])
    err_cols = [c for c in df.columns if "error" in c.lower()]

    if err_cols:
        return df[err_cols].notna().any(axis=1)

    if resp_col is not None:
        return df[resp_col].isna() | (df[resp_col].astype(str).str.strip() == "")

    return pd.Series([False] * len(df), index=df.index)


def clean_dataframe(df: pd.DataFrame) -> tuple[DataFrame, int, int]:
    error_mask = get_error_mask(df)
    df_clean = df.loc[~error_mask].copy()
    removed_errors = int(error_mask.sum())
    deduped = deduplicate_by_prompt_and_response(df_clean)
    removed_duplicates = len(df_clean) - len(deduped)

    return deduped, removed_duplicates, removed_errors
