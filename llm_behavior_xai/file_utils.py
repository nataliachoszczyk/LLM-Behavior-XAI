from pathlib import Path

import pandas as pd
from pandas import DataFrame


def read_llm_results(path: str | Path) -> DataFrame:
    return pd.read_csv(path)


def read_prompts(path: str | Path) -> DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8")


def save_results(df: DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False, header=True)


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return pd.DataFrame()
