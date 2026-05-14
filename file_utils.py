import pandas as pd
from pandas import DataFrame


def read_llm_results(path: str) -> DataFrame:
    return pd.read_csv(path)


def read_prompts(path: str) -> DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8")


def save_results(df: DataFrame, path: str) -> None:
    df.to_csv(path, index=False, header=True)
