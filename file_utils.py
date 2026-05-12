import pandas as pd
from pandas import DataFrame
from pandas.io.parsers import TextFileReader


def read_llm_results(path: str) -> TextFileReader | DataFrame:
    return pd.read_csv(path)


def read_prompts(path: str) -> TextFileReader | DataFrame:
    return pd.read_csv(path, sep=";", encoding="utf-8")


def save_results(df: DataFrame, path: str) -> None:
    df.to_csv(path, index=False, header=True)
