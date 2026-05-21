from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


def load_final_splits(split_paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {split: pd.read_csv(path) for split, path in split_paths.items()}


def create_split_overview(final_splits: dict[str, DataFrame]) -> list[dict[str, Any]]:
    split_overview = []

    for split, df in final_splits.items():
        split_overview.append(
            {
                "split": split,
                "rows": len(df),
                "models": df["model_key"].nunique(),
                "languages": df["language"].nunique(),
                "paraphrase_values": df["is_paraphrase"].nunique(),
            }
        )

    return split_overview
