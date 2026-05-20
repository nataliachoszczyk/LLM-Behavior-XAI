from pathlib import Path

import pandas as pd

from file_utils import safe_read_csv


def load_and_merge_datasets(temporary_datasets: list[tuple[str, list[Path]]]) -> dict[str, pd.DataFrame]:
    merged_dfs = {}

    for split_name, datasets_paths in temporary_datasets:
        dataframes = []

        for path in datasets_paths:
            df = safe_read_csv(path)

            if df.empty:
                print(f"Skipping empty {path.name}")
                continue

            dataframes.append(df)
            print(f"Successfully loaded: {path} - {df.shape}")

        if dataframes:
            merged_df = pd.concat(dataframes, ignore_index=True, sort=False)
        else:
            merged_df = pd.DataFrame()
            print("No dataframes were loaded successfully.")

        merged_dfs[split_name] = merged_df
        print(f"{split_name}: merged rows={len(merged_df)} from {len(datasets_paths)} datasets")

    return merged_dfs
