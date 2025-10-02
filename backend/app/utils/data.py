import os
from typing import Dict, Optional, Tuple

import pandas as pd


def get_dir_data(dir: str, diff=None) -> Tuple[Optional[pd.DataFrame], float]:
    dfs = []
    if diff is None:
        time, count = 0, 0
        for path in sorted(os.listdir(dir)):
            df = pd.read_csv(os.path.join(dir, path))
            dfs.append(df)
            time += df.iloc[-1]["time_sec"]
            count += df.shape[0] - 1
        if count == 0:
            return None, 0.0
        diff = time / count
    else:
        for path in sorted(os.listdir(dir)):
            df = pd.read_csv(os.path.join(dir, path))
            dfs.append(df)

    last_time = -diff
    for df in dfs:
        df["time_sec"] += last_time + diff
        last_time = df.iloc[-1]["time_sec"]
    return pd.concat(dfs, ignore_index=True), diff


def get_data(path: str) -> Optional[pd.DataFrame]:
    df_uterus, diff = get_dir_data(os.path.join(path, "uterus"))
    if df_uterus is None:
        return None
    df_uterus.rename(columns={"value": "uterus"}, inplace=True)
    df_bpm, _ = get_dir_data(os.path.join(path, "bpm"), diff)
    if df_bpm is None:
        return None
    df_bpm.rename(columns={"value": "bpm"}, inplace=True)
    return df_bpm.merge(df_uterus, on="time_sec", how="outer")


def get_all_data(dir_path: str, verbose: bool = False) -> Dict[str, pd.DataFrame]:
    res = {}
    for name in os.listdir(dir_path):
        df = get_data(os.path.join(dir_path, name))
        if df is not None:
            res[name] = df
        elif verbose:
            print(f"ignore {os.path.join(dir_path, name)}")
    return res
