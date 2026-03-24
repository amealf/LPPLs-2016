from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_FILE_PATH = Path(r"D:\Code\data\20260324\xagusd_15s_all.csv")
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_2H_PATH = DATA_DIR / "sample_xagusd_2h.csv"
OUTPUT_1D_PATH = DATA_DIR / "sample_xagusd_1d.csv"


def make_datetime_monotonic(dt: pd.Series, default_step_seconds: int = 15) -> pd.DatetimeIndex:
    dt = pd.to_datetime(dt, errors="coerce")
    if dt.isna().any():
        raise ValueError("Unparseable datetime values detected.")

    uniq = pd.Series(dt.unique()).dropna().sort_values()
    diffs = uniq.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    base_delta = diffs.mode().iloc[0] if len(diffs) else pd.Timedelta(seconds=default_step_seconds)

    s = pd.Series(dt)
    within_rank = s.groupby(s).cumcount()
    within_size = s.groupby(s).transform("size")

    base_ns = int(base_delta.value)
    step_ns = np.round(base_ns / within_size.to_numpy()).astype(np.int64)
    dt_ns = s.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    adj_ns = dt_ns + within_rank.to_numpy(dtype=np.int64) * step_ns
    adj = pd.to_datetime(adj_ns)

    adj_ns2 = adj.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for i in range(1, len(adj_ns2)):
        if adj_ns2[i] <= adj_ns2[i - 1]:
            adj_ns2[i] = adj_ns2[i - 1] + 1
    return pd.to_datetime(adj_ns2)


def read_source_file(file_path: Path) -> pd.DataFrame:
    df = pd.read_csv(file_path, header=None)
    if df.shape[1] < 6:
        raise ValueError(f"Unexpected source shape: {df.shape}")

    df = df.iloc[:, :6].copy()
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = make_datetime_monotonic(df["time"], default_step_seconds=15)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    df = df.sort_values("time").set_index("time")
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = (
        df.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    return out


def write_ohlcv_csv(df: pd.DataFrame, output_path: Path) -> None:
    out = df.reset_index().copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(output_path, index=False, header=False)


def main() -> None:
    print(f"Source file: {SOURCE_FILE_PATH}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = read_source_file(SOURCE_FILE_PATH)
    df_2h = resample_ohlcv(df, "2h")
    df_1d = resample_ohlcv(df, "1D")

    write_ohlcv_csv(df_2h, OUTPUT_2H_PATH)
    write_ohlcv_csv(df_1d, OUTPUT_1D_PATH)

    print(f"Saved 2H sample: {OUTPUT_2H_PATH} ({len(df_2h):,} rows)")
    print(f"Saved 1D sample: {OUTPUT_1D_PATH} ({len(df_1d):,} rows)")
    print(f"2H range: {df_2h.index.min()} -> {df_2h.index.max()}")
    print(f"1D range: {df_1d.index.min()} -> {df_1d.index.max()}")


if __name__ == "__main__":
    main()
