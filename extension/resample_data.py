from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "Data"
DEFAULT_DATA_FILE_NAME = "xagusd_30s_all.csv"
DEFAULT_RESAMPLE_RULES = ["1h", "2h", "day"]


def normalize_rule(rule: str) -> str:
    text = str(rule).replace("H", "h").replace("T", "min").strip().lower()
    if text == "day":
        return "1d"
    return text


def build_resampled_data_path(data_dir: Path, data_file_name: str, rule: str) -> Path:
    stem = Path(data_file_name).stem
    return data_dir / f"{stem}__{normalize_rule(rule)}.csv"


def make_datetime_monotonic(dt: pd.Series, default_step_seconds: int = 30) -> pd.DatetimeIndex:
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


def read_ohlcv_file(file_path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    df_raw: pd.DataFrame | None = None

    for sep in [",", "\t", r"\s+"]:
        try:
            candidate = pd.read_csv(file_path, sep=sep, header=None, engine="python")
        except Exception as exc:
            last_error = exc
            continue
        if candidate.shape[1] >= 6:
            df_raw = candidate
            break

    if df_raw is None:
        raise ValueError(f"Unable to parse OHLCV file: {file_path}. Last error: {last_error}")

    if df_raw.shape[1] >= 7:
        dt_str = df_raw.iloc[:, 0].astype(str) + " " + df_raw.iloc[:, 1].astype(str)
        data = df_raw.iloc[:, 2:7].copy()
    else:
        dt_str = df_raw.iloc[:, 0].astype(str)
        data = df_raw.iloc[:, 1:6].copy()

    data.columns = ["open", "high", "low", "close", "volume"]
    df = data.copy()
    df.insert(0, "datetime", dt_str)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df["datetime"] = make_datetime_monotonic(df["datetime"], default_step_seconds=30)
    return df.set_index("datetime").sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    use_rule = normalize_rule(rule)
    ohlc = df[["open", "high", "low", "close"]].resample(use_rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    vol = df[["volume"]].resample(use_rule).sum()
    return pd.concat([ohlc, vol], axis=1).dropna(subset=["open", "high", "low", "close"])


def write_ohlcv_file(df: pd.DataFrame, output_path: Path) -> None:
    out = df.reset_index().copy()
    out["datetime"] = out["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(output_path, index=False, header=False)


def create_resampled_file(data_dir: Path, data_file_name: str, rule: str) -> Path:
    raw_path = data_dir / data_file_name
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    df_raw = read_ohlcv_file(raw_path)
    df_out = resample_ohlcv(df_raw, rule)
    output_path = build_resampled_data_path(data_dir, data_file_name, rule)
    write_ohlcv_file(df_out, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create resampled OHLCV files for LPPL analysis.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--data-file", default=DEFAULT_DATA_FILE_NAME)
    parser.add_argument("--rules", nargs="+", default=DEFAULT_RESAMPLE_RULES)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    for rule in args.rules:
        output_path = create_resampled_file(data_dir, args.data_file, rule)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
