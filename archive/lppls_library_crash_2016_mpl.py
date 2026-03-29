from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lppls_tc_mpl_extension import FitConstraints, LPPLSModifiedTC

# =========================
# Paths and scan settings
# =========================
DATA_FILE_PATH = Path(__file__).resolve().parent / "data" / "sample_xagusd_2h.csv"
OUTPUT_DIR_NAME = "lppls_lib_output"
PREVIOUS_EVENTS_FILE_NAME = "lppls_prediction_events.csv"

SCAN_PROFILE = "bull_year"  # "crash_local" | "bull_year"
MAX_SEARCHES = 8
WORKERS = max(1, min(4, (os.cpu_count() or 1)))

if SCAN_PROFILE == "crash_local":
    RESAMPLE_RULE = "2h"
    WINDOW_SIZE_DAYS = 60.0
    SMALLEST_WINDOW_DAYS = 20.0
    OUTER_INCREMENT_DAYS = 1.0
    TC_GRID_PAST_DAYS = 5.0
    TC_GRID_FUTURE_DAYS = 25.0
    TC_GRID_STEP_DAYS = 0.25
    PEAK_CUTOFF = 0.20
elif SCAN_PROFILE == "bull_year":
    RESAMPLE_RULE = "1D"
    WINDOW_SIZE_DAYS = 300.0
    SMALLEST_WINDOW_DAYS = 120.0
    OUTER_INCREMENT_DAYS = 7.0
    TC_GRID_PAST_DAYS = 30.0
    TC_GRID_FUTURE_DAYS = 150.0
    TC_GRID_STEP_DAYS = 1.0
    PEAK_CUTOFF = 0.20
else:
    raise ValueError(f"Unknown SCAN_PROFILE: {SCAN_PROFILE}")

LIKELIHOOD_CUTOFF = 0.05
PLOT_FIGSIZE = (16, 10)

# Paper-style LPPLS filter domain.
FIT_CONSTRAINTS = FitConstraints(
    m_min=0.1,
    m_max=0.9,
    w_min=6.0,
    w_max=13.0,
    d_min=0.8,
    b_sign="negative",
)


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
    last_error: Optional[Exception] = None
    df_raw: Optional[pd.DataFrame] = None
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
    elif df_raw.shape[1] >= 6:
        dt_str = df_raw.iloc[:, 0].astype(str)
        data = df_raw.iloc[:, 1:6].copy()
    else:
        raise ValueError(f"Not enough OHLCV columns: {df_raw.shape}")

    data.columns = ["open", "high", "low", "close", "volume"]
    df = data.copy()
    df.insert(0, "datetime", dt_str)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df["datetime"] = make_datetime_monotonic(df["datetime"], default_step_seconds=30)
    return df.set_index("datetime").sort_index()


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    rule = str(rule).replace("H", "h").replace("T", "min")
    ohlc = df[["open", "high", "low", "close"]].resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    vol = df[["volume"]].resample(rule).sum()
    return pd.concat([ohlc, vol], axis=1).dropna(subset=["open", "high", "low", "close"])


def build_observations(close: pd.Series) -> np.ndarray:
    idx = pd.DatetimeIndex(close.index)
    midnight = idx.normalize()
    ordinal = midnight.map(pd.Timestamp.toordinal).to_numpy(dtype=float)
    fraction = (idx - midnight).total_seconds().to_numpy(dtype=float) / 86400.0
    time_axis = ordinal + fraction
    log_price = np.log(close.to_numpy(dtype=float))
    return np.array([time_axis, log_price], dtype=float)


def points_from_days(close: pd.Series, days: float, minimum: int = 1) -> int:
    dt_days = close.index.to_series().diff().dropna().dt.total_seconds().median() / 86400.0
    points = int(round(days / dt_days))
    return max(minimum, points)


def build_window_sizes(close: pd.Series) -> list[int]:
    largest = points_from_days(close, WINDOW_SIZE_DAYS, minimum=50)
    smallest = points_from_days(close, SMALLEST_WINDOW_DAYS, minimum=30)
    step = points_from_days(close, OUTER_INCREMENT_DAYS, minimum=1)
    window_sizes = list(range(smallest, largest + 1, step))
    if window_sizes[-1] != largest:
        window_sizes.append(largest)
    return sorted(set(window_sizes))


def build_tc_grid(current_t2: float) -> np.ndarray:
    tc_start = current_t2 - float(TC_GRID_PAST_DAYS)
    tc_stop = current_t2 + float(TC_GRID_FUTURE_DAYS)
    grid = np.arange(tc_start, tc_stop + 0.5 * TC_GRID_STEP_DAYS, TC_GRID_STEP_DAYS, dtype=float)
    return np.unique(np.round(grid, 8))


def build_window_best_table(surface: pd.DataFrame) -> pd.DataFrame:
    if surface.empty:
        return pd.DataFrame()
    rows = []
    for window_size, grp in surface.groupby("window_size", sort=True):
        grp_valid = grp[grp["qualified_conf"]].copy()
        ref = grp_valid if len(grp_valid) else grp
        best = ref.sort_values(["rm", "rp"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "window_size": int(window_size),
                "t2_time": best["t2_time"],
                "best_tc": float(best["tc"]),
                "best_tc_time": best["tc_time"],
                "rm": float(best["rm"]),
                "rp": float(best["rp"]),
                "qualified_conf": bool(best["qualified_conf"]),
                "qualified_strict": bool(best["qualified_strict"]),
                "m": float(best["m"]),
                "w": float(best["w"]),
                "D": float(best["D"]),
                "b": float(best["b"]),
                "m_lo": float(best["m_lo"]),
                "m_hi": float(best["m_hi"]),
                "w_lo": float(best["w_lo"]),
                "w_hi": float(best["w_hi"]),
                "D_lo": float(best["D_lo"]),
                "D_hi": float(best["D_hi"]),
            }
        )
    return pd.DataFrame(rows).sort_values("window_size")


def build_candidate_table(scenarios: pd.DataFrame) -> pd.DataFrame:
    if scenarios.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "peak_tc_time",
                "interval_lo_time",
                "interval_hi_time",
                "horizon_days",
                "support_windows",
                "support_share",
                "rm_max",
                "window_min",
                "window_max",
                "m",
                "w",
                "D",
                "b",
            ]
        )
    cols = [
        "scenario_id",
        "peak_tc_time",
        "interval_lo_time",
        "interval_hi_time",
        "horizon_days",
        "support_windows",
        "support_share",
        "rm_max",
        "window_min",
        "window_max",
        "m",
        "w",
        "D",
        "b",
    ]
    out = scenarios.loc[:, cols].copy()
    out["support_share"] = out["support_share"].astype(float)
    return out


def load_previous_prediction_events(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(file_path)
    required_cols = {"tc_median_time", "tc_q10_time", "tc_q90_time"}
    if not required_cols.issubset(df.columns):
        return pd.DataFrame()

    time_cols = [c for c in ["signal_time", "tc_median_time", "tc_q10_time", "tc_q90_time"] if c in df.columns]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if "signal_time" in df.columns:
        df = df.dropna(subset=["signal_time"]).set_index("signal_time").sort_index()
    else:
        df.index = pd.DatetimeIndex([pd.NaT] * len(df), name="signal_time")
    return df


def print_candidate_scenarios(scenarios: pd.DataFrame) -> None:
    if scenarios.empty:
        print("\nNo candidate scenario passed the confidence-aware modified-likelihood filter.")
        return

    print("\nCandidate t_c scenarios:")
    for _, row in scenarios.iterrows():
        print(
            f"\n[{row['scenario_id']}] peak={pd.Timestamp(row['peak_tc_time'])} | "
            f"interval={pd.Timestamp(row['interval_lo_time'])} ~ {pd.Timestamp(row['interval_hi_time'])}"
        )
        print(
            f"  horizon={float(row['horizon_days']):.2f}d | "
            f"support={int(row['support_windows'])} windows ({float(row['support_share']):.1%}) | "
            f"Rm(max)={float(row['rm_max']):.3f}"
        )
        print(
            f"  params: m={float(row['m']):.3f}, "
            f"omega={float(row['w']):.3f}, D={float(row['D']):.3f}, B={float(row['b']):.6f}"
        )


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"Data file: {DATA_FILE_PATH}")
    df_raw = read_ohlcv_file(DATA_FILE_PATH)
    print(f"Raw rows: {len(df_raw):,d} | range: {df_raw.index.min()} ~ {df_raw.index.max()}")

    df = resample_ohlcv(df_raw, RESAMPLE_RULE)
    close = df["close"].dropna().copy()
    print(
        f"Profile: {SCAN_PROFILE} | Resampled rule: {RESAMPLE_RULE} | "
        f"rows: {len(df):,d} | range: {df.index.min()} ~ {df.index.max()}"
    )

    observations = build_observations(close)
    model = LPPLSModifiedTC(observations=observations)

    window_sizes = build_window_sizes(close)
    tc_grid = build_tc_grid(float(observations[0, -1]))
    print(
        f"Modified-likelihood scan: windows={len(window_sizes)} | "
        f"window range={window_sizes[0]}..{window_sizes[-1]} pts | "
        f"tc grid={len(tc_grid)} points | searches per tc={MAX_SEARCHES}"
    )
    print(
        f"Constraints: m in [{FIT_CONSTRAINTS.m_min:.1f}, {FIT_CONSTRAINTS.m_max:.1f}], "
        f"omega in [{FIT_CONSTRAINTS.w_min:.1f}, {FIT_CONSTRAINTS.w_max:.1f}], "
        f"D >= {FIT_CONSTRAINTS.d_min:.1f}, B sign = {FIT_CONSTRAINTS.b_sign}"
    )

    result = model.scan_tc_surface(
        t2_index=-1,
        window_sizes=window_sizes,
        tc_grid=tc_grid,
        max_searches=MAX_SEARCHES,
        cutoff=LIKELIHOOD_CUTOFF,
        peak_cutoff=PEAK_CUTOFF,
        constraints=FIT_CONSTRAINTS,
        random_state=11,
    )

    surface = result["surface"].sort_values(["window_size", "tc"]).copy()
    intervals = result["intervals"].sort_values(["window_size", "interval_lo"]).copy()
    peaks = result["peaks"].sort_values(["window_size", "rm"], ascending=[True, False]).copy()
    scenarios = result["scenarios"].copy()
    window_best = build_window_best_table(surface)
    candidate_table = build_candidate_table(scenarios)

    print_candidate_scenarios(candidate_table)

    out_dir = DATA_FILE_PATH.parent / OUTPUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    previous_events_path = out_dir / PREVIOUS_EVENTS_FILE_NAME
    previous_events = load_previous_prediction_events(previous_events_path)
    if len(previous_events) > 0:
        print(f"\nLoaded {len(previous_events)} previous crash-prediction events from: {previous_events_path}")
    else:
        print(f"\nNo previous prediction-event overlay found at: {previous_events_path}")

    surface.to_csv(out_dir / "lppls_tc_surface.csv", encoding="utf-8-sig", index=False)
    intervals.to_csv(out_dir / "lppls_tc_intervals.csv", encoding="utf-8-sig", index=False)
    peaks.to_csv(out_dir / "lppls_tc_peaks.csv", encoding="utf-8-sig", index=False)
    window_best.to_csv(out_dir / "lppls_tc_window_best.csv", encoding="utf-8-sig", index=False)
    candidate_table.to_csv(out_dir / "lppls_tc_candidate_scenarios.csv", encoding="utf-8-sig", index=False)

    fig, _ = model.plot_tc_structure(
        result,
        use_qualified="qualified_conf",
        figsize=PLOT_FIGSIZE,
        title=(
            f"LPPLS modified-profile-likelihood t_c structure | "
            f"profile={SCAN_PROFILE} | t2={result['current_t2_time']} | cutoff={LIKELIHOOD_CUTOFF:.2f}"
        ),
        prediction_events=previous_events,
    )
    fig.savefig(out_dir / "01_lppls_tc_structure.png", dpi=160, bbox_inches="tight")

    print(f"\nOutput dir: {out_dir}")
    print(
        "Generated: 01_lppls_tc_structure.png, lppls_tc_surface.csv, lppls_tc_intervals.csv, "
        "lppls_tc_peaks.csv, lppls_tc_window_best.csv, lppls_tc_candidate_scenarios.csv"
    )
    plt.show()


if __name__ == "__main__":
    main()
