from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import transforms as mtransforms

from extension.lppls_tc_mpl_extension import FitConstraints, LPPLSModifiedTC
from extension.resample_data import build_resampled_data_path


# =========================
# Paths and core settings
# =========================
DATA_DIR = Path(__file__).resolve().parent / "Data"
DATA_FILE_NAME = "xagusd_30s_all.csv"
DATA_FILE_PATH = DATA_DIR / DATA_FILE_NAME
START_DATE = "20241201"
END_DATE = "" 

RESULT_DIR_NAME = "LPPL result"
SCAN_PROFILE = "bull_year"  # "crash_local" | "bull_year"
BACKTEST_SPEED_MODE = "fast_validation"  # "fast_validation" | "full_scan"
WORKERS = max(1, min(4, (os.cpu_count() or 1)))

if SCAN_PROFILE == "crash_local":
    RESAMPLE_RULE = "2h"
    WINDOW_SIZE_DAYS = 60.0
    SMALLEST_WINDOW_DAYS = 20.0
    OUTER_INCREMENT_DAYS = 2.0 if BACKTEST_SPEED_MODE == "fast_validation" else 1.0
    TC_GRID_PAST_DAYS = 0.0
    TC_GRID_FUTURE_DAYS = 18.0 if BACKTEST_SPEED_MODE == "fast_validation" else 25.0
    TC_GRID_STEP_DAYS = 0.5 if BACKTEST_SPEED_MODE == "fast_validation" else 0.25
    PEAK_CUTOFF = 0.20 
    ANALYSIS_STEP_BARS = 2 if BACKTEST_SPEED_MODE == "fast_validation" else 1
    MIN_HISTORY_DAYS = 25.0
    PREFERRED_HORIZON_DAYS = (2.0, 12.0)
    MAX_HORIZON_DAYS = 20.0
    MAX_INTERVAL_WIDTH_DAYS = 8.0
    RUNUP_LOOKBACK_DAYS = 18.0
    TARGET_RUNUP_PCT = 0.18
    FINAL_PUSH_DAYS = 4.0
    TARGET_FINAL_PUSH_PCT = 0.05
    VALIDATION_BUFFER_DAYS = 4.0
    MAX_SEARCHES = 4 if BACKTEST_SPEED_MODE == "fast_validation" else 8
    MAX_WINDOWS_PER_SCAN = 16 if BACKTEST_SPEED_MODE == "fast_validation" else 999
else:
    RESAMPLE_RULE = "1D"
    WINDOW_SIZE_DAYS = 300.0
    SMALLEST_WINDOW_DAYS = 120.0
    OUTER_INCREMENT_DAYS = 14.0 if BACKTEST_SPEED_MODE == "fast_validation" else 7.0
    TC_GRID_PAST_DAYS = 0.0
    TC_GRID_FUTURE_DAYS = 90.0 if BACKTEST_SPEED_MODE == "fast_validation" else 150.0
    TC_GRID_STEP_DAYS = 1.0
    PEAK_CUTOFF = 0.20
    ANALYSIS_STEP_BARS = 3
    MIN_HISTORY_DAYS = 140.0
    PREFERRED_HORIZON_DAYS = (7.0, 45.0)
    MAX_HORIZON_DAYS = 90.0
    MAX_INTERVAL_WIDTH_DAYS = 35.0
    RUNUP_LOOKBACK_DAYS = 30.0
    TARGET_RUNUP_PCT = 0.25
    FINAL_PUSH_DAYS = 7.0
    TARGET_FINAL_PUSH_PCT = 0.06
    VALIDATION_BUFFER_DAYS = 10.0
    MAX_SEARCHES = 4 if BACKTEST_SPEED_MODE == "fast_validation" else 8
    MAX_WINDOWS_PER_SCAN = 12 if BACKTEST_SPEED_MODE == "fast_validation" else 999

LIKELIHOOD_CUTOFF = 0.05
MODEL_SCORE_THRESHOLD = 0.68
MIN_SUPPORT_SHARE = 0.18
MIN_RM_MAX = 0.25
ZONE_MERGE_MULTIPLIER = 1.6
VALIDATION_DRAWDOWN_THRESHOLD = -0.12
PLOT_FIGSIZE = (17, 10)

FIT_CONSTRAINTS = FitConstraints(
    m_min=0.1,
    m_max=0.9,
    w_min=6.0,
    w_max=13.0,
    d_min=0.8,
    b_sign="negative",
)


def infer_symbol_name(file_path: Path) -> str:
    tokens = [t for t in re.split(r"[_\-\s]+", file_path.stem.lower()) if t]
    filtered = [t for t in tokens if t not in {"sample", "all", "data", "ohlcv"}]
    if not filtered:
        return "unknown"
    if len(filtered) >= 2 and re.fullmatch(r"\d+[smhdw]", filtered[-1]):
        filtered = filtered[:-1]
    return filtered[0] if filtered else "unknown"


def make_output_tag(close: pd.Series) -> str:
    symbol = infer_symbol_name(Path(DATA_FILE_NAME))
    start_text = pd.Timestamp(close.index.min()).strftime("%Y%m%d")
    end_text = pd.Timestamp(close.index.max()).strftime("%Y%m%d")
    return f"{symbol}_{start_text}_{end_text}"


def resolve_input_data_path() -> tuple[Path, bool]:
    converted_path = build_resampled_data_path(DATA_DIR, DATA_FILE_NAME, RESAMPLE_RULE)
    if converted_path.exists():
        return converted_path, True
    return DATA_FILE_PATH, False


def parse_date_yyyymmdd(text: str) -> pd.Timestamp | None:
    text = str(text).strip()
    if not text:
        return None
    return pd.Timestamp(pd.to_datetime(text, format="%Y%m%d"))


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
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df["datetime"] = make_datetime_monotonic(df["datetime"], default_step_seconds=30)
    return df.set_index("datetime").sort_index()


def apply_date_range(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = parse_date_yyyymmdd(start_date)
    end_ts = parse_date_yyyymmdd(end_date)

    out = df
    if start_ts is not None:
        out = out.loc[out.index >= start_ts]
    if end_ts is not None:
        out = out.loc[out.index < end_ts + pd.Timedelta(days=1)]
    if out.empty:
        raise ValueError("No rows left after applying START_DATE and END_DATE.")
    return out


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


def thin_window_sizes(window_sizes: list[int], max_windows: int) -> list[int]:
    if max_windows <= 0 or len(window_sizes) <= max_windows:
        return window_sizes
    sample_pos = np.linspace(0, len(window_sizes) - 1, num=max_windows)
    sampled = [window_sizes[int(round(pos))] for pos in sample_pos]
    sampled.extend([window_sizes[0], window_sizes[-1]])
    return sorted(set(sampled))


def build_tc_grid(current_t2: float) -> np.ndarray:
    tc_start = current_t2 - float(TC_GRID_PAST_DAYS)
    tc_stop = current_t2 + float(TC_GRID_FUTURE_DAYS)
    grid = np.arange(tc_start, tc_stop + 0.5 * TC_GRID_STEP_DAYS, TC_GRID_STEP_DAYS, dtype=float)
    return np.unique(np.round(grid, 8))


def trailing_runup_from_low(close: pd.Series, end_time: pd.Timestamp, lookback_days: float) -> float:
    seg = close.loc[end_time - pd.Timedelta(days=float(lookback_days)):end_time]
    if len(seg) < 2:
        return np.nan
    low = float(seg.min())
    end_price = float(seg.iloc[-1])
    if low <= 0.0:
        return np.nan
    return end_price / low - 1.0


def trailing_window_return(close: pd.Series, end_time: pd.Timestamp, lookback_days: float) -> float:
    seg = close.loc[end_time - pd.Timedelta(days=float(lookback_days)):end_time]
    if len(seg) < 2:
        return np.nan
    start_price = float(seg.iloc[0])
    end_price = float(seg.iloc[-1])
    if start_price <= 0.0:
        return np.nan
    return end_price / start_price - 1.0


def forward_drawdown(close: pd.Series, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
    seg = close.loc[start_time:end_time]
    if len(seg) < 2:
        return np.nan
    p0 = float(seg.iloc[0])
    pmin = float(seg.min())
    if p0 <= 0.0:
        return np.nan
    return pmin / p0 - 1.0


def first_drawdown_breach_time(
    close: pd.Series,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    drawdown_threshold: float,
) -> pd.Timestamp:
    seg = close.loc[start_time:end_time]
    if len(seg) < 2:
        return pd.NaT
    p0 = float(seg.iloc[0])
    if p0 <= 0.0:
        return pd.NaT
    dd = seg / p0 - 1.0
    hit = dd[dd <= float(drawdown_threshold)]
    return hit.index[0] if len(hit) else pd.NaT


def safe_ratio(value: float, target: float) -> float:
    if not np.isfinite(value) or target <= 0.0:
        return 0.0
    return float(np.clip(value / target, 0.0, 1.0))


def bounded_score(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if high <= low:
        return 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def horizon_score(horizon_days: float) -> float:
    if not np.isfinite(horizon_days) or horizon_days <= 0.0 or horizon_days > MAX_HORIZON_DAYS:
        return 0.0
    preferred_lo, preferred_hi = PREFERRED_HORIZON_DAYS
    if horizon_days < preferred_lo:
        return float(np.clip(horizon_days / max(preferred_lo, 1e-6), 0.0, 1.0))
    if horizon_days <= preferred_hi:
        return 1.0
    tail = max(MAX_HORIZON_DAYS - preferred_hi, 1e-6)
    return float(np.clip(1.0 - (horizon_days - preferred_hi) / tail, 0.0, 1.0))


def build_primary_scenario_row(
    scenarios: pd.DataFrame,
    close: pd.Series,
    t2_time: pd.Timestamp,
    t2_price: float,
) -> dict[str, Any]:
    base_row: dict[str, Any] = {
        "t2_time": t2_time,
        "t2_price": float(t2_price),
        "has_scenario": False,
        "scenario_id": None,
        "scenario_count": 0,
        "peak_tc_time": pd.NaT,
        "interval_lo_time": pd.NaT,
        "interval_hi_time": pd.NaT,
        "horizon_days": np.nan,
        "interval_width_days": np.nan,
        "support_windows": 0,
        "support_share": np.nan,
        "rm_max": np.nan,
        "window_min": np.nan,
        "window_max": np.nan,
        "m": np.nan,
        "omega": np.nan,
        "D": np.nan,
        "B": np.nan,
        "runup_pct": trailing_runup_from_low(close, t2_time, RUNUP_LOOKBACK_DAYS),
        "final_push_pct": trailing_window_return(close, t2_time, FINAL_PUSH_DAYS),
        "support_score": 0.0,
        "rm_score": 0.0,
        "width_score": 0.0,
        "horizon_score": 0.0,
        "runup_score": 0.0,
        "push_score": 0.0,
        "signal_score": 0.0,
        "is_high_confidence": False,
        "validation_end_time": pd.NaT,
        "forward_drawdown": np.nan,
        "first_breach_time": pd.NaT,
        "validated_break": False,
        "reason": "no_scenario",
    }
    if scenarios.empty:
        return base_row

    runup_pct = base_row["runup_pct"]
    final_push_pct = base_row["final_push_pct"]
    scored_rows: list[dict[str, Any]] = []
    for _, row in scenarios.iterrows():
        peak_tc_time = pd.Timestamp(row["peak_tc_time"])
        interval_lo_time = pd.Timestamp(row["interval_lo_time"])
        interval_hi_time = pd.Timestamp(row["interval_hi_time"])
        horizon_days = float((peak_tc_time - t2_time) / pd.Timedelta(days=1))
        interval_width_days = float((interval_hi_time - interval_lo_time) / pd.Timedelta(days=1))
        support_share = float(row["support_share"])
        rm_max = float(row["rm_max"])

        support_score = bounded_score(support_share, MIN_SUPPORT_SHARE, 0.55)
        rm_score = bounded_score(rm_max, MIN_RM_MAX, 0.85)
        width_score = float(np.clip(1.0 - interval_width_days / max(MAX_INTERVAL_WIDTH_DAYS, 1e-6), 0.0, 1.0))
        h_score = horizon_score(horizon_days)
        runup_score = safe_ratio(float(runup_pct), TARGET_RUNUP_PCT)
        push_score = safe_ratio(float(final_push_pct), TARGET_FINAL_PUSH_PCT)
        signal_score = (
            0.30 * support_score
            + 0.27 * rm_score
            + 0.16 * width_score
            + 0.15 * h_score
            + 0.08 * runup_score
            + 0.04 * push_score
        )
        is_high_conf = bool(
            signal_score >= MODEL_SCORE_THRESHOLD
            and support_share >= MIN_SUPPORT_SHARE
            and rm_max >= MIN_RM_MAX
            and horizon_days > 0.0
            and interval_width_days <= MAX_INTERVAL_WIDTH_DAYS
        )
        scored_rows.append(
            {
                **row.to_dict(),
                "scenario_count": int(len(scenarios)),
                "horizon_days": horizon_days,
                "interval_width_days": interval_width_days,
                "support_score": support_score,
                "rm_score": rm_score,
                "width_score": width_score,
                "horizon_score": h_score,
                "runup_score": runup_score,
                "push_score": push_score,
                "signal_score": signal_score,
                "is_high_confidence": is_high_conf,
            }
        )

    best = max(
        scored_rows,
        key=lambda row: (
            bool(row["horizon_days"] > 0.0),
            float(row["signal_score"]),
            float(row["support_share"]),
            float(row["rm_max"]),
        ),
    )

    base_row.update(
        {
            "has_scenario": True,
            "scenario_id": best["scenario_id"],
            "scenario_count": int(best["scenario_count"]),
            "peak_tc_time": pd.Timestamp(best["peak_tc_time"]),
            "interval_lo_time": pd.Timestamp(best["interval_lo_time"]),
            "interval_hi_time": pd.Timestamp(best["interval_hi_time"]),
            "horizon_days": float(best["horizon_days"]),
            "interval_width_days": float(best["interval_width_days"]),
            "support_windows": int(best["support_windows"]),
            "support_share": float(best["support_share"]),
            "rm_max": float(best["rm_max"]),
            "window_min": int(best["window_min"]),
            "window_max": int(best["window_max"]),
            "m": float(best["m"]),
            "omega": float(best["w"]),
            "D": float(best["D"]),
            "B": float(best["b"]),
            "support_score": float(best["support_score"]),
            "rm_score": float(best["rm_score"]),
            "width_score": float(best["width_score"]),
            "horizon_score": float(best["horizon_score"]),
            "runup_score": float(best["runup_score"]),
            "push_score": float(best["push_score"]),
            "signal_score": float(best["signal_score"]),
            "is_high_confidence": bool(best["is_high_confidence"]),
            "reason": (
                f"scenario={best['scenario_id']}, support={float(best['support_share']):.1%}, "
                f"Rm={float(best['rm_max']):.3f}, horizon={float(best['horizon_days']):.1f}d, "
                f"width={float(best['interval_width_days']):.1f}d, "
                f"runup={float(runup_pct):.1%}, final_push={float(final_push_pct):.1%}"
            ),
        }
    )
    return base_row


def build_fit_cache_entry(surface: pd.DataFrame, row: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(row.get("is_high_confidence")):
        return []
    if not bool(row.get("has_scenario")):
        return []
    return select_diverse_omega_surface_fits(surface, row, max_curves=5)


def build_interval_cache_entry(intervals: pd.DataFrame, row: dict[str, Any]) -> list[dict[str, Any]]:
    if intervals.empty:
        return []
    if not bool(row.get("is_high_confidence")):
        return []
    if not bool(row.get("has_scenario")):
        return []

    lo_time = pd.Timestamp(row["interval_lo_time"])
    hi_time = pd.Timestamp(row["interval_hi_time"])
    overlap = intervals[
        (intervals["interval_hi_time"] >= lo_time)
        & (intervals["interval_lo_time"] <= hi_time)
    ].copy()
    if overlap.empty:
        return []

    overlap = overlap.sort_values(["window_size", "peak_rm"], ascending=[True, False])
    chosen = overlap.groupby("window_size", as_index=False).first()
    return chosen.sort_values("window_size").to_dict("records")


def validate_prediction_row(close: pd.Series, row: pd.Series) -> pd.Series:
    out = row.copy()
    if not bool(row["has_scenario"]) or pd.isna(row["interval_hi_time"]):
        return out

    validation_end = min(
        pd.Timestamp(row["interval_hi_time"]) + pd.Timedelta(days=float(VALIDATION_BUFFER_DAYS)),
        close.index[-1],
    )
    out["validation_end_time"] = validation_end
    if validation_end <= row.name:
        return out

    out["forward_drawdown"] = forward_drawdown(close, row.name, validation_end)
    out["first_breach_time"] = first_drawdown_breach_time(
        close,
        row.name,
        validation_end,
        VALIDATION_DRAWDOWN_THRESHOLD,
    )
    out["validated_break"] = bool(pd.notna(out["first_breach_time"]))
    return out


def validate_prediction_rows(close: pd.Series, state_df: pd.DataFrame) -> pd.DataFrame:
    if state_df.empty:
        return state_df

    out = state_df.copy()
    for signal_time, row in out.loc[out["has_scenario"] & out["interval_hi_time"].notna()].iterrows():
        validation_end = min(
            pd.Timestamp(row["interval_hi_time"]) + pd.Timedelta(days=float(VALIDATION_BUFFER_DAYS)),
            close.index[-1],
        )
        out.at[signal_time, "validation_end_time"] = validation_end
        if validation_end <= signal_time:
            continue
        out.at[signal_time, "forward_drawdown"] = forward_drawdown(close, signal_time, validation_end)
        breach_time = first_drawdown_breach_time(
            close,
            signal_time,
            validation_end,
            VALIDATION_DRAWDOWN_THRESHOLD,
        )
        out.at[signal_time, "first_breach_time"] = breach_time
        out.at[signal_time, "validated_break"] = bool(pd.notna(breach_time))
    return out


def build_bubble_zones(state_df: pd.DataFrame, close: pd.Series, analysis_gap: pd.Timedelta) -> pd.DataFrame:
    high_conf = state_df[state_df["is_high_confidence"]].copy()
    if high_conf.empty:
        return pd.DataFrame(
            columns=[
                "zone_id",
                "entry_time",
                "exit_time",
                "entry_price",
                "peak_signal_time",
                "max_signal_score",
                "peak_tc_time",
                "interval_lo_time",
                "interval_hi_time",
                "support_share",
                "rm_max",
                "n_signals",
                "validated_break",
                "first_breach_time",
                "forward_drawdown",
            ]
        ).set_index("entry_time")

    merge_gap = analysis_gap * float(ZONE_MERGE_MULTIPLIER)
    groups: list[list[tuple[pd.Timestamp, pd.Series]]] = []
    current: list[tuple[pd.Timestamp, pd.Series]] = []
    prev_time: Optional[pd.Timestamp] = None
    for signal_time, row in high_conf.iterrows():
        if prev_time is None or (signal_time - prev_time) <= merge_gap:
            current.append((signal_time, row))
        else:
            groups.append(current)
            current = [(signal_time, row)]
        prev_time = signal_time
    if current:
        groups.append(current)

    zone_rows: list[dict[str, Any]] = []
    for zone_id, group in enumerate(groups, start=1):
        entry_time = group[0][0]
        exit_time = group[-1][0]
        best_time, best_row = max(group, key=lambda x: (float(x[1]["signal_score"]), x[0].value))
        zone_rows.append(
            {
                "zone_id": zone_id,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": float(close.loc[entry_time]),
                "peak_signal_time": best_time,
                "max_signal_score": float(best_row["signal_score"]),
                "peak_tc_time": best_row["peak_tc_time"],
                "interval_lo_time": best_row["interval_lo_time"],
                "interval_hi_time": best_row["interval_hi_time"],
                "support_share": float(best_row["support_share"]),
                "rm_max": float(best_row["rm_max"]),
                "n_signals": len(group),
                "validated_break": bool(best_row["validated_break"]),
                "first_breach_time": best_row["first_breach_time"],
                "forward_drawdown": float(best_row["forward_drawdown"]) if np.isfinite(best_row["forward_drawdown"]) else np.nan,
            }
        )

    return pd.DataFrame(zone_rows).set_index("entry_time").sort_index()


def estimate_backtest_workload(
    observations: np.ndarray,
    full_window_sizes: list[int],
    analysis_indices: list[int],
) -> tuple[int, int]:
    total_fits = 0
    total_windows = 0
    for t2_index in analysis_indices:
        eligible = [ws for ws in full_window_sizes if ws <= (t2_index + 1)]
        eligible = thin_window_sizes(eligible, MAX_WINDOWS_PER_SCAN)
        total_windows += len(eligible)
        total_fits += len(eligible) * len(build_tc_grid(float(observations[0, t2_index])))
    return total_windows, total_fits


def run_backtest(
    close: pd.Series,
) -> tuple[
    pd.DataFrame,
    dict[pd.Timestamp, list[dict[str, Any]]],
    dict[pd.Timestamp, list[dict[str, Any]]],
]:
    observations = build_observations(close)
    model = LPPLSModifiedTC(observations=observations)
    full_window_sizes = build_window_sizes(close)
    min_history_bars = max(points_from_days(close, MIN_HISTORY_DAYS, minimum=20), full_window_sizes[0])
    analysis_indices = list(range(min_history_bars - 1, len(close), ANALYSIS_STEP_BARS))
    if analysis_indices[-1] != len(close) - 1:
        analysis_indices.append(len(close) - 1)

    print(
        f"Rolling backtest: {len(analysis_indices)} analysis points | "
        f"step={ANALYSIS_STEP_BARS} bars | window range={full_window_sizes[0]}..{full_window_sizes[-1]} pts"
    )
    approx_windows, approx_fits = estimate_backtest_workload(observations, full_window_sizes, analysis_indices)
    print(
        f"Speed mode: {BACKTEST_SPEED_MODE} | max_searches={MAX_SEARCHES} | "
        f"windows_per_scan<={MAX_WINDOWS_PER_SCAN} | estimated fits={approx_fits:,d}"
    )
    if len(analysis_indices):
        print(f"Average windows per scan: {approx_windows / len(analysis_indices):.1f}")

    rows: list[dict[str, Any]] = []
    fit_cache: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    interval_cache: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    total = len(analysis_indices)
    for pos, t2_index in enumerate(analysis_indices, start=1):
        t2_time = close.index[t2_index]
        current_t2 = float(observations[0, t2_index])
        tc_grid = build_tc_grid(current_t2)
        window_sizes = [ws for ws in full_window_sizes if ws <= (t2_index + 1)]
        window_sizes = thin_window_sizes(window_sizes, MAX_WINDOWS_PER_SCAN)
        if len(window_sizes) == 0:
            continue

        result = model.scan_tc_surface(
            t2_index=t2_index,
            window_sizes=window_sizes,
            tc_grid=tc_grid,
            max_searches=MAX_SEARCHES,
            cutoff=LIKELIHOOD_CUTOFF,
            peak_cutoff=PEAK_CUTOFF,
            constraints=FIT_CONSTRAINTS,
            random_state=11 + pos * 1000,
        )
        scenarios = result["scenarios"].copy()
        primary = build_primary_scenario_row(scenarios, close, t2_time, float(close.iloc[t2_index]))
        rows.append(primary)
        cached_fits = build_fit_cache_entry(result["surface"], primary)
        if len(cached_fits):
            fit_cache[pd.Timestamp(t2_time)] = cached_fits
        cached_intervals = build_interval_cache_entry(result["intervals"], primary)
        if len(cached_intervals):
            interval_cache[pd.Timestamp(t2_time)] = cached_intervals

        if pos == 1 or pos % 10 == 0 or pos == total:
            print(
                f"Progress {pos}/{total} | t2={t2_time} | "
                f"score={primary['signal_score']:.3f} | high_conf={primary['is_high_confidence']} | "
                f"scenario={primary['scenario_id']}"
            )

    state_df = pd.DataFrame(rows).set_index("t2_time").sort_index()
    if len(state_df):
        state_df = validate_prediction_rows(close, state_df)
    return state_df, fit_cache, interval_cache


def format_timestamp(value: Any) -> str:
    if pd.isna(value):
        return "NaT"
    ts = pd.Timestamp(value)
    return ts.round("s").strftime("%Y-%m-%d %H:%M:%S")


def format_float(value: Any, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "nan"
    try:
        val = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(val):
        return "nan"
    return f"{val:.{digits}f}"


def select_diverse_omega_surface_fits(surface: pd.DataFrame, row: dict[str, Any], max_curves: int = 5) -> list[dict[str, Any]]:
    if surface.empty:
        return []
    mask = (
        surface["qualified_conf"].astype(bool)
        & np.isfinite(surface["w"])
        & np.isfinite(surface["m"])
        & np.isfinite(surface["b"])
        & np.isfinite(surface["a"])
        & np.isfinite(surface["c1"])
        & np.isfinite(surface["c2"])
        & (surface["b"] < 0.0)
        & (surface["tc_time"] >= pd.Timestamp(row["interval_lo_time"]))
        & (surface["tc_time"] <= pd.Timestamp(row["interval_hi_time"]))
    )
    qualified = surface.loc[mask].copy()
    if qualified.empty:
        return []
    qualified = qualified.sort_values(["w", "rm", "window_size"], ascending=[True, False, True])
    if len(qualified) <= max_curves:
        return qualified.to_dict("records")
    picks = np.linspace(0, len(qualified) - 1, num=max_curves)
    return qualified.iloc[np.unique(np.round(picks).astype(int))].to_dict("records")


def build_lppls_curve_from_fit(fit: dict[str, Any], curve_points: int = 320) -> tuple[pd.DatetimeIndex, np.ndarray, pd.Timestamp]:
    t1 = float(fit["t1"])
    t2 = float(fit["t2"])
    tc = float(fit["tc"])
    if not np.isfinite(t1) or not np.isfinite(t2) or not np.isfinite(tc) or tc <= t1:
        raise ValueError("Invalid LPPLS fit range.")

    t_grid = np.linspace(t1, tc, curve_points)
    dt = np.maximum(np.abs(tc - t_grid), 1e-8)
    dt_m = np.power(dt, float(fit["m"]))
    log_dt = np.log(dt)
    y_grid = (
        float(fit["a"])
        + float(fit["b"]) * dt_m
        + float(fit["c1"]) * dt_m * np.cos(float(fit["w"]) * log_dt)
        + float(fit["c2"]) * dt_m * np.sin(float(fit["w"]) * log_dt)
    )
    price_grid = np.exp(np.asarray(y_grid, dtype=float))
    time_grid = pd.DatetimeIndex([pd.Timestamp.fromordinal(int(np.floor(val))) + pd.to_timedelta(val - np.floor(val), unit="D") for val in t_grid])
    return time_grid, price_grid, pd.Timestamp(fit["t2_time"])


def plot_backtest_overview(
    close: pd.Series,
    state_df: pd.DataFrame,
    zone_df: pd.DataFrame,
    fit_cache: dict[pd.Timestamp, list[dict[str, Any]]],
    interval_cache: dict[pd.Timestamp, list[dict[str, Any]]],
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PLOT_FIGSIZE, sharex=True, height_ratios=[3.0, 1.2])
    fig.patch.set_facecolor("#f8fafc")
    for ax in (ax1, ax2):
        ax.set_facecolor("#fbfdff")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.plot(close.index, close.values, linewidth=1.15, color="#2563eb", label="Price", zorder=2)
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle="--", alpha=0.22, color="#94a3b8")

    ax2.plot(state_df.index, state_df["signal_score"], linewidth=1.1, color="#0f766e", label="Signal score")
    ax2.axhline(MODEL_SCORE_THRESHOLD, linestyle="--", linewidth=1.0, color="#dc2626", label=f"High-conf threshold {MODEL_SCORE_THRESHOLD:.2f}")
    ax2.set_ylabel("Score")
    ax2.set_xlabel("Time")
    ax2.grid(True, linestyle="--", alpha=0.22, color="#94a3b8")

    if len(zone_df):
        zone_colors = plt.cm.OrRd(np.linspace(0.45, 0.9, max(len(zone_df), 1)))
        y_top = float(np.nanmax(close.values))
        y_bottom = float(np.nanmin(close.values))
        y_pad = max((y_top - y_bottom) * 0.05, 1.0)
        for idx, (entry_time, row) in enumerate(zone_df.iterrows(), start=1):
            color = zone_colors[idx - 1]
            ax1.axvspan(entry_time, row["exit_time"], color=color, alpha=0.10, zorder=0)
            ax2.axvspan(entry_time, row["exit_time"], color=color, alpha=0.06, zorder=0)
            ax1.scatter(
                entry_time,
                row["entry_price"],
                marker="^",
                s=68,
                color=color,
                edgecolors="black",
                linewidths=0.55,
                zorder=6,
            )
            ax1.text(
                entry_time,
                y_top + y_pad * (0.12 if idx % 2 else 0.28),
                f"Z{int(row['zone_id'])}",
                fontsize=8,
                color="#7f1d1d",
                ha="center",
                va="bottom",
                zorder=7,
            )

    high_conf = state_df[state_df["is_high_confidence"]].copy()
    score_scatter_top = None
    score_scatter_bottom = None
    cbar = None
    if len(high_conf):
        cmap = plt.get_cmap("viridis")
        score_scatter_top = ax1.scatter(
            high_conf.index,
            high_conf["t2_price"],
            c=high_conf["signal_score"],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=28,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.9,
            zorder=5,
            label="High-confidence short candidates",
        )
        score_scatter_top.set_pickradius(8)
        score_scatter_bottom = ax2.scatter(
            high_conf.index,
            high_conf["signal_score"],
            c=high_conf["signal_score"],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            s=28,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.9,
            zorder=5,
        )
        score_scatter_bottom.set_pickradius(8)

    annot_top = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="#94a3b8", alpha=0.35),
        fontsize=8,
    )
    annot_top.set_visible(False)
    annot_bottom = ax2.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="#94a3b8", alpha=0.35),
        fontsize=8,
    )
    annot_bottom.set_visible(False)
    fit_annot = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(12, -14),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="#94a3b8", alpha=0.28),
        fontsize=8,
    )
    fit_annot.set_visible(False)
    xaxis_axes_transform = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)

    active_artists: list[Any] = []
    active_fit_artists: list[Any] = []
    active_signal_time: Optional[pd.Timestamp] = None

    def clear_prediction_overlay() -> None:
        nonlocal active_artists
        for artist in active_artists:
            try:
                artist.remove()
            except Exception:
                pass
        active_artists = []

    def reset_fit_artist_style() -> None:
        for artist in active_fit_artists:
            meta = getattr(artist, "_lppls_meta", None)
            if meta is None:
                continue
            artist.set_alpha(meta["base_alpha"])
            artist.set_linewidth(meta["base_linewidth"])

    def clear_fit_bundle() -> None:
        nonlocal active_fit_artists, active_signal_time
        for artist in active_fit_artists:
            try:
                artist.remove()
            except Exception:
                pass
        active_fit_artists = []
        active_signal_time = None
        fit_annot.set_visible(False)

    def show_prediction_overlay(row: pd.Series) -> None:
        clear_prediction_overlay()
        lo_time = pd.Timestamp(row["interval_lo_time"])
        hi_time = pd.Timestamp(row["interval_hi_time"])
        peak_time = pd.Timestamp(row["peak_tc_time"])
        signal_time = pd.Timestamp(row["t2_time"])
        color = "#ef4444"
        active_artists.append(ax1.axvspan(lo_time, hi_time, color=color, alpha=0.10, zorder=1))
        active_artists.append(ax2.axvspan(lo_time, hi_time, color=color, alpha=0.08, zorder=1))
        active_artists.append(ax1.axvline(peak_time, color=color, linestyle="--", linewidth=1.1, alpha=0.9, zorder=7))
        active_artists.append(ax2.axvline(peak_time, color=color, linestyle="--", linewidth=1.1, alpha=0.9, zorder=7))

        raw_intervals = interval_cache.get(signal_time, [])
        if len(raw_intervals) == 0:
            return

        interval_colors = plt.cm.Blues(np.linspace(0.45, 0.88, max(len(raw_intervals), 1)))
        usable_height = 0.22
        step = usable_height / max(len(raw_intervals), 1)
        for idx, (interval_color, interval_row) in enumerate(zip(interval_colors, raw_intervals), start=1):
            y_frac = 0.96 - (idx - 0.5) * step
            line, = ax1.plot(
                [pd.Timestamp(interval_row["interval_lo_time"]), pd.Timestamp(interval_row["interval_hi_time"])],
                [y_frac, y_frac],
                color=interval_color,
                linewidth=2.0,
                alpha=0.75,
                transform=xaxis_axes_transform,
                solid_capstyle="round",
                zorder=8,
            )
            active_artists.append(line)
            peak_dot = ax1.scatter(
                [pd.Timestamp(interval_row["peak_tc_time"])],
                [y_frac],
                s=16,
                color=interval_color,
                edgecolors="white",
                linewidths=0.4,
                transform=xaxis_axes_transform,
                zorder=9,
            )
            active_artists.append(peak_dot)

    def show_signal_fit_bundle(row: pd.Series) -> None:
        nonlocal active_signal_time
        signal_time = pd.Timestamp(row["t2_time"])
        if active_signal_time == signal_time and len(active_fit_artists) > 0:
            return

        clear_fit_bundle()
        fits = fit_cache.get(signal_time, [])
        if len(fits) == 0:
            return

        price_ymax = float(np.nanmax(close.values)) * 1.08
        omega_colors = plt.cm.viridis(np.linspace(0.10, 0.90, max(len(fits), 1)))
        for fit_id, (fit_color, fit) in enumerate(zip(omega_colors, fits), start=1):
            try:
                curve_time, curve_price, fit_end_time = build_lppls_curve_from_fit(fit)
            except Exception:
                continue

            curve_price = np.asarray(curve_price, dtype=float)
            curve_price = np.where(
                np.isfinite(curve_price) & (curve_price > 0.0) & (curve_price <= price_ymax),
                curve_price,
                np.nan,
            )
            if np.all(np.isnan(curve_price)):
                continue

            in_sample_mask = curve_time <= fit_end_time
            future_mask = curve_time > fit_end_time
            line_meta = {
                "fit_id": fit_id,
                "signal_time": signal_time,
                "omega": float(fit["w"]),
                "m": float(fit["m"]),
                "b": float(fit["b"]),
                "tc": pd.Timestamp(fit["tc_time"]),
                "base_alpha": 0.5,
                "base_linewidth": 1.2,
            }

            hist_line, = ax1.plot(
                curve_time[in_sample_mask],
                curve_price[in_sample_mask],
                color=fit_color,
                linewidth=1.2,
                alpha=0.5,
                zorder=3,
            )
            hist_line._lppls_meta = line_meta
            active_fit_artists.append(hist_line)

            if np.any(future_mask):
                fut_line, = ax1.plot(
                    curve_time[future_mask],
                    curve_price[future_mask],
                    color=fit_color,
                    linewidth=1.2,
                    alpha=0.5,
                    linestyle="--",
                    zorder=3,
                )
                fut_line._lppls_meta = line_meta
                active_fit_artists.append(fut_line)

        active_signal_time = signal_time

    high_conf_points = high_conf.reset_index() if len(high_conf) else pd.DataFrame()

    def signal_text(idx: int) -> str:
        row = high_conf_points.iloc[idx]
        return (
            f"t2: {format_timestamp(row['t2_time'])}\n"
            f"price: {format_float(row['t2_price'], 4)}\n"
            f"score: {format_float(row['signal_score'], 3)}\n"
            f"peak tc: {format_timestamp(row['peak_tc_time'])}\n"
            f"interval: {format_timestamp(row['interval_lo_time'])}\n"
            f"          {format_timestamp(row['interval_hi_time'])}\n"
            f"horizon={format_float(row['horizon_days'], 1)}d  width={format_float(row['interval_width_days'], 1)}d\n"
            f"support={format_float(row['support_share'], 3)}  Rm={format_float(row['rm_max'], 3)}\n"
            f"raw LI windows={len(interval_cache.get(pd.Timestamp(row['t2_time']), []))}\n"
            f"runup={format_float(100.0 * row['runup_pct'], 1)}%  push={format_float(100.0 * row['final_push_pct'], 1)}%\n"
            f"m={format_float(row['m'], 3)}  omega={format_float(row['omega'], 3)}  D={format_float(row['D'], 3)}"
        )

    def hide_annotations() -> None:
        annot_top.set_visible(False)
        annot_bottom.set_visible(False)
        fit_annot.set_visible(False)

    def on_move(event: Any) -> None:
        if event.inaxes not in (ax1, ax2) or event.xdata is None or event.ydata is None:
            hide_annotations()
            clear_prediction_overlay()
            clear_fit_bundle()
            fig.canvas.draw_idle()
            return

        shown = False
        line_highlighted = False
        if score_scatter_top is not None:
            contains_top, top_info = score_scatter_top.contains(event)
            if contains_top and len(top_info.get("ind", [])) > 0:
                idx = int(top_info["ind"][0])
                row = high_conf_points.iloc[idx]
                show_prediction_overlay(row)
                show_signal_fit_bundle(row)
                annot_top.xy = (row["t2_time"], row["t2_price"])
                annot_top.set_text(signal_text(idx))
                annot_top.set_visible(True)
                annot_bottom.set_visible(False)
                fit_annot.set_visible(False)
                shown = True

        if (not shown) and score_scatter_bottom is not None:
            contains_bottom, bottom_info = score_scatter_bottom.contains(event)
            if contains_bottom and len(bottom_info.get("ind", [])) > 0:
                idx = int(bottom_info["ind"][0])
                row = high_conf_points.iloc[idx]
                show_prediction_overlay(row)
                show_signal_fit_bundle(row)
                annot_bottom.xy = (row["t2_time"], row["signal_score"])
                annot_bottom.set_text(signal_text(idx))
                annot_bottom.set_visible(True)
                annot_top.set_visible(False)
                fit_annot.set_visible(False)
                shown = True

        if not shown and event.inaxes == ax1 and len(active_fit_artists) > 0:
            reset_fit_artist_style()
            for artist in active_fit_artists:
                contains_line, _ = artist.contains(event)
                if not contains_line:
                    continue
                meta = getattr(artist, "_lppls_meta", None)
                if meta is None:
                    continue
                for other in active_fit_artists:
                    other_meta = getattr(other, "_lppls_meta", None)
                    if other_meta and other_meta["fit_id"] == meta["fit_id"]:
                        other.set_alpha(0.95)
                        other.set_linewidth(2.0)
                fit_annot.xy = (event.xdata, event.ydata)
                fit_annot.set_text(
                    f"omega={meta['omega']:.3f}\n"
                    f"m={meta['m']:.3f}\n"
                    f"B={meta['b']:.5f}\n"
                    f"tc={format_timestamp(meta['tc'])}"
                )
                fit_annot.set_visible(True)
                line_highlighted = True
                break

        if not shown:
            annot_top.set_visible(False)
            annot_bottom.set_visible(False)
            if line_highlighted:
                pass
            else:
                fit_annot.set_visible(False)
                clear_prediction_overlay()
                clear_fit_bundle()
                reset_fit_artist_style()

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    legend_handles = [
        Line2D([0], [0], color="#2563eb", linewidth=1.15, label="Price"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#84cc16",
            markeredgecolor="white",
            markersize=6,
            label="High-confidence short candidate",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#fb923c",
            markeredgecolor="black",
            markersize=7,
            label="Zone entry",
        ),
        Patch(facecolor="#fb923c", alpha=0.10, label="Bubble-entry zone"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")
    if score_scatter_top is not None:
        fig.tight_layout(rect=(0.0, 0.0, 0.935, 1.0))
        cax = fig.add_axes([0.947, 0.14, 0.014, 0.68])
        cbar = fig.colorbar(score_scatter_top, cax=cax)
        cbar.set_label("Model signal score")
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")


def print_summary(state_df: pd.DataFrame, zone_df: pd.DataFrame) -> None:
    high_conf = state_df[state_df["is_high_confidence"]]
    print(f"\nRolling predictions: {len(state_df)}")
    print(f"High-confidence predictions: {len(high_conf)}")
    print(f"Bubble-entry zones: {len(zone_df)}")

    if len(zone_df) == 0:
        print("\nNo bubble-entry zone passed the current signal threshold.")
        return

    print("\nBubble-entry zones:")
    for entry_time, row in zone_df.iterrows():
        dd_text = "nan" if not np.isfinite(row["forward_drawdown"]) else f"{row['forward_drawdown']:.2%}"
        breach_text = format_timestamp(row["first_breach_time"])
        print(
            f"\n[Z{int(row['zone_id'])}] entry={format_timestamp(entry_time)} | "
            f"exit={format_timestamp(row['exit_time'])} | "
            f"score={row['max_signal_score']:.3f} | "
            f"peak_tc={format_timestamp(row['peak_tc_time'])}"
        )
        print(
            f"  interval={format_timestamp(row['interval_lo_time'])} ~ {format_timestamp(row['interval_hi_time'])} | "
            f"support={row['support_share']:.1%} | Rm={row['rm_max']:.3f} | signals={int(row['n_signals'])}"
        )
        print(
            f"  ex-post drawdown={dd_text} | breach_time={breach_text} | validated_break={bool(row['validated_break'])}"
        )


def print_plot_guide(zone_df: pd.DataFrame) -> None:
    print("\n图中元素说明：")
    print("  1. 蓝色折线：重采样后的收盘价。")
    print("  2. 绿色圆点：高置信度预测点。每个圆点对应一个历史分析时点 t2。")
    print("     右侧色柱对应圆点颜色，范围是 0 到 1，表示模型信号分数 signal_score。")
    print("  3. 红色虚线：高置信度阈值。下方面板的分数高于这条线时，才有机会进入候选区。")
    print("  4. 橙色半透明竖向阴影：泡沫进入区。程序会将相邻的高置信度预测点合并成一个区间。")
    print("  5. 三角形：每个泡沫进入区的入场时点 entry_time。")
    print("  6. Z1、Z2、Z3：泡沫进入区编号 zone_id。标签写在各自区间的起点上方。")
    print("  7. 悬停在绿色圆点上时，图中会显示该次预测的聚合破裂区间 [interval_lo, interval_hi] 与 peak_tc。")
    print("  8. 同一次悬停还会显示一排蓝色短条。每一条对应一个窗口的原始 LI(tc)，圆点是该窗口的 peak_tc。")

    if len(zone_df) == 0:
        print("\n当前没有进入高置信度泡沫区的区间。")
        return

    print("\n当前泡沫进入区：")
    for entry_time, row in zone_df.iterrows():
        print(
            f"  Z{int(row['zone_id'])}: entry={format_timestamp(entry_time)} | "
            f"exit={format_timestamp(row['exit_time'])} | "
            f"peak_tc={format_timestamp(row['peak_tc_time'])} | "
            f"signals={int(row['n_signals'])}"
        )


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    input_path, using_converted = resolve_input_data_path()
    print(f"Configured raw data file: {DATA_FILE_PATH}")
    print(f"Resolved input file: {input_path}")

    df_loaded = read_ohlcv_file(input_path)
    df_loaded = apply_date_range(df_loaded, START_DATE, END_DATE)

    if using_converted:
        print(f"Using converted data file for rule {RESAMPLE_RULE}.")
        df = df_loaded
        print(f"Converted rows: {len(df):,d} | range: {df.index.min()} ~ {df.index.max()}")
    else:
        print(f"Converted data file not found for rule {RESAMPLE_RULE}. Resampling raw data in memory.")
        print(f"Raw rows: {len(df_loaded):,d} | range: {df_loaded.index.min()} ~ {df_loaded.index.max()}")
        df = resample_ohlcv(df_loaded, RESAMPLE_RULE)

    close = df["close"].dropna().copy()
    print(
        f"Profile: {SCAN_PROFILE} | Resampled rule: {RESAMPLE_RULE} | "
        f"rows: {len(df):,d} | range: {df.index.min()} ~ {df.index.max()}"
    )
    print(
        f"Backtest setup: mode={BACKTEST_SPEED_MODE} | analysis_step={ANALYSIS_STEP_BARS} bars | "
        f"score_threshold={MODEL_SCORE_THRESHOLD:.2f} | "
        f"horizon_pref={PREFERRED_HORIZON_DAYS[0]:.0f}..{PREFERRED_HORIZON_DAYS[1]:.0f}d"
    )

    state_df, fit_cache, interval_cache = run_backtest(close)
    median_bar_days = close.index.to_series().diff().dropna().dt.total_seconds().median() / 86400.0
    analysis_gap = pd.Timedelta(days=float(median_bar_days) * ANALYSIS_STEP_BARS)
    zone_df = build_bubble_zones(state_df, close, analysis_gap)
    print_summary(state_df, zone_df)
    print_plot_guide(zone_df)

    program_name = Path(__file__).resolve().stem
    out_dir = Path(__file__).resolve().parent / RESULT_DIR_NAME / program_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_tag = make_output_tag(close)
    states_path = out_dir / f"tc_backtest_states_{output_tag}.csv"
    zones_path = out_dir / f"tc_backtest_zones_{output_tag}.csv"
    figure_path = out_dir / f"lppls_backtest_overview_{output_tag}.png"

    state_df.to_csv(states_path, encoding="utf-8-sig")
    zone_df.to_csv(zones_path, encoding="utf-8-sig")
    plot_backtest_overview(close, state_df, zone_df, fit_cache, interval_cache, figure_path)

    print(f"\nOutput dir: {out_dir}")
    print(f"Generated figure: {figure_path.name}")
    print(f"Generated states: {states_path.name}")
    print(f"Generated zones: {zones_path.name}")
    plt.show()


if __name__ == "__main__":
    main()
