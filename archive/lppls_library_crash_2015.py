from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lppls import lppls

# =========================
# Paths and core settings
# =========================
DATA_FILE_PATH = Path(__file__).resolve().parent / "data" / "sample_xagusd_2h.csv"
OUTPUT_DIR_NAME = "lppls_lib_output"

SCAN_PROFILE = "bull_year"  # "crash_local" | "bull_year"
MAX_SEARCHES = 8
WORKERS = max(1, min(4, (os.cpu_count() or 1)))

if SCAN_PROFILE == "crash_local":
    RESAMPLE_RULE = "2h"
    WINDOW_SIZE_DAYS = 60.0
    SMALLEST_WINDOW_DAYS = 20.0
    OUTER_INCREMENT_DAYS = 1.0
    INNER_INCREMENT_DAYS = 2.0
elif SCAN_PROFILE == "bull_year":
    RESAMPLE_RULE = "1D"
    WINDOW_SIZE_DAYS = 300.0
    SMALLEST_WINDOW_DAYS = 120.0
    OUTER_INCREMENT_DAYS = 7.0
    INNER_INCREMENT_DAYS = 3.0
else:
    raise ValueError(f"Unknown SCAN_PROFILE: {SCAN_PROFILE}")

PLOT_PRICE_CAP_FACTOR = 1.35
PLOT_PRICE_YMAX: Optional[float] = None

# Bubble and crash filters, still targeted at 2026-01-30 style blow-off + crash.
POS_CONF_THRESHOLD = 0.30
PREDICTION_CONFIDENCE = 0.50
PREDICTION_MIN_GAP_DAYS = 8.0
EVENT_SELECTION_MODE = "earliest_tc"  # 'first' | 'peak' | 'last' | 'earliest_tc'

RUNUP_LOOKBACK_DAYS = 20.0
MIN_RUNUP_PCT = 0.35
FINAL_PUSH_DAYS = 5.0
MIN_FINAL_PUSH_PCT = 0.10
MAJOR_CRASH_DAYS = 3.0
MAJOR_CRASH_THRESHOLD = -0.18


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


def trailing_runup_from_low(close: pd.Series, end_time: pd.Timestamp, lookback_days: float) -> float:
    seg = close.loc[end_time - pd.Timedelta(days=float(lookback_days)):end_time]
    if len(seg) < 2:
        return np.nan
    low = float(seg.min())
    end_price = float(seg.iloc[-1])
    if low <= 0:
        return np.nan
    return end_price / low - 1.0


def trailing_window_return(close: pd.Series, end_time: pd.Timestamp, lookback_days: float) -> float:
    seg = close.loc[end_time - pd.Timedelta(days=float(lookback_days)):end_time]
    if len(seg) < 2:
        return np.nan
    start_price = float(seg.iloc[0])
    end_price = float(seg.iloc[-1])
    if start_price <= 0:
        return np.nan
    return end_price / start_price - 1.0


def forward_drawdown(close: pd.Series, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
    seg = close.loc[start_time:end_time]
    if len(seg) < 2:
        return np.nan
    p0 = float(seg.iloc[0])
    pmin = float(seg.min())
    if p0 <= 0:
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
    if p0 <= 0:
        return pd.NaT
    dd = seg / p0 - 1.0
    hit = dd[dd <= float(drawdown_threshold)]
    return hit.index[0] if len(hit) else pd.NaT


def build_observations(close: pd.Series) -> tuple[np.ndarray, pd.Timestamp]:
    t0 = close.index[0]
    time_days = (close.index - t0).total_seconds() / 86400.0
    log_price = np.log(close.to_numpy(dtype=float))
    observations = np.array([time_days.to_numpy(dtype=float), log_price], dtype=float)
    return observations, t0


def points_from_days(close: pd.Series, days: float, minimum: int = 1) -> int:
    dt_days = close.index.to_series().diff().dropna().dt.total_seconds().median() / 86400.0
    points = int(round(days / dt_days))
    return max(minimum, points)


def indicator_row_to_summary(row: pd.Series, t0: pd.Timestamp) -> Optional[dict[str, Any]]:
    fits = row["_fits"]
    qualified_pos = [
        f for f in fits
        if f.get("is_qualified") and float(f.get("b", 0)) < 0 and np.isfinite(float(f.get("tc", np.nan)))
    ]
    if len(qualified_pos) == 0:
        return None

    tc_vals = np.array([float(f["tc"]) for f in qualified_pos], dtype=float)
    m_vals = np.array([float(f["m"]) for f in qualified_pos], dtype=float)
    w_vals = np.array([float(f["w"]) for f in qualified_pos], dtype=float)
    b_vals = np.array([float(f["b"]) for f in qualified_pos], dtype=float)

    tc_med = float(np.median(tc_vals))
    tc_q10 = float(np.quantile(tc_vals, 0.10))
    tc_q90 = float(np.quantile(tc_vals, 0.90))
    signal_time = t0 + pd.to_timedelta(float(row["time"]), unit="D")

    return {
        "signal_time": signal_time,
        "signal_price": float(np.exp(row["price"])),
        "confidence": float(row["pos_conf"]),
        "n_valid": int(len(qualified_pos)),
        "n_success": int(len([f for f in fits if float(f.get("b", 0)) < 0])),
        "tc_median_time": t0 + pd.to_timedelta(tc_med, unit="D"),
        "tc_q10_time": t0 + pd.to_timedelta(tc_q10, unit="D"),
        "tc_q90_time": t0 + pd.to_timedelta(tc_q90, unit="D"),
        "m_median": float(np.median(m_vals)),
        "omega_median": float(np.median(w_vals)),
        "B_median": float(np.median(b_vals)),
        "_fits": fits,
    }


def summarize_signal_fits_text(
    fits: list[dict[str, Any]],
    t0: pd.Timestamp,
    signal_time: pd.Timestamp,
) -> str:
    qualified = [
        f for f in fits
        if f.get("is_qualified")
        and float(f.get("b", 0.0)) < 0.0
        and np.isfinite(float(f.get("tc", np.nan)))
        and np.isfinite(float(f.get("w", np.nan)))
        and np.isfinite(float(f.get("m", np.nan)))
        and np.isfinite(float(f.get("t1", np.nan)))
        and np.isfinite(float(f.get("t2", np.nan)))
    ]
    if len(qualified) == 0:
        return "No qualified positive fits for this signal."

    tc_vals = np.array([float(f["tc"]) for f in qualified], dtype=float)
    w_vals = np.array([float(f["w"]) for f in qualified], dtype=float)
    m_vals = np.array([float(f["m"]) for f in qualified], dtype=float)
    b_vals = np.array([float(f["b"]) for f in qualified], dtype=float)

    def _ts_from_days(day_value: float) -> pd.Timestamp:
        return t0 + pd.to_timedelta(day_value, unit="D")

    rows = []
    qualified_sorted = sorted(qualified, key=lambda f: (float(f["tc"]), float(f["w"])))
    for idx, fit in enumerate(qualified_sorted, start=1):
        t1 = float(fit["t1"])
        t2 = float(fit["t2"])
        tc = float(fit["tc"])
        t1_time = _ts_from_days(t1)
        t2_time = _ts_from_days(t2)
        tc_time = _ts_from_days(tc)
        rows.append(
            f"{idx:02d}. win={t2 - t1:7.2f}d  "
            f"end={t2_time.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"tc={tc_time.strftime('%Y-%m-%d %H:%M:%S')}  "
            f"lead={tc - t2:6.2f}d  "
            f"omega={float(fit['w']):6.3f}  "
            f"m={float(fit['m']):5.3f}  "
            f"B={float(fit['b']): .6f}"
        )

    summary = [
        "",
        "=" * 88,
        f"Signal fit distribution | signal={signal_time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Qualified positive fits: {len(qualified)} / total fits: {len(fits)}",
        (
            "tc days   : "
            f"min={np.min(tc_vals):.2f}, q10={np.quantile(tc_vals, 0.10):.2f}, "
            f"median={np.median(tc_vals):.2f}, q90={np.quantile(tc_vals, 0.90):.2f}, "
            f"max={np.max(tc_vals):.2f}"
        ),
        (
            "tc times  : "
            f"min={_ts_from_days(np.min(tc_vals)).strftime('%Y-%m-%d %H:%M:%S')}, "
            f"median={_ts_from_days(np.median(tc_vals)).strftime('%Y-%m-%d %H:%M:%S')}, "
            f"max={_ts_from_days(np.max(tc_vals)).strftime('%Y-%m-%d %H:%M:%S')}"
        ),
        (
            "omega     : "
            f"min={np.min(w_vals):.3f}, q10={np.quantile(w_vals, 0.10):.3f}, "
            f"median={np.median(w_vals):.3f}, q90={np.quantile(w_vals, 0.90):.3f}, "
            f"max={np.max(w_vals):.3f}"
        ),
        (
            "m         : "
            f"min={np.min(m_vals):.3f}, q10={np.quantile(m_vals, 0.10):.3f}, "
            f"median={np.median(m_vals):.3f}, q90={np.quantile(m_vals, 0.90):.3f}, "
            f"max={np.max(m_vals):.3f}"
        ),
        (
            "B         : "
            f"min={np.min(b_vals):.6f}, q10={np.quantile(b_vals, 0.10):.6f}, "
            f"median={np.median(b_vals):.6f}, q90={np.quantile(b_vals, 0.90):.6f}, "
            f"max={np.max(b_vals):.6f}"
        ),
        "-" * 88,
        "Qualified fits sorted by tc:",
        *rows,
        "=" * 88,
    ]
    return "\n".join(summary)


def build_prediction_events(summary_df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    candidates = summary_df[
        (summary_df["confidence"] >= PREDICTION_CONFIDENCE) &
        summary_df["tc_median_time"].notna()
    ].copy().sort_index()

    if len(candidates) == 0:
        return pd.DataFrame(columns=[
            "event_id", "signal_time", "signal_price", "confidence", "n_valid", "n_success",
            "tc_median_time", "tc_q10_time", "tc_q90_time", "m_median", "omega_median",
            "B_median", "runup_pct", "final_push_pct", "major_crash_drawdown",
            "major_break_time", "selection_mode", "reason",
        ]).set_index("signal_time")

    groups = []
    current = []
    prev_time = None
    gap = pd.Timedelta(days=PREDICTION_MIN_GAP_DAYS)
    for signal_time, row in candidates.iterrows():
        if prev_time is None or (signal_time - prev_time) <= gap:
            current.append((signal_time, row))
        else:
            groups.append(current)
            current = [(signal_time, row)]
        prev_time = signal_time
    if current:
        groups.append(current)

    last_time = close.index[-1]
    rows: list[dict[str, Any]] = []
    for event_id, group in enumerate(groups, start=1):
        peak_time, peak_row = max(group, key=lambda x: (float(x[1]["confidence"]), x[0].value))
        if EVENT_SELECTION_MODE == "first":
            signal_time, signal_row = group[0]
        elif EVENT_SELECTION_MODE == "last":
            signal_time, signal_row = group[-1]
        elif EVENT_SELECTION_MODE == "peak":
            signal_time, signal_row = peak_time, peak_row
        elif EVENT_SELECTION_MODE == "earliest_tc":
            signal_time, signal_row = min(group, key=lambda x: (x[1]["tc_median_time"].value, -x[0].value))
        else:
            raise ValueError(f"Unknown EVENT_SELECTION_MODE: {EVENT_SELECTION_MODE}")

        runup_pct = trailing_runup_from_low(close, signal_time, RUNUP_LOOKBACK_DAYS)
        final_push_pct = trailing_window_return(close, signal_time, FINAL_PUSH_DAYS)
        is_large_setup = (
            np.isfinite(runup_pct) and np.isfinite(final_push_pct) and
            runup_pct >= MIN_RUNUP_PCT and final_push_pct >= MIN_FINAL_PUSH_PCT
        )
        if not is_large_setup:
            continue

        crash_end = min(signal_time + pd.Timedelta(days=MAJOR_CRASH_DAYS), last_time)
        major_crash_drawdown = forward_drawdown(close, signal_time, crash_end)
        major_break_time = first_drawdown_breach_time(close, signal_time, crash_end, MAJOR_CRASH_THRESHOLD)
        has_crash_window = signal_time + pd.Timedelta(days=MAJOR_CRASH_DAYS) <= last_time
        if has_crash_window and not (
            np.isfinite(major_crash_drawdown) and major_crash_drawdown <= MAJOR_CRASH_THRESHOLD
        ):
            continue

        reason = (
            f"pos_conf={signal_row['confidence']:.3f}, "
            f"runup={runup_pct:.2%}, final_push={final_push_pct:.2%}, "
            f"major_crash={major_crash_drawdown:.2%}, "
            f"tc={signal_row['tc_median_time']} "
            f"[{signal_row['tc_q10_time']} ~ {signal_row['tc_q90_time']}], "
            f"m={signal_row['m_median']:.3f}, "
            f"omega={signal_row['omega_median']:.3f}, "
            f"B={signal_row['B_median']:.6f}, "
            f"cluster_peak={peak_time} ({peak_row['confidence']:.3f})"
        )

        rows.append({
            "event_id": event_id,
            "signal_time": signal_time,
            "signal_price": float(signal_row["signal_price"]),
            "confidence": float(signal_row["confidence"]),
            "n_valid": int(signal_row["n_valid"]),
            "n_success": int(signal_row["n_success"]),
            "tc_median_time": signal_row["tc_median_time"],
            "tc_q10_time": signal_row["tc_q10_time"],
            "tc_q90_time": signal_row["tc_q90_time"],
            "m_median": float(signal_row["m_median"]),
            "omega_median": float(signal_row["omega_median"]),
            "B_median": float(signal_row["B_median"]),
            "runup_pct": float(runup_pct),
            "final_push_pct": float(final_push_pct),
            "major_crash_drawdown": float(major_crash_drawdown) if np.isfinite(major_crash_drawdown) else np.nan,
            "major_break_time": major_break_time,
            "selection_mode": EVENT_SELECTION_MODE,
            "reason": reason,
        })

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "event_id", "signal_time", "signal_price", "confidence", "n_valid", "n_success",
            "tc_median_time", "tc_q10_time", "tc_q90_time", "m_median", "omega_median",
            "B_median", "runup_pct", "final_push_pct", "major_crash_drawdown",
            "major_break_time", "selection_mode", "reason",
        ]).set_index("signal_time")

    return pd.DataFrame(rows).set_index("signal_time").sort_index()


def select_diverse_omega_fits(fits: list[dict[str, Any]], max_curves: int = 5) -> list[dict[str, Any]]:
    qualified = [
        fit for fit in fits
        if fit.get("is_qualified")
        and float(fit.get("b", 0.0)) < 0.0
        and np.isfinite(float(fit.get("tc", np.nan)))
        and np.isfinite(float(fit.get("w", np.nan)))
        and np.isfinite(float(fit.get("a", np.nan)))
        and np.isfinite(float(fit.get("c1", np.nan)))
        and np.isfinite(float(fit.get("c2", np.nan)))
    ]
    if len(qualified) <= max_curves:
        return sorted(qualified, key=lambda x: float(x["w"]))

    qualified = sorted(qualified, key=lambda x: float(x["w"]))
    picks = np.linspace(0, len(qualified) - 1, max_curves, dtype=int)
    chosen: list[dict[str, Any]] = []
    seen = set()
    for idx in picks:
        if idx not in seen:
            chosen.append(qualified[idx])
            seen.add(idx)
    return chosen


def build_lppls_curve(
    fit: dict[str, Any],
    t0: pd.Timestamp,
    curve_points: int = 320,
) -> tuple[pd.DatetimeIndex, np.ndarray, pd.Timestamp]:
    t1 = float(fit["t1"])
    t2 = float(fit["t2"])
    tc = float(fit["tc"])
    if tc <= t1:
        raise ValueError("tc must be greater than t1 to build LPPLS curve.")

    t_grid = np.linspace(t1, tc, curve_points)
    y_grid = lppls.LPPLS.lppls(
        t_grid,
        tc,
        float(fit["m"]),
        float(fit["w"]),
        float(fit["a"]),
        float(fit["b"]),
        float(fit["c1"]),
        float(fit["c2"]),
    )
    price_grid = np.exp(np.asarray(y_grid, dtype=float))
    time_grid = t0 + pd.to_timedelta(t_grid, unit="D")
    t2_time = t0 + pd.to_timedelta(t2, unit="D")
    return time_grid, price_grid, t2_time


def resolve_plot_price_cap(close: pd.Series) -> float:
    base_cap = float(np.nanmax(close.to_numpy(dtype=float))) * float(PLOT_PRICE_CAP_FACTOR)
    if PLOT_PRICE_YMAX is None:
        return base_cap
    return min(float(PLOT_PRICE_YMAX), base_cap)


def plot_overview(close: pd.Series, summary_df: pd.DataFrame, prediction_events: pd.DataFrame, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, height_ratios=[3, 1])
    ax1.plot(close.index, close.values, linewidth=1.0, color="tab:blue")
    ax1.set_title("lppls 0.6.23 Crash-Focused Scan")
    ax1.set_ylabel("Price")
    price_ymax = resolve_plot_price_cap(close)
    price_ymin = max(0.0, float(np.nanmin(close.to_numpy(dtype=float))) * 0.95)
    ax1.set_ylim(price_ymin, price_ymax)

    bubble = summary_df[summary_df["confidence"] >= POS_CONF_THRESHOLD].copy()
    signal_scatter_top = None
    signal_scatter_bottom = None
    if len(bubble):
        signal_scatter_top = ax1.scatter(
            bubble.index,
            bubble["signal_price"],
            s=28,
            color="tab:green",
            alpha=0.65,
            label="Positive LPPLS signal",
            zorder=4,
        )
        signal_scatter_top.set_pickradius(8)

    ax2.plot(summary_df.index, summary_df["confidence"], linewidth=1.0, color="tab:blue", label="pos_conf")
    ax2.axhline(POS_CONF_THRESHOLD, linestyle="--", linewidth=1.0, color="tab:blue", label=f"base {POS_CONF_THRESHOLD:.2f}")
    ax2.axhline(PREDICTION_CONFIDENCE, linestyle="--", linewidth=1.0, color="tab:red", label=f"event {PREDICTION_CONFIDENCE:.2f}")
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Time")
    if len(bubble):
        signal_scatter_bottom = ax2.scatter(
            bubble.index,
            bubble["confidence"],
            s=28,
            color="tab:green",
            alpha=0.65,
            zorder=4,
        )
        signal_scatter_bottom.set_pickradius(8)

    t0 = close.index[0]

    ax1.text(
        0.01,
        0.98,
        "Hover a green signal to show LPPL fits. Hover a fit line to highlight it.",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.35",
        bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85),
        zorder=10,
    )

    if len(prediction_events):
        colors = plt.cm.get_cmap("tab10", max(len(prediction_events), 1))
        y_top = min(float(np.nanmax(close.values)), price_ymax)
        y_bottom = max(float(np.nanmin(close.values)), price_ymin)
        y_pad = (y_top - y_bottom) * 0.04 if y_top > y_bottom else 1.0
        for idx, (signal_time, row) in enumerate(prediction_events.iterrows(), start=1):
            color = colors(idx - 1)
            ax1.axvline(row["tc_median_time"], color=color, linestyle="--", linewidth=1.8, label=f"Crash event {int(row['event_id'])}")
            ax1.axvspan(row["tc_q10_time"], row["tc_q90_time"], color=color, alpha=0.08)
            ax1.scatter(signal_time, row["signal_price"], color=color, marker="s", s=60, edgecolors="black", linewidths=0.5, zorder=5)
            ax1.annotate(f"E{int(row['event_id'])}", xy=(row["tc_median_time"], y_top + y_pad), xytext=(2, -2),
                         textcoords="offset points", rotation=90, va="top", ha="left", fontsize=8, color=color)
            ax2.scatter(signal_time, row["confidence"], color=color, s=50, zorder=4)

    tc_vline_top = ax1.axvline(close.index[0], color="magenta", linestyle="--", linewidth=0.8, alpha=0.85, visible=False, zorder=9)
    tc_vline_bottom = ax2.axvline(close.index[0], color="magenta", linestyle="--", linewidth=0.8, alpha=0.85, visible=False, zorder=9)

    annot_top = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.95),
        fontsize=8,
    )
    annot_top.set_visible(False)
    annot_bottom = ax2.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.95),
        fontsize=8,
    )
    annot_bottom.set_visible(False)
    fit_annot = ax1.annotate(
        "",
        xy=(0, 0),
        xytext=(12, -14),
        textcoords="offset points",
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.95),
        fontsize=8,
    )
    fit_annot.set_visible(False)

    bubble_points = bubble.reset_index() if len(bubble) else pd.DataFrame()
    active_fit_artists: list[Any] = []
    active_signal_idx: Optional[int] = None
    last_logged_signal_idx: Optional[int] = None

    def format_timestamp(ts: Any) -> str:
        if pd.isna(ts):
            return "NaT"
        try:
            stamp = pd.Timestamp(ts).round("s")
        except Exception:
            return str(ts)
        if pd.isna(stamp):
            return "NaT"
        return stamp.strftime("%Y-%m-%d %H:%M:%S")

    def format_float(value: Any, digits: int = 3) -> str:
        if value is None or not np.isfinite(float(value)):
            return "nan"
        return f"{float(value):.{digits}f}"

    def format_signal_text(idx: int) -> str:
        row = bubble_points.iloc[idx]
        return (
            f"time: {format_timestamp(row['signal_time'])}\n"
            f"price: {format_float(row['signal_price'], 4)}\n"
            f"pos_conf: {format_float(row['confidence'], 3)}\n"
            f"valid fits: {int(row['n_valid'])}/{max(int(row['n_success']), 1)}\n"
            f"tc: {format_timestamp(row['tc_median_time'])}\n"
            f"tc band: {format_timestamp(row['tc_q10_time'])}\n"
            f"        {format_timestamp(row['tc_q90_time'])}\n"
            f"m={format_float(row['m_median'], 3)}, "
            f"omega={format_float(row['omega_median'], 3)}, "
            f"B={format_float(row['B_median'], 5)}"
        )

    def hide_annotations() -> None:
        annot_top.set_visible(False)
        annot_bottom.set_visible(False)
        fit_annot.set_visible(False)

    def reset_fit_artist_style() -> None:
        for artist in active_fit_artists:
            meta = getattr(artist, "_lppls_meta", None)
            if meta is None:
                continue
            artist.set_alpha(meta["base_alpha"])
            artist.set_linewidth(meta["base_linewidth"])

    def clear_dynamic_fit_artists() -> None:
        nonlocal active_fit_artists, active_signal_idx
        for artist in active_fit_artists:
            artist.remove()
        active_fit_artists = []
        active_signal_idx = None
        fit_annot.set_visible(False)

    def log_signal_fit_distribution(idx: int) -> None:
        nonlocal last_logged_signal_idx
        if last_logged_signal_idx == idx:
            return
        row = bubble_points.iloc[idx]
        fits = row.get("_fits", [])
        if not isinstance(fits, list):
            return
        print(summarize_signal_fits_text(fits, t0, pd.Timestamp(row["signal_time"])))
        last_logged_signal_idx = idx

    def show_signal_fit_bundle(idx: int) -> None:
        nonlocal active_fit_artists, active_signal_idx
        if active_signal_idx == idx and len(active_fit_artists) > 0:
            return

        clear_dynamic_fit_artists()
        row = bubble_points.iloc[idx]
        fits = row.get("_fits", [])
        if not isinstance(fits, list):
            return

        omega_fits = select_diverse_omega_fits(fits, max_curves=5)
        omega_colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(omega_fits), 1)))
        for fit_id, (fit_color, fit) in enumerate(zip(omega_colors, omega_fits), start=1):
            try:
                curve_time, curve_price, fit_end_time = build_lppls_curve(fit, t0)
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
                "signal_idx": idx,
                "omega": float(fit["w"]),
                "m": float(fit["m"]),
                "b": float(fit["b"]),
                "tc": t0 + pd.to_timedelta(float(fit["tc"]), unit="D"),
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

        active_signal_idx = idx

    def on_move(event) -> None:
        if event.inaxes not in (ax1, ax2) or event.xdata is None or event.ydata is None:
            tc_vline_top.set_visible(False)
            tc_vline_bottom.set_visible(False)
            hide_annotations()
            clear_dynamic_fit_artists()
            fig.canvas.draw_idle()
            return

        shown = False
        if signal_scatter_top is not None:
            contains_top, top_info = signal_scatter_top.contains(event)
            if contains_top and len(top_info.get("ind", [])) > 0:
                idx = int(top_info["ind"][0])
                tc_time = bubble["tc_median_time"].iloc[idx]
                show_signal_fit_bundle(idx)
                log_signal_fit_distribution(idx)
                annot_top.xy = (bubble.index[idx], bubble["signal_price"].iloc[idx])
                annot_top.set_text(format_signal_text(idx))
                annot_top.set_visible(True)
                annot_bottom.set_visible(False)
                fit_annot.set_visible(False)
                if pd.notna(tc_time):
                    tc_vline_top.set_xdata([tc_time, tc_time])
                    tc_vline_bottom.set_xdata([tc_time, tc_time])
                    tc_vline_top.set_visible(True)
                    tc_vline_bottom.set_visible(True)
                shown = True

        if (not shown) and signal_scatter_bottom is not None:
            contains_bottom, bottom_info = signal_scatter_bottom.contains(event)
            if contains_bottom and len(bottom_info.get("ind", [])) > 0:
                idx = int(bottom_info["ind"][0])
                tc_time = bubble["tc_median_time"].iloc[idx]
                show_signal_fit_bundle(idx)
                log_signal_fit_distribution(idx)
                annot_bottom.xy = (bubble.index[idx], bubble["confidence"].iloc[idx])
                annot_bottom.set_text(format_signal_text(idx))
                annot_bottom.set_visible(True)
                annot_top.set_visible(False)
                fit_annot.set_visible(False)
                if pd.notna(tc_time):
                    tc_vline_top.set_xdata([tc_time, tc_time])
                    tc_vline_bottom.set_xdata([tc_time, tc_time])
                    tc_vline_top.set_visible(True)
                    tc_vline_bottom.set_visible(True)
                shown = True

        line_highlighted = False
        if not shown and event.inaxes == ax1 and len(active_fit_artists) > 0:
            reset_fit_artist_style()
            for artist in active_fit_artists:
                contains_line, _ = artist.contains(event)
                if contains_line:
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
                        f"ω={meta['omega']:.3f}\n"
                        f"m={meta['m']:.3f}\n"
                        f"B={meta['b']:.5f}\n"
                        f"tc={format_timestamp(meta['tc'])}"
                    )
                    fit_annot.set_text(
                        f"omega={meta['omega']:.3f}\n"
                        f"m={meta['m']:.3f}\n"
                        f"B={meta['b']:.5f}\n"
                        f"tc={format_timestamp(meta['tc'])}"
                    )
                    fit_annot.set_visible(True)
                    line_highlighted = True
                    break
            if not line_highlighted:
                fit_annot.set_visible(False)

        if not shown:
            annot_top.set_visible(False)
            annot_bottom.set_visible(False)
            tc_vline_top.set_visible(False)
            tc_vline_bottom.set_visible(False)
            if not line_highlighted:
                fit_annot.set_visible(False)
                reset_fit_artist_style()

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)

    ax1.legend(loc="best")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)


def print_events(prediction_events: pd.DataFrame) -> None:
    if len(prediction_events) == 0:
        print("\nNo crash-focused LPPLS events matched the filters.")
        return

    print("\nCrash-focused LPPLS events:")
    for signal_time, row in prediction_events.iterrows():
        print(
            f"\n[Event {int(row['event_id'])}] signal={signal_time} "
            f"price={row['signal_price']:.4f} pos_conf={row['confidence']:.3f} "
            f"({int(row['n_valid'])}/{max(int(row['n_success']), 1)})"
        )
        print(f"  tc median: {row['tc_median_time']}")
        print(f"  tc q10~q90: {row['tc_q10_time']} ~ {row['tc_q90_time']}")
        print(
            f"  params: m={row['m_median']:.3f}, "
            f"omega={row['omega_median']:.3f}, B={row['B_median']:.6f}"
        )
        print(
            f"  filters: runup={row['runup_pct']:.2%}, "
            f"final_push={row['final_push_pct']:.2%}, "
            f"major_crash={row['major_crash_drawdown']:.2%}"
        )
        if pd.notna(row["major_break_time"]):
            print(f"  major crash first breach: {row['major_break_time']}")
        else:
            print("  major crash first breach: not seen")
        print(f"  reason: {row['reason']}")


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    print(f"Using lppls from: {lppls.__file__}")
    print(f"Data file: {DATA_FILE_PATH}")
    df_raw = read_ohlcv_file(DATA_FILE_PATH)
    print(f"Raw rows: {len(df_raw):,d} | range: {df_raw.index.min()} ~ {df_raw.index.max()}")

    df = resample_ohlcv(df_raw, RESAMPLE_RULE)
    close = df["close"].dropna().copy()
    print(
        f"Profile: {SCAN_PROFILE} | Resampled rule: {RESAMPLE_RULE} | "
        f"rows: {len(df):,d} | range: {df.index.min()} ~ {df.index.max()}"
    )
    print(f"Plot price cap: {resolve_plot_price_cap(close):.4f}")

    observations, t0 = build_observations(close)
    model = lppls.LPPLS(observations=observations)

    window_size = points_from_days(close, WINDOW_SIZE_DAYS, minimum=50)
    smallest_window_size = points_from_days(close, SMALLEST_WINDOW_DAYS, minimum=30)
    outer_increment = points_from_days(close, OUTER_INCREMENT_DAYS, minimum=1)
    inner_increment = points_from_days(close, INNER_INCREMENT_DAYS, minimum=1)

    print(
        f"Nested fits: window={window_size} pts, smallest={smallest_window_size} pts, "
        f"outer_inc={outer_increment}, inner_inc={inner_increment}, "
        f"workers={WORKERS}, max_searches={MAX_SEARCHES}"
    )

    res = model.mp_compute_nested_fits(
        workers=WORKERS,
        window_size=window_size,
        smallest_window_size=smallest_window_size,
        outer_increment=outer_increment,
        inner_increment=inner_increment,
        max_searches=MAX_SEARCHES,
    )

    indicators = model.compute_indicators(res)
    summary_rows = []
    for _, row in indicators.iterrows():
        summary = indicator_row_to_summary(row, t0)
        if summary is not None:
            summary_rows.append(summary)

    if len(summary_rows) == 0:
        summary_df = pd.DataFrame(columns=[
            "signal_time", "signal_price", "confidence", "n_valid", "n_success",
            "tc_median_time", "tc_q10_time", "tc_q90_time", "m_median",
            "omega_median", "B_median", "_fits",
        ]).set_index("signal_time")
    else:
        summary_df = pd.DataFrame(summary_rows).set_index("signal_time").sort_index()
    prediction_events = build_prediction_events(summary_df, close)

    print(
        f"Filters: pos_conf >= {PREDICTION_CONFIDENCE:.2f}, "
        f"runup >= {MIN_RUNUP_PCT:.0%} over {RUNUP_LOOKBACK_DAYS:.0f}d, "
        f"final push >= {MIN_FINAL_PUSH_PCT:.0%} over {FINAL_PUSH_DAYS:.0f}d, "
        f"major crash <= {MAJOR_CRASH_THRESHOLD:.0%} within {MAJOR_CRASH_DAYS:.0f}d"
    )
    print_events(prediction_events)

    out_dir = DATA_FILE_PATH.parent / OUTPUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    indicators.to_csv(out_dir / "lppls_indicators_raw.csv", encoding="utf-8-sig", index=False)
    summary_df.to_csv(out_dir / "lppls_positive_summary.csv", encoding="utf-8-sig")
    prediction_events.to_csv(out_dir / "lppls_prediction_events.csv", encoding="utf-8-sig")
    plot_overview(close, summary_df, prediction_events, out_dir / "01_lppls_overview.png")

    print(f"\nOutput dir: {out_dir}")
    print("Generated: 01_lppls_overview.png, lppls_indicators_raw.csv, lppls_positive_summary.csv, lppls_prediction_events.csv")
    plt.show()


if __name__ == "__main__":
    main()
