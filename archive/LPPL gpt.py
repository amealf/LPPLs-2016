# -*- coding: utf-8 -*-
"""
LPPLS 气泡识别与临界时间 tc 预测（面向 OHLCV 时间序列）

方法要点（与经典文献表述保持一致的核心结构）：
1) 以对数价格 y(t)=ln p(t) 为拟合对象，使用 LPPLS 形式刻画「幂律加速 + 对数周期振荡」：
       y(t) = A + B (tc - t)^m + C (tc - t)^m cos( ω ln(tc - t) + φ )
   其中 tc 为临界时间（常用于表征泡沫破裂的高风险时点），m∈(0,1)，ω 为对数周期角频率。

2) 为提高数值稳定性，采用线性化写法，将模型改写为：
       y(t) = A + B f(t) + C1 g(t) + C2 h(t)
   f(t)=(tc-t)^m
   g(t)=f(t)cos(ω ln(tc-t))
   h(t)=f(t)sin(ω ln(tc-t))
   给定非线性参数 (tc,m,ω) 后，(A,B,C1,C2) 可由线性最小二乘一次求解。

3) 采用「多尺度随机窗口」扫描：
   对每个窗口终点随机抽取多个窗口起点（等价于随机窗口长度），并对每个窗口执行多次随机初值拟合，
   统计通过可解释性约束的拟合比例，构造气泡置信度指标。

4) 事后验证：
   对历史信号，在预测 tc 附近的前瞻区间计算最大回撤，以检验信号是否被后续价格路径支持。

运行结果：
- 控制台输出：历史信号表格、末端是否存在气泡信号、末端 tc 预测区间
- 图像输出（matplotlib）：总览图、验证散点图、末端窗口拟合图
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# =========================
# 路径与关键参数（按需修改）
# =========================
# 直接填写完整文件路径时，优先使用它；设为 None 则回退到 folder_path + file_name
data_file_path = r"F:\Data\XAGUSD\xagusd_30s_all.csv"
folder_path = r"F:\Data\XAGUSD\\"
file_name = "xagusd_30s_all"

# 建议：对 30s 数据先重采样（LPPLS 主要刻画中低频结构）
# 当前默认改为“急杀捕捉版”参数，更贴近 2026-01-30 这类末端崩溃
# 可选：'30min', '1h', '2h', '4h', '1D'
RESAMPLE_RULE = "2h"

# 扫描参数（以「天」为单位，代码内部将换算为点数）
SCAN_STEP_DAYS = 1.0          # 扫描终点步长：1 天
MIN_WINDOW_DAYS = 20.0        # 最短窗口：20 天
MAX_WINDOW_DAYS = 60.0        # 最长窗口：60 天
WINDOWS_PER_END = 10          # 每个终点随机窗口数量

# 拟合参数
FIT_MIN_TC_DAYS = 0.5         # 约束 tc 距离窗口终点的最小天数（避免 tc 贴近窗口端点导致退化）
FIT_MAX_TC_DAYS = 15.0        # 约束 tc 距离窗口终点的最大天数
FIT_N_RANDOM = 24             # 每个窗口随机初值数量（粗搜索）
FIT_N_LOCAL = 4               # 参与局部优化的候选数量

# 过滤约束（文献中常用的经验区间）
BUBBLE_SIGN = "positive"      # 'positive'：正向泡沫；'negative'：负向泡沫；'both'：仅按 m/ω/振荡次数筛选
MIN_OSCILLATIONS = 2.5        # 窗口内最少对数周期振荡圈数
MAX_ABS_C = 1.0               # 振荡项幅度上界（用于抑制非物理解）

# 信号阈值与事后验证
CONF_THRESHOLD = 0.30           # 基础气泡置信度阈值
PREDICTION_CONFIDENCE = 0.50    # 认为“泡沫明显成立”并触发一次预测的更高阈值
PREDICTION_MIN_GAP_DAYS = 8.0   # 两次历史预测至少间隔多少天，避免同一段行情重复触发
EVENT_SELECTION_MODE = "earliest_tc"  # 'first' | 'peak' | 'last' | 'earliest_tc'
VALIDATION_HORIZON_DAYS = 10.0
TC_VALIDATION_BUFFER_DAYS = 2.0
DRAWDOWN_THRESHOLD = -0.12      # 例如 -0.12 表示 12% 的急跌/破裂阈值

# 仅保留“大泡沫 + 大破裂”候选的附加过滤
RUNUP_LOOKBACK_DAYS = 20.0      # 回看多少天衡量这一段是否已经显著加速
MIN_RUNUP_PCT = 0.35            # 从回看窗口低点到触发价，至少上涨 35%
FINAL_PUSH_DAYS = 5.0           # 末端冲顶的观察窗口
MIN_FINAL_PUSH_PCT = 0.10       # 最近 FINAL_PUSH_DAYS 至少再上涨 10%
MAJOR_CRASH_DAYS = 3.0          # 只把短时间内的大跌视为目标破裂
MAJOR_CRASH_THRESHOLD = -0.18   # 例如 3 天内至少下跌 18%

# 随机种子：设为 None 可获得每次运行不同的随机窗口与初值
RANDOM_SEED = 42


# =========================
# 数据读取与预处理
# =========================
def resolve_data_file(data_path: Optional[str], folder: str, name: str) -> Path:
    """
    解析数据文件路径。
    - 若提供完整路径且文件存在，优先直接使用。
    - 否则回退到 folder + name，并兼容常见扩展名的自动探测。
    """
    if data_path:
        fp = Path(data_path)
        if fp.exists():
            return fp
        raise FileNotFoundError(f"未找到数据文件：{fp}")

    p = Path(folder)
    candidates = [name, f"{name}.csv", f"{name}.txt", f"{name}.tsv", f"{name}.dat"]
    for c in candidates:
        fp = p / c
        if fp.exists():
            return fp
    raise FileNotFoundError(f"未找到数据文件：{p} / {name}（已尝试常见扩展名）")


def make_datetime_monotonic(dt: pd.Series, default_step_seconds: int = 30) -> pd.DatetimeIndex:
    """
    将可能存在重复时间戳的序列调整为严格递增，便于后续重采样与建模。

    现象背景：
    - 30s Bar 数据在部分导出格式中可能仅保留到「分钟」，导致同一分钟内出现重复时间戳。
    - 重采样与时间序列操作通常要求索引单调且尽量无重复。

    处理策略：
    - 估计「基础时间间隔」：对去重后排序的时间戳做差，取正差的众数。
    - 对同一时间戳的重复记录，按原始行顺序添加 step = base_delta / 组内重复次数 的时间偏移。
    - 若无法估计基础间隔，采用 default_step_seconds。
    """
    dt = pd.to_datetime(dt, errors="coerce")
    if dt.isna().any():
        raise ValueError("存在无法解析的时间字段，请先清洗数据。")

    uniq = pd.Series(dt.unique()).dropna().sort_values()
    diffs = uniq.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if len(diffs) == 0:
        base_delta = pd.Timedelta(seconds=default_step_seconds)
    else:
        base_delta = diffs.mode().iloc[0] if hasattr(diffs, "mode") else diffs.iloc[0]

    s = pd.Series(dt)
    within_rank = s.groupby(s).cumcount()
    within_size = s.groupby(s).transform("size")

    base_ns = int(base_delta.value)
    step_ns = np.round(base_ns / within_size.to_numpy()).astype(np.int64)
    dt_ns = s.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    adj_ns = dt_ns + within_rank.to_numpy(dtype=np.int64) * step_ns
    adj = pd.to_datetime(adj_ns)

    # 二次安全处理：确保严格递增（极少数情况下可能出现相等或逆序）
    adj_ns2 = adj.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    for i in range(1, len(adj_ns2)):
        if adj_ns2[i] <= adj_ns2[i - 1]:
            adj_ns2[i] = adj_ns2[i - 1] + 1
    return pd.to_datetime(adj_ns2)


def read_ohlcv_file(file_path: Path) -> pd.DataFrame:
    """
    读取无表头 OHLCV 文件，兼容两类常见格式：
    A) 制表符分隔：datetime | open | high | low | close | volume
    B) 空白分隔：date | time | open | high | low | close | volume
    """
    last_error: Optional[Exception] = None
    df_raw: Optional[pd.DataFrame] = None
    sep_candidates = [",", "\t", r"\s+"]

    for sep in sep_candidates:
        try:
            candidate = pd.read_csv(file_path, sep=sep, header=None, engine="python")
        except Exception as exc:
            last_error = exc
            continue
        if candidate.shape[1] >= 6:
            df_raw = candidate
            break

    if df_raw is None:
        detail = f"，最近一次异常：{last_error}" if last_error is not None else ""
        raise ValueError(f"无法识别 OHLCV 文件分隔符或列结构：{file_path}{detail}")

    if df_raw.shape[1] >= 7:
        dt_str = df_raw.iloc[:, 0].astype(str) + " " + df_raw.iloc[:, 1].astype(str)
        data = df_raw.iloc[:, 2:7].copy()
    elif df_raw.shape[1] >= 6:
        dt_str = df_raw.iloc[:, 0].astype(str)
        data = df_raw.iloc[:, 1:6].copy()
    else:
        raise ValueError(f"列数不足，无法解析 OHLCV：{df_raw.shape}")

    data.columns = ["open", "high", "low", "close", "volume"]
    df = data.copy()
    df.insert(0, "datetime", dt_str)

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["datetime", "open", "high", "low", "close"]).copy()
    df["datetime"] = make_datetime_monotonic(df["datetime"], default_step_seconds=30)
    df = df.set_index("datetime").sort_index()
    return df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    将高频 OHLCV 重采样为更低频率序列。
    - open: 窗口内第一笔
    - high: 窗口内最高
    - low : 窗口内最低
    - close: 窗口内最后一笔
    - volume: 窗口内求和
    """
    normalized_rule = str(rule).replace("H", "h").replace("T", "min")

    ohlc = df[["open", "high", "low", "close"]].resample(normalized_rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    })
    vol = df[["volume"]].resample(normalized_rule).sum()
    out = pd.concat([ohlc, vol], axis=1).dropna(subset=["open", "high", "low", "close"])
    return out


# =========================
# LPPLS 核心：线性化拟合
# =========================
def build_design_matrix(t: np.ndarray, tc: float, m: float, omega: float) -> Optional[np.ndarray]:
    dt = tc - t
    if np.any(dt <= 0):
        return None
    log_dt = np.log(dt)
    f = np.power(dt, m)
    g = f * np.cos(omega * log_dt)
    h = f * np.sin(omega * log_dt)
    return np.column_stack([np.ones_like(t), f, g, h])


def lppls_sse_and_beta(t: np.ndarray, y: np.ndarray, tc: float, m: float, omega: float):
    X = build_design_matrix(t, tc, m, omega)
    if X is None:
        return np.inf, None, None
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    sse = float(np.sum((y_hat - y) ** 2))
    return sse, beta, y_hat


def extract_lppls_params(beta: np.ndarray, tc: float, m: float, omega: float) -> Dict[str, float]:
    A, B, C1, C2 = beta
    C = float(np.sqrt(C1 ** 2 + C2 ** 2))
    phi = float(np.arctan2(C2, C1))
    return {
        "A": float(A),
        "B": float(B),
        "C1": float(C1),
        "C2": float(C2),
        "C": C,
        "phi": phi,
        "tc": float(tc),
        "m": float(m),
        "omega": float(omega),
    }


def lppls_filter(params: Dict[str, float], t_start: float, t_end: float,
                 bubble_sign: str,
                 min_oscillations: float,
                 max_abs_C: Optional[float]) -> bool:
    m = params["m"]
    omega = params["omega"]
    tc = params["tc"]
    B = params["B"]
    C = params["C"]

    if not (0.01 < m < 0.99):
        return False
    if not (6.0 <= omega <= 13.0):
        return False
    if not (tc > t_end):
        return False

    # 有效振荡圈数：ω/(2π) * ln((tc-t_start)/(tc-t_end))
    try:
        n_osc = omega / (2 * np.pi) * np.log((tc - t_start) / (tc - t_end))
    except Exception:
        return False
    if (not np.isfinite(n_osc)) or (n_osc < float(min_oscillations)):
        return False

    if bubble_sign == "positive" and B >= 0:
        return False
    if bubble_sign == "negative" and B <= 0:
        return False

    if (max_abs_C is not None) and (abs(C) > float(max_abs_C)):
        return False

    return True


def lppls_sse_only(t: np.ndarray, y: np.ndarray, x: np.ndarray) -> float:
    tc, m, omega = float(x[0]), float(x[1]), float(x[2])
    sse, _, _ = lppls_sse_and_beta(t, y, tc, m, omega)
    if not np.isfinite(sse):
        return 1e100
    return sse


def fit_lppls_window(t: np.ndarray, y: np.ndarray,
                     min_tc_days: float,
                     max_tc_days: float,
                     m_range: Tuple[float, float],
                     omega_range: Tuple[float, float],
                     n_random: int,
                     n_local: int,
                     bubble_sign: str,
                     min_oscillations: float,
                     max_abs_C: Optional[float],
                     seed: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    对单个时间窗执行 LPPLS 拟合并返回最优解。
    """
    rng = np.random.default_rng(seed)

    t_start = float(t[0])
    t_end = float(t[-1])

    if max_tc_days <= min_tc_days:
        raise ValueError("max_tc_days 需要大于 min_tc_days。")

    tc_low = t_end + float(min_tc_days)
    tc_high = t_end + float(max_tc_days)
    m_low, m_high = float(m_range[0]), float(m_range[1])
    w_low, w_high = float(omega_range[0]), float(omega_range[1])

    # 随机粗搜索
    candidates: List[Tuple[float, float, float, float]] = []
    for _ in range(int(n_random)):
        tc0 = rng.uniform(tc_low, tc_high)
        m0 = rng.uniform(m_low, m_high)
        w0 = rng.uniform(w_low, w_high)
        sse0, beta0, _ = lppls_sse_and_beta(t, y, tc0, m0, w0)
        if np.isfinite(sse0) and beta0 is not None:
            candidates.append((sse0, tc0, m0, w0))

    if len(candidates) == 0:
        return None
    candidates.sort(key=lambda x: x[0])

    bounds = [(tc_low, tc_high), (m_low, m_high), (w_low, w_high)]

    best: Optional[Dict[str, Any]] = None
    top = candidates[:max(1, int(n_local))]

    for sse0, tc0, m0, w0 in top:
        x0 = np.array([tc0, m0, w0], dtype=float)

        res = minimize(
            fun=lambda x: lppls_sse_only(t, y, x),
            x0=x0,
            bounds=bounds,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-12},
        )

        tc1, m1, w1 = res.x
        sse1, beta1, y_hat1 = lppls_sse_and_beta(t, y, float(tc1), float(m1), float(w1))
        if (beta1 is None) or (not np.isfinite(sse1)):
            continue

        params = extract_lppls_params(beta1, float(tc1), float(m1), float(w1))
        is_valid = lppls_filter(
            params=params,
            t_start=t_start,
            t_end=t_end,
            bubble_sign=bubble_sign,
            min_oscillations=min_oscillations,
            max_abs_C=max_abs_C,
        )

        sst = float(np.sum((y - np.mean(y)) ** 2))
        r2 = float(1.0 - sse1 / sst) if sst > 0 else np.nan

        fit = {
            "params": params,
            "sse": float(sse1),
            "r2": r2,
            "y_hat": y_hat1,
            "t_start": t_start,
            "t_end": t_end,
            "is_valid": bool(is_valid),
        }
        if (best is None) or (fit["sse"] < best["sse"]):
            best = fit

    return best


# =========================
# 扫描、验证与可视化
# =========================
def forward_drawdown(close: pd.Series, start_time: pd.Timestamp, end_time: pd.Timestamp) -> float:
    seg = close.loc[start_time:end_time]
    if len(seg) < 2:
        return np.nan
    p0 = float(seg.iloc[0])
    pmin = float(seg.min())
    if p0 <= 0:
        return np.nan
    return pmin / p0 - 1.0


def scan_lppls_bubbles(df: pd.DataFrame,
                       price_col: str,
                       scan_step_days: float,
                       min_window_days: float,
                       max_window_days: float,
                       windows_per_end: int,
                       fit_min_tc_days: float,
                       fit_max_tc_days: float,
                       fit_n_random: int,
                       fit_n_local: int,
                       bubble_sign: str,
                       min_oscillations: float,
                       max_abs_C: Optional[float],
                       conf_threshold: float,
                       validation_horizon_days: float,
                       tc_validation_buffer_days: float,
                       drawdown_threshold: float,
                       seed: Optional[int]) -> Tuple[pd.DataFrame, pd.Series]:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index 需要为 DatetimeIndex。")

    close = df[price_col].dropna().copy()
    if len(close) < 200:
        raise ValueError("有效序列过短，难以执行多尺度窗口扫描。")

    deltas = close.index.to_series().diff().dropna()
    dt_seconds = float(deltas.median().total_seconds())
    if dt_seconds <= 0:
        raise ValueError("时间索引间隔异常。")
    points_per_day = 86400.0 / dt_seconds

    step_points = max(1, int(round(scan_step_days * points_per_day)))
    min_win_points = max(50, int(round(min_window_days * points_per_day)))
    max_win_points = max(min_win_points + 1, int(round(max_window_days * points_per_day)))

    t0 = close.index[0]
    t_all = (close.index - t0).total_seconds() / 86400.0
    t_all = t_all.to_numpy(dtype=float)
    y_all = np.log(close.to_numpy(dtype=float))

    rng = np.random.default_rng(seed)

    end_indices = list(range(max_win_points, len(close), step_points))
    if (len(close) - 1) not in end_indices:
        end_indices.append(len(close) - 1)

    last_time = close.index[-1]
    rows: List[Dict[str, Any]] = []

    total_ends = len(end_indices)
    for k, end_idx in enumerate(end_indices):
        end_time = close.index[end_idx]

        valid_tcs: List[float] = []
        valid_ms: List[float] = []
        valid_ws: List[float] = []
        valid_Bs: List[float] = []
        best_fit: Optional[Dict[str, Any]] = None

        n_success = 0

        for j in range(int(windows_per_end)):
            win_size = int(rng.integers(min_win_points, max_win_points + 1))
            start_idx = end_idx - win_size
            if start_idx < 0:
                continue

            t_win = t_all[start_idx:end_idx + 1]
            y_win = y_all[start_idx:end_idx + 1]

            fit_seed = None if seed is None else int(seed + 100000 * k + j)

            fit = fit_lppls_window(
                t=t_win,
                y=y_win,
                min_tc_days=fit_min_tc_days,
                max_tc_days=fit_max_tc_days,
                m_range=(0.1, 0.9),
                omega_range=(6.0, 13.0),
                n_random=fit_n_random,
                n_local=fit_n_local,
                bubble_sign=bubble_sign,
                min_oscillations=min_oscillations,
                max_abs_C=max_abs_C,
                seed=fit_seed,
            )

            if fit is None:
                continue

            n_success += 1
            if (best_fit is None) or (fit["sse"] < best_fit["sse"]):
                best_fit = fit

            if fit["is_valid"]:
                p = fit["params"]
                valid_tcs.append(p["tc"])
                valid_ms.append(p["m"])
                valid_ws.append(p["omega"])
                valid_Bs.append(p["B"])

        n_valid = len(valid_tcs)
        confidence = n_valid / n_success if n_success > 0 else 0.0

        if n_valid > 0:
            tc_med = float(np.median(valid_tcs))
            tc_q10 = float(np.quantile(valid_tcs, 0.10))
            tc_q90 = float(np.quantile(valid_tcs, 0.90))
            tc_dt_med = t0 + pd.to_timedelta(tc_med, unit="D")
            tc_dt_q10 = t0 + pd.to_timedelta(tc_q10, unit="D")
            tc_dt_q90 = t0 + pd.to_timedelta(tc_q90, unit="D")
            m_med = float(np.median(valid_ms))
            w_med = float(np.median(valid_ws))
            B_med = float(np.median(valid_Bs))
        else:
            tc_dt_med, tc_dt_q10, tc_dt_q90 = pd.NaT, pd.NaT, pd.NaT
            m_med, w_med, B_med = np.nan, np.nan, np.nan

        is_bubble = bool(confidence >= conf_threshold)

        dd = np.nan
        status = "样本不足"
        if is_bubble and pd.notna(tc_dt_med):
            verify_end = min(
                tc_dt_med + pd.Timedelta(days=float(tc_validation_buffer_days)),
                end_time + pd.Timedelta(days=float(validation_horizon_days)),
                last_time,
            )
            dd = forward_drawdown(close, end_time, verify_end)

            if end_time + pd.Timedelta(days=float(validation_horizon_days)) <= last_time:
                status = "已验证" if (np.isfinite(dd) and dd <= float(drawdown_threshold)) else "未验证"
            else:
                status = "样本不足"

        rows.append({
            "end_time": end_time,
            "confidence": float(confidence),
            "n_success": int(n_success),
            "n_valid": int(n_valid),
            "tc_median_time": tc_dt_med,
            "tc_q10_time": tc_dt_q10,
            "tc_q90_time": tc_dt_q90,
            "m_median": m_med,
            "omega_median": w_med,
            "B_median": B_med,
            "is_bubble": is_bubble,
            "forward_drawdown": float(dd) if np.isfinite(dd) else np.nan,
            "validation": status,
            "best_fit": best_fit,
        })

        # 进度输出（避免长时间无反馈）
        if (k + 1) % 10 == 0 or (k + 1) == total_ends:
            print(f"扫描进度：{k + 1}/{total_ends}，当前终点：{end_time}，置信度：{confidence:.3f}，有效拟合：{n_valid}/{max(n_success,1)}")

    scan_df = pd.DataFrame(rows).set_index("end_time").sort_index()
    return scan_df, close


def first_drawdown_breach_time(close: pd.Series,
                               start_time: pd.Timestamp,
                               end_time: pd.Timestamp,
                               drawdown_threshold: float) -> pd.Timestamp:
    seg = close.loc[start_time:end_time]
    if len(seg) < 2:
        return pd.NaT

    p0 = float(seg.iloc[0])
    if p0 <= 0:
        return pd.NaT

    dd = seg / p0 - 1.0
    hit = dd[dd <= float(drawdown_threshold)]
    if len(hit) == 0:
        return pd.NaT
    return hit.index[0]


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


def build_prediction_events(scan_df: pd.DataFrame,
                            close: pd.Series,
                            trigger_confidence: float,
                            min_gap_days: float,
                            event_selection_mode: str,
                            validation_horizon_days: float,
                            tc_validation_buffer_days: float,
                            drawdown_threshold: float,
                            runup_lookback_days: float,
                            min_runup_pct: float,
                            final_push_days: float,
                            min_final_push_pct: float,
                            major_crash_days: float,
                            major_crash_threshold: float) -> pd.DataFrame:
    candidates = scan_df[
        (scan_df["confidence"] >= float(trigger_confidence)) &
        scan_df["tc_median_time"].notna()
    ].copy().sort_index()

    if len(candidates) == 0:
        return pd.DataFrame(columns=[
            "event_id", "signal_time", "signal_price", "confidence", "n_valid", "n_success",
            "tc_median_time", "tc_q10_time", "tc_q90_time", "validation", "forward_drawdown",
            "realized_break_time", "verify_end_time", "m_median", "omega_median", "B_median",
            "runup_pct", "final_push_pct", "major_crash_drawdown", "major_break_time",
            "is_large_setup", "is_major_crash", "reason",
        ])

    groups: List[List[Tuple[pd.Timestamp, pd.Series]]] = []
    current_group: List[Tuple[pd.Timestamp, pd.Series]] = []
    prev_time: Optional[pd.Timestamp] = None
    gap = pd.Timedelta(days=float(min_gap_days))

    for signal_time, row in candidates.iterrows():
        if (prev_time is None) or ((signal_time - prev_time) <= gap):
            current_group.append((signal_time, row))
        else:
            groups.append(current_group)
            current_group = [(signal_time, row)]
        prev_time = signal_time

    if current_group:
        groups.append(current_group)

    last_time = close.index[-1]
    rows: List[Dict[str, Any]] = []

    for event_id, group in enumerate(groups, start=1):
        peak_time, peak_row = max(group, key=lambda x: (float(x[1]["confidence"]), x[0].value))
        if event_selection_mode == "first":
            signal_time, signal_row = group[0]
        elif event_selection_mode == "last":
            signal_time, signal_row = group[-1]
        elif event_selection_mode == "peak":
            signal_time, signal_row = peak_time, peak_row
        elif event_selection_mode == "earliest_tc":
            signal_time, signal_row = min(
                group,
                key=lambda x: (
                    x[1]["tc_median_time"].value if pd.notna(x[1]["tc_median_time"]) else np.iinfo(np.int64).max,
                    -x[0].value,
                ),
            )
        else:
            raise ValueError(f"未知的 event_selection_mode: {event_selection_mode}")

        verify_end_time = min(
            signal_row["tc_median_time"] + pd.Timedelta(days=float(tc_validation_buffer_days)),
            signal_time + pd.Timedelta(days=float(validation_horizon_days)),
            last_time,
        )
        signal_price = float(close.loc[signal_time])
        runup_pct = trailing_runup_from_low(close, signal_time, runup_lookback_days)
        final_push_pct = trailing_window_return(close, signal_time, final_push_days)
        is_large_setup = bool(
            np.isfinite(runup_pct) and np.isfinite(final_push_pct) and
            (runup_pct >= float(min_runup_pct)) and
            (final_push_pct >= float(min_final_push_pct))
        )

        major_crash_end = min(signal_time + pd.Timedelta(days=float(major_crash_days)), last_time)
        major_crash_drawdown = forward_drawdown(close, signal_time, major_crash_end)
        major_break_time = first_drawdown_breach_time(
            close=close,
            start_time=signal_time,
            end_time=major_crash_end,
            drawdown_threshold=major_crash_threshold,
        )
        has_major_crash_window = (signal_time + pd.Timedelta(days=float(major_crash_days))) <= last_time
        is_major_crash = bool(np.isfinite(major_crash_drawdown) and (major_crash_drawdown <= float(major_crash_threshold)))

        realized_break_time = first_drawdown_breach_time(
            close=close,
            start_time=signal_time,
            end_time=verify_end_time,
            drawdown_threshold=drawdown_threshold,
        )

        # Keep only large blow-off setups. For historical events with enough future sample,
        # also require the subsequent move to look like a major crash.
        if not is_large_setup:
            continue
        if has_major_crash_window and (not is_major_crash):
            continue

        reason = (
            f"confidence={signal_row['confidence']:.3f} "
            f"({int(signal_row['n_valid'])}/{max(int(signal_row['n_success']), 1)}), "
            f"tc={signal_row['tc_median_time']} "
            f"[{signal_row['tc_q10_time']} ~ {signal_row['tc_q90_time']}], "
            f"m={signal_row['m_median']:.3f}, "
            f"omega={signal_row['omega_median']:.3f}, "
            f"B={signal_row['B_median']:.6f}, "
            f"runup={runup_pct:.2%}, "
            f"final_push={final_push_pct:.2%}, "
            f"major_crash={major_crash_drawdown:.2%}, "
            f"cluster_peak={peak_time} ({peak_row['confidence']:.3f}), "
            f"selection_mode={event_selection_mode}"
        )

        rows.append({
            "event_id": int(event_id),
            "signal_time": signal_time,
            "signal_price": signal_price,
            "confidence": float(signal_row["confidence"]),
            "n_valid": int(signal_row["n_valid"]),
            "n_success": int(signal_row["n_success"]),
            "tc_median_time": signal_row["tc_median_time"],
            "tc_q10_time": signal_row["tc_q10_time"],
            "tc_q90_time": signal_row["tc_q90_time"],
            "validation": signal_row["validation"],
            "forward_drawdown": float(signal_row["forward_drawdown"]) if np.isfinite(signal_row["forward_drawdown"]) else np.nan,
            "realized_break_time": realized_break_time,
            "verify_end_time": verify_end_time,
            "m_median": float(signal_row["m_median"]) if np.isfinite(signal_row["m_median"]) else np.nan,
            "omega_median": float(signal_row["omega_median"]) if np.isfinite(signal_row["omega_median"]) else np.nan,
            "B_median": float(signal_row["B_median"]) if np.isfinite(signal_row["B_median"]) else np.nan,
            "peak_confidence": float(peak_row["confidence"]),
            "peak_time": peak_time,
            "selection_mode": event_selection_mode,
            "reason": reason,
            "runup_pct": float(runup_pct) if np.isfinite(runup_pct) else np.nan,
            "final_push_pct": float(final_push_pct) if np.isfinite(final_push_pct) else np.nan,
            "major_crash_drawdown": float(major_crash_drawdown) if np.isfinite(major_crash_drawdown) else np.nan,
            "major_break_time": major_break_time,
            "is_large_setup": is_large_setup,
            "is_major_crash": is_major_crash,
        })

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            "event_id", "signal_time", "signal_price", "confidence", "n_valid", "n_success",
            "tc_median_time", "tc_q10_time", "tc_q90_time", "validation", "forward_drawdown",
            "realized_break_time", "verify_end_time", "m_median", "omega_median", "B_median",
            "runup_pct", "final_push_pct", "major_crash_drawdown", "major_break_time",
            "is_large_setup", "is_major_crash", "selection_mode", "reason",
        ]).set_index("signal_time")
    return pd.DataFrame(rows).set_index("signal_time").sort_index()


def print_prediction_events(prediction_events: pd.DataFrame):
    if len(prediction_events) == 0:
        print("\n未找到达到历史预测触发阈值的泡沫事件。")
        return

    print("\n历史破裂预测事件：")
    for signal_time, row in prediction_events.iterrows():
        print(
            f"\n[事件 {int(row['event_id'])}] 触发时点：{signal_time} | "
            f"触发价：{row['signal_price']:.4f} | "
            f"置信度：{row['confidence']:.3f} "
            f"({int(row['n_valid'])}/{max(int(row['n_success']), 1)})"
        )
        print(f"  事件提炼方式：{row['selection_mode']}")
        print(f"  预测破裂时间 tc：{row['tc_median_time']}")
        print(f"  预测区间 q10~q90：{row['tc_q10_time']} ~ {row['tc_q90_time']}")
        print(
            f"  参数中位数：m={row['m_median']:.3f}, "
            f"omega={row['omega_median']:.3f}, B={row['B_median']:.6f}"
        )
        print(
            f"  大级别过滤：runup={row['runup_pct']:.2%}, "
            f"final_push={row['final_push_pct']:.2%}, "
            f"major_crash={row['major_crash_drawdown']:.2%}"
        )
        print(f"  聚类内峰值信号：{row['peak_time']} | 峰值置信度：{row['peak_confidence']:.3f}")
        print(f"  触发理由：{row['reason']}")
        print(f"  验证截止：{row['verify_end_time']}")
        if pd.notna(row["major_break_time"]):
            print(f"  大破裂首次触发时间：{row['major_break_time']}")
        else:
            print("  大破裂首次触发时间：未出现")
        if pd.notna(row["realized_break_time"]):
            print(f"  实际首次触发回撤阈值的时间：{row['realized_break_time']}")
        else:
            print("  实际首次触发回撤阈值的时间：未出现")
        if np.isfinite(row["forward_drawdown"]):
            print(f"  事后区间最深回撤：{row['forward_drawdown']:.2%}")
        else:
            print("  事后区间最深回撤：样本不足")
        print(f"  验证结论：{row['validation']}")


def plot_bubble_overview(close: pd.Series,
                         scan_df: pd.DataFrame,
                         conf_threshold: float,
                         prediction_events: Optional[pd.DataFrame] = None,
                         out_path: Optional[Path] = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3, 1])

    ax1.plot(close.index, close.values, linewidth=1.0)
    ax1.set_title("价格与气泡信号（LPPLS 扫描）")
    ax1.set_ylabel("Price")

    bubble = scan_df[scan_df["confidence"] >= conf_threshold].copy()
    if len(bubble) > 0:
        bubble_price = close.reindex(bubble.index)

        m_valid = bubble["validation"] == "已验证"
        m_invalid = bubble["validation"] == "未验证"
        m_na = bubble["validation"] == "样本不足"

        ax1.scatter(bubble.index[m_valid], bubble_price[m_valid], marker="o", s=30, label="气泡信号：已验证")
        ax1.scatter(bubble.index[m_invalid], bubble_price[m_invalid], marker="x", s=40, label="气泡信号：未验证")
        ax1.scatter(bubble.index[m_na], bubble_price[m_na], marker="^", s=35, label="气泡信号：样本不足")

        # 为避免图面过载，仅对置信度最高的若干条信号标注 tc
        top = bubble.dropna(subset=["tc_median_time"]).sort_values("confidence", ascending=False).head(5)
        for t_end, row in top.iterrows():
            tc = row["tc_median_time"]
            ax1.axvline(tc, linestyle=":", linewidth=1.0)
            ax1.annotate("tc", xy=(tc, ax1.get_ylim()[1]), xytext=(2, -2),
                         textcoords="offset points", rotation=90,
                         va="top", ha="left", fontsize=8)

    ax1.legend(loc="best")

    ax2.plot(scan_df.index, scan_df["confidence"].values, linewidth=1.0)
    ax2.axhline(conf_threshold, linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Time")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_validation_scatter(scan_df: pd.DataFrame, out_path: Optional[Path] = None):
    df = scan_df.copy()
    df = df[df["is_bubble"]].copy()
    df = df[np.isfinite(df["forward_drawdown"])]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(df["confidence"], df["forward_drawdown"], s=30)
    ax.set_title("置信度与前瞻回撤关系")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Forward drawdown (min/entry - 1)")
    ax.axhline(0, linewidth=1.0)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_prediction_overview(close: pd.Series,
                             scan_df: pd.DataFrame,
                             prediction_events: pd.DataFrame,
                             conf_threshold: float,
                             prediction_threshold: float,
                             out_path: Optional[Path] = None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True, height_ratios=[3, 1])

    ax1.plot(close.index, close.values, linewidth=1.0, color="tab:blue")
    ax1.set_title("价格、泡沫信号与历史破裂预测（LPPLS 扫描）")
    ax1.set_ylabel("Price")

    bubble = scan_df[scan_df["confidence"] >= conf_threshold].copy()
    if len(bubble) > 0:
        bubble_price = close.reindex(bubble.index)
        m_valid = bubble["validation"] == "已验证"
        m_invalid = bubble["validation"] == "未验证"
        m_na = bubble["validation"] == "样本不足"

        ax1.scatter(bubble.index[m_valid], bubble_price[m_valid], marker="o", s=28, label="气泡信号：已验证")
        ax1.scatter(bubble.index[m_invalid], bubble_price[m_invalid], marker="x", s=36, label="气泡信号：未验证")
        ax1.scatter(bubble.index[m_na], bubble_price[m_na], marker="^", s=34, label="气泡信号：样本不足")

    ax2.plot(scan_df.index, scan_df["confidence"].values, linewidth=1.0, color="tab:blue", label="Confidence")
    ax2.axhline(conf_threshold, linestyle="--", linewidth=1.0, color="tab:blue",
                label=f"基础阈值 {conf_threshold:.2f}")
    ax2.axhline(prediction_threshold, linestyle="--", linewidth=1.0, color="tab:red",
                label=f"预测触发阈值 {prediction_threshold:.2f}")
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel("Time")

    if len(prediction_events) > 0:
        colors = plt.cm.get_cmap("tab10", max(len(prediction_events), 1))
        y_top = float(np.nanmax(close.values))
        y_bottom = float(np.nanmin(close.values))
        y_pad = (y_top - y_bottom) * 0.04 if y_top > y_bottom else 1.0

        for idx, (signal_time, row) in enumerate(prediction_events.iterrows(), start=1):
            color = colors(idx - 1)
            tc_time = row["tc_median_time"]
            if pd.isna(tc_time):
                continue

            ax1.axvline(tc_time, color=color, linestyle="--", linewidth=1.8,
                        label=f"预测破裂 {int(row['event_id'])}")
            if pd.notna(row["tc_q10_time"]) and pd.notna(row["tc_q90_time"]):
                ax1.axvspan(row["tc_q10_time"], row["tc_q90_time"], color=color, alpha=0.08)

            ax1.scatter(signal_time, row["signal_price"], color=color, marker="s", s=58,
                        edgecolors="black", linewidths=0.5, zorder=5)
            ax1.annotate(
                f"P{int(row['event_id'])}",
                xy=(tc_time, y_top + y_pad),
                xytext=(2, -2),
                textcoords="offset points",
                rotation=90,
                va="top",
                ha="left",
                fontsize=8,
                color=color,
            )

            ax2.scatter(signal_time, row["confidence"], color=color, s=46, zorder=4)
            ax2.axvline(signal_time, color=color, linestyle=":", linewidth=0.8, alpha=0.45)

    ax1.legend(loc="best")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=160)
    return fig


def plot_last_window_fit(close: pd.Series, scan_df: pd.DataFrame, out_path: Optional[Path] = None):
    if len(scan_df) == 0:
        return None
    last_row = scan_df.iloc[-1]
    best_fit = last_row.get("best_fit", None)
    if best_fit is None:
        return None

    t0 = close.index[0]
    w_start = t0 + pd.to_timedelta(best_fit["t_start"], unit="D")
    w_end = t0 + pd.to_timedelta(best_fit["t_end"], unit="D")
    seg = close.loc[w_start:w_end]
    if len(seg) < 10:
        return None

    t_seg = (seg.index - t0).total_seconds() / 86400.0
    t_seg = t_seg.to_numpy(dtype=float)
    y_seg = np.log(seg.to_numpy(dtype=float))

    p = best_fit["params"]
    tc, m, omega = p["tc"], p["m"], p["omega"]
    _, beta, _ = lppls_sse_and_beta(t_seg, y_seg, float(tc), float(m), float(omega))
    if beta is None:
        return None

    t_ext = np.linspace(float(t_seg[0]), float(tc), 400)
    X_ext = build_design_matrix(t_ext, float(tc), float(m), float(omega))
    if X_ext is None:
        return None
    y_ext = X_ext @ beta
    price_ext = np.exp(y_ext)
    time_ext = t0 + pd.to_timedelta(t_ext, unit="D")
    tc_time = t0 + pd.to_timedelta(float(tc), unit="D")

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(seg.index, seg.values, linewidth=1.0, label="Observed")
    ax.plot(time_ext, price_ext, linewidth=1.2, label="LPPLS fit / extrapolation")
    ax.axvline(tc_time, linestyle="--", linewidth=1.0, label="Predicted tc")

    ax.set_title("末端窗口 LPPLS 拟合与临界时间预测")
    ax.set_ylabel("Price")
    ax.set_xlabel("Time")
    ax.legend(loc="best")

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=160)
    return fig


def main():
    # 中文字体设置（Windows 通常可用 SimHei 或 Microsoft YaHei）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    file_path = resolve_data_file(data_file_path, folder_path, file_name)
    print(f"数据文件：{file_path}")

    df_raw = read_ohlcv_file(file_path)
    print(f"原始数据行数：{len(df_raw):,d}，时间范围：{df_raw.index.min()} ~ {df_raw.index.max()}")

    df = resample_ohlcv(df_raw, RESAMPLE_RULE)
    print(f"重采样规则：{RESAMPLE_RULE}，行数：{len(df):,d}，时间范围：{df.index.min()} ~ {df.index.max()}")

    data_span_days = max((df.index.max() - df.index.min()).total_seconds() / 86400.0, 1.0)
    effective_max_window_days = min(
        MAX_WINDOW_DAYS,
        max(MIN_WINDOW_DAYS + SCAN_STEP_DAYS, data_span_days * 0.8),
    )
    print(f"扫描窗口：{MIN_WINDOW_DAYS:.1f} ~ {effective_max_window_days:.1f} 天")

    scan_df, close = scan_lppls_bubbles(
        df=df,
        price_col="close",
        scan_step_days=SCAN_STEP_DAYS,
        min_window_days=MIN_WINDOW_DAYS,
        max_window_days=effective_max_window_days,
        windows_per_end=WINDOWS_PER_END,
        fit_min_tc_days=FIT_MIN_TC_DAYS,
        fit_max_tc_days=FIT_MAX_TC_DAYS,
        fit_n_random=FIT_N_RANDOM,
        fit_n_local=FIT_N_LOCAL,
        bubble_sign=BUBBLE_SIGN,
        min_oscillations=MIN_OSCILLATIONS,
        max_abs_C=MAX_ABS_C,
        conf_threshold=CONF_THRESHOLD,
        validation_horizon_days=VALIDATION_HORIZON_DAYS,
        tc_validation_buffer_days=TC_VALIDATION_BUFFER_DAYS,
        drawdown_threshold=DRAWDOWN_THRESHOLD,
        seed=RANDOM_SEED,
    )

    out_dir = file_path.parent / "lppls_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction_events = build_prediction_events(
        scan_df=scan_df,
        close=close,
        trigger_confidence=PREDICTION_CONFIDENCE,
        min_gap_days=PREDICTION_MIN_GAP_DAYS,
        event_selection_mode=EVENT_SELECTION_MODE,
        validation_horizon_days=VALIDATION_HORIZON_DAYS,
        tc_validation_buffer_days=TC_VALIDATION_BUFFER_DAYS,
        drawdown_threshold=DRAWDOWN_THRESHOLD,
        runup_lookback_days=RUNUP_LOOKBACK_DAYS,
        min_runup_pct=MIN_RUNUP_PCT,
        final_push_days=FINAL_PUSH_DAYS,
        min_final_push_pct=MIN_FINAL_PUSH_PCT,
        major_crash_days=MAJOR_CRASH_DAYS,
        major_crash_threshold=MAJOR_CRASH_THRESHOLD,
    )

    print(
        f"历史预测触发阈值：{PREDICTION_CONFIDENCE:.2f}，"
        f"最小事件间隔：{PREDICTION_MIN_GAP_DAYS:.1f} 天，"
        f"事件提炼方式：{EVENT_SELECTION_MODE}，"
        f"大泡沫过滤：{RUNUP_LOOKBACK_DAYS:.0f} 天涨幅 >= {MIN_RUNUP_PCT:.0%}，"
        f"末端 {FINAL_PUSH_DAYS:.0f} 天再涨 >= {MIN_FINAL_PUSH_PCT:.0%}，"
        f"{MAJOR_CRASH_DAYS:.0f} 天急跌 <= {MAJOR_CRASH_THRESHOLD:.0%}，"
        f"提炼得到 {len(prediction_events)} 条独立预测事件"
    )
    print_prediction_events(prediction_events)

    # 历史信号表格
    bubbles = scan_df[scan_df["is_bubble"]].copy().sort_values("confidence", ascending=False)
    show_cols = ["confidence", "tc_median_time", "tc_q10_time", "tc_q90_time",
                 "validation", "forward_drawdown", "m_median", "omega_median", "B_median"]

    if len(bubbles) > 0:
        print("\n历史气泡信号（按置信度降序，截取前 30 条）：")
        print(bubbles[show_cols].head(30).to_string())
        bubbles[show_cols].head(200).to_csv(out_dir / "bubble_signals_top.csv", encoding="utf-8-sig")
    else:
        print("\n历史区间内未获得满足阈值的气泡信号。")

    if len(prediction_events) > 0:
        event_cols = [
            "event_id", "signal_price", "confidence", "n_valid", "n_success",
            "tc_median_time", "tc_q10_time", "tc_q90_time", "validation",
            "forward_drawdown", "realized_break_time", "verify_end_time",
            "m_median", "omega_median", "B_median", "peak_time", "peak_confidence",
            "selection_mode", "runup_pct", "final_push_pct", "major_crash_drawdown",
            "major_break_time", "is_large_setup", "is_major_crash", "reason",
        ]
        prediction_events[event_cols].to_csv(out_dir / "prediction_events.csv", encoding="utf-8-sig")

    # 末端预测结论
    last = scan_df.iloc[-1]
    print("\n末端时刻预测：")
    print(f"- 扫描终点：{scan_df.index[-1]}")
    print(f"- 置信度：{last['confidence']:.3f}（阈值 {CONF_THRESHOLD:.2f}）")
    if (last["confidence"] >= CONF_THRESHOLD) and pd.notna(last["tc_median_time"]):
        print(f"- 预测临界时间 tc（中位数）：{last['tc_median_time']}")
        print(f"- tc 区间（q10~q90）：{last['tc_q10_time']} ~ {last['tc_q90_time']}")
        print(f"- 参数中位数：m={last['m_median']:.3f}, ω={last['omega_median']:.3f}, B={last['B_median']:.6f}")
    else:
        print("- 末端未形成稳定的气泡信号（基于当前阈值与筛选条件）。")

    # 图像输出（多图）
    plot_prediction_overview(
        close,
        scan_df,
        prediction_events,
        CONF_THRESHOLD,
        PREDICTION_CONFIDENCE,
        out_path=out_dir / "01_overview.png",
    )
    plot_validation_scatter(scan_df, out_path=out_dir / "02_validation_scatter.png")
    plot_last_window_fit(close, scan_df, out_path=out_dir / "03_last_window_fit.png")

    # 保存扫描结果
    scan_df.to_csv(out_dir / "scan_results.csv", encoding="utf-8-sig")

    print(f"\n输出目录：{out_dir}")
    print("已生成：01_overview.png, 02_validation_scatter.png, 03_last_window_fit.png, scan_results.csv, prediction_events.csv")

    plt.show()


if __name__ == "__main__":
    main()
