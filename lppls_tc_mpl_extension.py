from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math
import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import dates as mdates
from matplotlib import colors as mcolors
from scipy.optimize import minimize
from scipy.signal import find_peaks

from lppls.lppls import LPPLS


@dataclass
class FitConstraints:
    m_min: float = 0.1
    m_max: float = 0.9
    w_min: float = 6.0
    w_max: float = 13.0
    d_min: float = 0.8
    b_sign: str = "negative"


class LPPLSModifiedTC(LPPLS):
    """LPPLS extension for modified-profile-likelihood inference on t_c.

    The class adds a fixed-t_c calibration, an approximate modified profile
    likelihood for t_c, likelihood intervals, multi-scale t_c surfaces, scenario
    clustering, and Matplotlib hover annotations.

    Parameters
    ----------
    observations:
        A 2 x n array with time on the first row and log-price on the second row.
        The time axis may be absolute ordinals or relative day offsets.
    time_origin:
        Needed only when observations[0] contains relative day offsets. The model
        will use it to convert t, t2 and t_c into timestamps for plotting and CSV
        exports.
    """

    def __init__(self, observations: np.ndarray | pd.DataFrame, time_origin: Any | None = None) -> None:
        super().__init__(observations)
        self.time_origin = None if time_origin is None else pd.Timestamp(time_origin)

    # ------------------------------------------------------------------
    # time conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_array(observations: np.ndarray | pd.DataFrame) -> np.ndarray:
        if isinstance(observations, pd.DataFrame):
            return observations.to_numpy().T
        return observations

    @staticmethod
    def _looks_like_ordinal(value: float) -> bool:
        return 5.0e5 <= float(value) <= 1.0e6

    def value_to_timestamp(self, value: float) -> pd.Timestamp:
        value = float(value)
        if self._looks_like_ordinal(value):
            day = int(math.floor(value))
            frac = value - day
            base = pd.Timestamp.fromordinal(day)
            if frac == 0:
                return base
            return base + pd.to_timedelta(frac, unit="D")
        if self.time_origin is None:
            raise ValueError(
                "Relative-day observations require time_origin so the model can convert t and t_c to timestamps."
            )
        return self.time_origin + pd.to_timedelta(value, unit="D")

    def values_to_timestamps(self, values: Iterable[float]) -> pd.DatetimeIndex:
        return pd.DatetimeIndex([self.value_to_timestamp(v) for v in values])

    # ------------------------------------------------------------------
    # LPPLS derivatives and fixed-tc calibration
    # ------------------------------------------------------------------
    @staticmethod
    def _dt(tc: float, t: np.ndarray) -> np.ndarray:
        return np.abs(tc - t) + 1e-8

    @staticmethod
    def _phase(dt: np.ndarray, w: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        log_dt = np.log(dt)
        wt = w * log_dt
        return log_dt, np.cos(wt), np.sin(wt)

    def _lppls_and_terms(
        self,
        t: np.ndarray,
        tc: float,
        m: float,
        w: float,
        a: float,
        b: float,
        c1: float,
        c2: float,
    ) -> dict[str, np.ndarray]:
        dt = self._dt(tc, t)
        log_dt, cos_w, sin_w = self._phase(dt, w)
        dt_m = np.power(dt, m)
        f = dt_m
        g = dt_m * cos_w
        h = dt_m * sin_w
        y_hat = a + b * f + c1 * g + c2 * h
        return {
            "dt": dt,
            "log_dt": log_dt,
            "cos_w": cos_w,
            "sin_w": sin_w,
            "f": f,
            "g": g,
            "h": h,
            "y_hat": y_hat,
        }

    def _first_derivatives(self, t: np.ndarray, tc: float, params: dict[str, float]) -> np.ndarray:
        terms = self._lppls_and_terms(
            t,
            tc,
            params["m"],
            params["w"],
            params["a"],
            params["b"],
            params["c1"],
            params["c2"],
        )
        log_dt = terms["log_dt"]
        f = terms["f"]
        g = terms["g"]
        h = terms["h"]
        cos_w = terms["cos_w"]
        sin_w = terms["sin_w"]
        b = params["b"]
        c1 = params["c1"]
        c2 = params["c2"]

        d_m = log_dt * (b * f + c1 * g + c2 * h)
        d_w = f * log_dt * (-c1 * sin_w + c2 * cos_w)
        d_a = np.ones_like(f)
        d_b = f
        d_c1 = g
        d_c2 = h
        return np.column_stack([d_m, d_w, d_a, d_b, d_c1, d_c2])

    def _second_derivative_stack(self, t: np.ndarray, tc: float, params: dict[str, float]) -> np.ndarray:
        terms = self._lppls_and_terms(
            t,
            tc,
            params["m"],
            params["w"],
            params["a"],
            params["b"],
            params["c1"],
            params["c2"],
        )
        log_dt = terms["log_dt"]
        f = terms["f"]
        g = terms["g"]
        h = terms["h"]
        cos_w = terms["cos_w"]
        sin_w = terms["sin_w"]
        b = params["b"]
        c1 = params["c1"]
        c2 = params["c2"]

        n = len(t)
        stack = np.zeros((n, 6, 6), dtype=float)

        d_mm = (log_dt ** 2) * (b * f + c1 * g + c2 * h)
        d_mw = f * (log_dt ** 2) * (-c1 * sin_w + c2 * cos_w)
        d_mB = f * log_dt
        d_mC1 = g * log_dt
        d_mC2 = h * log_dt
        d_ww = -f * (log_dt ** 2) * (c1 * cos_w + c2 * sin_w)
        d_wC1 = -f * log_dt * sin_w
        d_wC2 = f * log_dt * cos_w

        stack[:, 0, 0] = d_mm
        stack[:, 0, 1] = d_mw
        stack[:, 1, 0] = d_mw
        stack[:, 0, 3] = d_mB
        stack[:, 3, 0] = d_mB
        stack[:, 0, 4] = d_mC1
        stack[:, 4, 0] = d_mC1
        stack[:, 0, 5] = d_mC2
        stack[:, 5, 0] = d_mC2
        stack[:, 1, 1] = d_ww
        stack[:, 1, 4] = d_wC1
        stack[:, 4, 1] = d_wC1
        stack[:, 1, 5] = d_wC2
        stack[:, 5, 1] = d_wC2
        return stack

    @staticmethod
    def _safe_slogdet(mat: np.ndarray, ridge: float = 1e-8) -> float:
        m = np.array(mat, dtype=float, copy=True)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise ValueError("Expected square matrix for determinant")
        m = m + ridge * np.eye(m.shape[0])
        sign, logabsdet = np.linalg.slogdet(m)
        if not np.isfinite(logabsdet) or sign == 0:
            m = m + 1e-5 * np.eye(m.shape[0])
            sign, logabsdet = np.linalg.slogdet(m)
        if not np.isfinite(logabsdet):
            return float("-inf")
        return float(logabsdet)

    def _sse_for_fixed_tc(self, x: np.ndarray, tc: float, obs: np.ndarray) -> float:
        m, w = float(x[0]), float(x[1])
        rM = self.matrix_equation(obs, tc, m, w)
        a, b, c1, c2 = rM[:, 0].tolist()
        y_hat = self.lppls(obs[0, :], tc, m, w, a, b, c1, c2)
        resid = y_hat - obs[1, :]
        return float(np.dot(resid, resid))

    def fit_fixed_tc(
        self,
        tc: float,
        obs: np.ndarray | None = None,
        max_searches: int = 25,
        minimizer: str = "L-BFGS-B",
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.1, 0.9), (6.0, 13.0)),
        warm_start: tuple[float, float] | None = None,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        obs = self._as_array(self.observations if obs is None else obs)
        rng = random.Random(random_state)
        best: dict[str, Any] | None = None

        seeds: list[np.ndarray] = []
        if warm_start is not None:
            seeds.append(np.array(warm_start, dtype=float))

        for _ in range(max_searches):
            m0 = rng.uniform(bounds[0][0], bounds[0][1])
            w0 = rng.uniform(bounds[1][0], bounds[1][1])
            seeds.append(np.array([m0, w0], dtype=float))

        for seed in seeds:
            try:
                kwargs: dict[str, Any] = {}
                if minimizer.upper() in {"L-BFGS-B", "TNC", "SLSQP", "POWELL", "TRUST-CONSTR"}:
                    kwargs["bounds"] = bounds
                opt = minimize(
                    fun=lambda x, fixed_tc=tc, fixed_obs=obs: self._sse_for_fixed_tc(x, fixed_tc, fixed_obs),
                    x0=seed,
                    method=minimizer,
                    **kwargs,
                )
                if not opt.success and best is not None:
                    continue
                m = float(opt.x[0])
                w = float(opt.x[1])
                rM = self.matrix_equation(obs, tc, m, w)
                a, b, c1, c2 = rM[:, 0].tolist()
                c = float(self.get_c(c1, c2))
                sse = self._sse_for_fixed_tc(np.array([m, w]), tc, obs)
                t1, t2 = float(obs[0, 0]), float(obs[0, -1])
                fit = {
                    "tc": float(tc),
                    "m": m,
                    "w": w,
                    "a": float(a),
                    "b": float(b),
                    "c": c,
                    "c1": float(c1),
                    "c2": float(c2),
                    "O": float(self.get_oscillations(w, tc, t1, t2)),
                    "D": float(self.get_damping(m, w, b, c)) if c != 0 else float("nan"),
                    "sse": float(sse),
                    "n": int(obs.shape[1]),
                    "t1": t1,
                    "t2": t2,
                    "success": bool(opt.success),
                }
                if best is None or fit["sse"] < best["sse"]:
                    best = fit
            except Exception:
                continue

        if best is None:
            t1, t2 = float(obs[0, 0]), float(obs[0, -1])
            return {
                "tc": float(tc),
                "m": float("nan"),
                "w": float("nan"),
                "a": float("nan"),
                "b": float("nan"),
                "c": float("nan"),
                "c1": float("nan"),
                "c2": float("nan"),
                "O": float("nan"),
                "D": float("nan"),
                "sse": float("inf"),
                "n": int(obs.shape[1]),
                "t1": t1,
                "t2": t2,
                "success": False,
            }
        return best

    # ------------------------------------------------------------------
    # covariance, modified profile likelihood, intervals
    # ------------------------------------------------------------------
    def _covariance_for_fit(self, obs: np.ndarray, fit: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = obs[0, :]
        y = obs[1, :]
        params = {k: fit[k] for k in ["m", "w", "a", "b", "c1", "c2"]}
        X = self._first_derivatives(t, fit["tc"], params)
        y_hat = self.lppls(t, fit["tc"], fit["m"], fit["w"], fit["a"], fit["b"], fit["c1"], fit["c2"])
        resid = y - y_hat
        second_stack = self._second_derivative_stack(t, fit["tc"], params)
        H = np.einsum("i,ijk->jk", resid, second_stack)
        J = X.T @ X - H
        s = fit["sse"] / max(fit["n"], 1)
        fisher = J / max(s, 1e-12)
        cov = np.linalg.pinv(fisher)
        return X, resid, J, cov

    def _qualification_from_intervals(
        self,
        fit: dict[str, Any],
        cov_psi: np.ndarray,
        cutoff: float,
        constraints: FitConstraints,
    ) -> dict[str, Any]:
        if not np.isfinite(cov_psi).all():
            cov_psi = np.nan_to_num(cov_psi, nan=0.0, posinf=0.0, neginf=0.0)

        q = -2.0 * math.log(max(cutoff, 1e-12))
        delta_m = math.sqrt(max(q * cov_psi[0, 0], 0.0))
        delta_w = math.sqrt(max(q * cov_psi[1, 1], 0.0))

        d_val = float(fit["D"]) if np.isfinite(fit["D"]) else float("nan")
        cabs = max(abs(fit["c"]), 1e-12)
        abs_b = abs(fit["b"]) if np.isfinite(fit["b"]) else np.nan
        sign_b = 1.0 if fit["b"] >= 0 else -1.0
        grad = np.array(
            [
                abs_b / (fit["w"] * cabs) if np.isfinite(abs_b) and fit["w"] != 0 else 0.0,
                -fit["m"] * abs_b / ((fit["w"] ** 2) * cabs) if np.isfinite(abs_b) and fit["w"] != 0 else 0.0,
                0.0,
                fit["m"] * sign_b / (fit["w"] * cabs) if fit["w"] != 0 else 0.0,
                -fit["m"] * abs_b * fit["c1"] / (fit["w"] * (cabs ** 3)) if fit["w"] != 0 else 0.0,
                -fit["m"] * abs_b * fit["c2"] / (fit["w"] * (cabs ** 3)) if fit["w"] != 0 else 0.0,
            ],
            dtype=float,
        )
        var_d = float(grad @ cov_psi @ grad.T)
        delta_d = math.sqrt(max(q * var_d, 0.0))

        m_lo, m_hi = fit["m"] - delta_m, fit["m"] + delta_m
        w_lo, w_hi = fit["w"] - delta_w, fit["w"] + delta_w
        d_lo, d_hi = d_val - delta_d, d_val + delta_d

        strict_b = fit["b"] < 0 if constraints.b_sign == "negative" else fit["b"] > 0
        strict = (
            strict_b
            and constraints.m_min <= fit["m"] <= constraints.m_max
            and constraints.w_min <= fit["w"] <= constraints.w_max
            and d_val >= constraints.d_min
        )
        conf = (
            strict_b
            and (m_hi >= constraints.m_min and m_lo <= constraints.m_max)
            and (w_hi >= constraints.w_min and w_lo <= constraints.w_max)
            and (d_hi >= constraints.d_min)
        )
        return {
            "m_lo": m_lo,
            "m_hi": m_hi,
            "w_lo": w_lo,
            "w_hi": w_hi,
            "D_lo": d_lo,
            "D_hi": d_hi,
            "qualified_strict": bool(strict),
            "qualified_conf": bool(conf),
        }

    def scan_tc_for_window(
        self,
        obs: np.ndarray,
        tc_grid: Iterable[float],
        max_searches: int = 20,
        minimizer: str = "L-BFGS-B",
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.1, 0.9), (6.0, 13.0)),
        cutoff: float = 0.05,
        constraints: FitConstraints | None = None,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        constraints = constraints or FitConstraints()
        obs = self._as_array(obs)
        tc_values = [float(x) for x in tc_grid]
        rows: list[dict[str, Any]] = []
        warm: tuple[float, float] | None = None

        for idx, tc in enumerate(tc_values):
            fit = self.fit_fixed_tc(
                tc=tc,
                obs=obs,
                max_searches=max_searches,
                minimizer=minimizer,
                bounds=bounds,
                warm_start=warm,
                random_state=None if random_state is None else random_state + idx,
            )
            if np.isfinite(fit["m"]) and np.isfinite(fit["w"]):
                warm = (fit["m"], fit["w"])
            rows.append(fit)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        best_idx = df["sse"].replace([np.inf, -np.inf], np.nan).idxmin()
        full_mle = df.loc[best_idx].to_dict()
        params_hat = {k: full_mle[k] for k in ["m", "w", "a", "b", "c1", "c2"]}
        X_hat = self._first_derivatives(obs[0, :], full_mle["tc"], params_hat)
        p = 6
        n = int(obs.shape[1])

        extras: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            fit = row.to_dict()
            s = fit["sse"] / max(n, 1)
            if not np.isfinite(s) or s <= 0:
                extras.append(
                    {
                        "log_lp": float("-inf"),
                        "log_lm": float("-inf"),
                        "m_lo": np.nan,
                        "m_hi": np.nan,
                        "w_lo": np.nan,
                        "w_hi": np.nan,
                        "D_lo": np.nan,
                        "D_hi": np.nan,
                        "qualified_strict": False,
                        "qualified_conf": False,
                    }
                )
                continue

            X_tc, _, J_tc, cov_psi = self._covariance_for_fit(obs, fit)
            K_tc = X_hat.T @ X_tc
            log_lp = -0.5 * n * math.log(s)
            log_adj_num = 0.5 * self._safe_slogdet(J_tc)
            log_adj_den = self._safe_slogdet(K_tc)
            log_lm = -0.5 * (n - p - 2) * math.log(s) + log_adj_num - log_adj_den
            qual = self._qualification_from_intervals(fit, cov_psi, cutoff, constraints)
            qual.update({"log_lp": log_lp, "log_lm": log_lm})
            extras.append(qual)

        df = pd.concat([df.reset_index(drop=True), pd.DataFrame(extras).reset_index(drop=True)], axis=1)
        max_lp = np.nanmax(df["log_lp"].to_numpy())
        max_lm = np.nanmax(df["log_lm"].to_numpy())
        df["rp"] = np.exp(df["log_lp"] - max_lp)
        df["rm"] = np.exp(df["log_lm"] - max_lm)
        df["window_n"] = n
        df["t1_time"] = self.value_to_timestamp(float(obs[0, 0]))
        df["t2_time"] = self.value_to_timestamp(float(obs[0, -1]))
        df["tc_time"] = self.values_to_timestamps(df["tc"].to_numpy())
        return df

    @staticmethod
    def _extract_intervals_from_curve(
        tc_values: np.ndarray,
        rm_values: np.ndarray,
        cutoff: float = 0.05,
        valid_mask: np.ndarray | None = None,
    ) -> list[dict[str, float]]:
        if valid_mask is None:
            valid_mask = np.ones_like(rm_values, dtype=bool)
        keep = np.asarray(rm_values >= cutoff, dtype=bool) & np.asarray(valid_mask, dtype=bool)
        if keep.size == 0:
            return []
        intervals: list[dict[str, float]] = []
        idx = 0
        while idx < len(keep):
            if not keep[idx]:
                idx += 1
                continue
            start = idx
            while idx + 1 < len(keep) and keep[idx + 1]:
                idx += 1
            end = idx
            seg_tc = tc_values[start : end + 1]
            seg_rm = rm_values[start : end + 1]
            peak_offset = int(np.nanargmax(seg_rm))
            intervals.append(
                {
                    "interval_lo": float(seg_tc[0]),
                    "interval_hi": float(seg_tc[-1]),
                    "peak_tc": float(seg_tc[peak_offset]),
                    "peak_rm": float(seg_rm[peak_offset]),
                }
            )
            idx += 1
        return intervals

    def _cluster_scenarios(
        self,
        peaks_df: pd.DataFrame,
        total_windows: int,
        current_t2: float,
        cluster_gap_days: int = 10,
    ) -> pd.DataFrame:
        if peaks_df.empty:
            return pd.DataFrame(
                columns=[
                    "scenario_id",
                    "peak_tc",
                    "peak_tc_time",
                    "interval_lo",
                    "interval_hi",
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
        peaks_df = peaks_df.sort_values("tc").reset_index(drop=True)
        cluster_ids: list[int] = []
        cluster_id = 0
        prev_tc: float | None = None
        for tc in peaks_df["tc"]:
            if prev_tc is None or abs(tc - prev_tc) > cluster_gap_days:
                cluster_id += 1
            cluster_ids.append(cluster_id)
            prev_tc = tc
        peaks_df = peaks_df.copy()
        peaks_df["cluster_id"] = cluster_ids

        scenario_rows: list[dict[str, Any]] = []
        for _, grp in peaks_df.groupby("cluster_id", sort=True):
            grp = grp.sort_values("rm", ascending=False)
            best = grp.iloc[0]
            support_windows = int(grp["window_size"].nunique())
            lo = float(grp["interval_lo"].min())
            hi = float(grp["interval_hi"].max())
            peak = float(best["tc"])
            scenario_rows.append(
                {
                    "scenario_id": f"S{len(scenario_rows) + 1}",
                    "peak_tc": peak,
                    "peak_tc_time": self.value_to_timestamp(peak),
                    "interval_lo": lo,
                    "interval_hi": hi,
                    "interval_lo_time": self.value_to_timestamp(lo),
                    "interval_hi_time": self.value_to_timestamp(hi),
                    "horizon_days": float(best["tc"] - current_t2),
                    "support_windows": support_windows,
                    "support_share": support_windows / max(total_windows, 1),
                    "rm_max": float(grp["rm"].max()),
                    "window_min": int(grp["window_size"].min()),
                    "window_max": int(grp["window_size"].max()),
                    "m": float(best["m"]),
                    "w": float(best["w"]),
                    "D": float(best["D"]),
                    "b": float(best["b"]),
                }
            )
        scenarios = pd.DataFrame(scenario_rows).sort_values(["rm_max", "support_windows"], ascending=[False, False])
        scenarios = scenarios.reset_index(drop=True)
        scenarios["scenario_id"] = [f"S{i + 1}" for i in range(len(scenarios))]
        return scenarios

    def scan_tc_surface(
        self,
        t2_index: int = -1,
        window_sizes: Iterable[int] | None = None,
        tc_grid: Iterable[float] | None = None,
        max_searches: int = 15,
        minimizer: str = "L-BFGS-B",
        bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.1, 0.9), (6.0, 13.0)),
        cutoff: float = 0.05,
        peak_cutoff: float = 0.2,
        constraints: FitConstraints | None = None,
        random_state: int | None = None,
    ) -> dict[str, Any]:
        constraints = constraints or FitConstraints()
        obs = self._as_array(self.observations)
        n_total = obs.shape[1]
        if t2_index < 0:
            t2_index = n_total + t2_index
        if t2_index <= 0 or t2_index >= n_total:
            raise IndexError("t2_index is out of range")

        current_t2 = float(obs[0, t2_index])
        if tc_grid is None:
            tc_grid = np.arange(int(current_t2) - 50, int(current_t2) + 151, 1)
        tc_grid = np.array(list(tc_grid), dtype=float)

        if window_sizes is None:
            max_ws = min(400, t2_index + 1)
            min_ws = min(120, max_ws)
            window_sizes = np.arange(min_ws, max_ws + 1, 20)
        window_sizes = [int(ws) for ws in window_sizes if 10 <= ws <= (t2_index + 1)]
        if not window_sizes:
            raise ValueError("No valid window sizes remain after filtering")

        curves: list[pd.DataFrame] = []
        interval_rows: list[dict[str, Any]] = []
        peak_rows: list[dict[str, Any]] = []
        for pos, ws in enumerate(sorted(window_sizes)):
            start = t2_index - ws + 1
            sub_obs = obs[:, start : t2_index + 1]
            curve = self.scan_tc_for_window(
                obs=sub_obs,
                tc_grid=tc_grid,
                max_searches=max_searches,
                minimizer=minimizer,
                bounds=bounds,
                cutoff=cutoff,
                constraints=constraints,
                random_state=None if random_state is None else random_state + pos * 10000,
            )
            curve["window_size"] = ws
            curves.append(curve)

            intervals = self._extract_intervals_from_curve(
                curve["tc"].to_numpy(),
                curve["rm"].to_numpy(),
                cutoff=cutoff,
                valid_mask=curve["qualified_conf"].to_numpy(),
            )
            for interval in intervals:
                interval_rows.append(
                    {
                        "window_size": ws,
                        **interval,
                        "interval_lo_time": self.value_to_timestamp(interval["interval_lo"]),
                        "interval_hi_time": self.value_to_timestamp(interval["interval_hi"]),
                        "peak_tc_time": self.value_to_timestamp(interval["peak_tc"]),
                    }
                )

            rm = curve["rm"].to_numpy()
            valid = curve["qualified_conf"].to_numpy().astype(bool)
            peaks, _ = find_peaks(rm)
            peak_candidates = [int(p) for p in peaks if valid[p] and rm[p] >= peak_cutoff]
            if rm.size:
                edge_best = int(np.nanargmax(rm))
                if valid[edge_best] and rm[edge_best] >= peak_cutoff and edge_best not in peak_candidates:
                    peak_candidates.append(edge_best)
            peak_candidates = sorted(set(peak_candidates), key=lambda i: rm[i], reverse=True)[:3]
            for peak_idx in peak_candidates:
                tc_val = float(curve.iloc[peak_idx]["tc"])
                containing = next(
                    (it for it in intervals if it["interval_lo"] <= tc_val <= it["interval_hi"]),
                    None,
                )
                interval_lo = containing["interval_lo"] if containing else tc_val
                interval_hi = containing["interval_hi"] if containing else tc_val
                peak_rows.append(
                    {
                        **curve.iloc[peak_idx].to_dict(),
                        "window_size": ws,
                        "interval_lo": interval_lo,
                        "interval_hi": interval_hi,
                        "interval_lo_time": self.value_to_timestamp(interval_lo),
                        "interval_hi_time": self.value_to_timestamp(interval_hi),
                        "peak_tc_time": self.value_to_timestamp(tc_val),
                    }
                )

        surface = pd.concat(curves, ignore_index=True)
        intervals_df = pd.DataFrame(interval_rows)
        peaks_df = pd.DataFrame(peak_rows)
        scenarios = self._cluster_scenarios(
            peaks_df=peaks_df,
            total_windows=len(window_sizes),
            current_t2=current_t2,
            cluster_gap_days=max(7, int(np.median(np.diff(tc_grid))) * 10 if len(tc_grid) > 1 else 10),
        )
        return {
            "surface": surface,
            "intervals": intervals_df,
            "peaks": peaks_df,
            "scenarios": scenarios,
            "tc_grid": tc_grid,
            "window_sizes": np.array(sorted(window_sizes), dtype=int),
            "current_t2": current_t2,
            "current_t2_time": self.value_to_timestamp(current_t2),
            "cutoff": cutoff,
            "peak_cutoff": peak_cutoff,
            "constraints": constraints,
        }

    # ------------------------------------------------------------------
    # plotting with hover
    # ------------------------------------------------------------------
    def plot_tc_structure(
        self,
        result: dict[str, Any],
        use_qualified: str = "qualified_conf",
        contour_levels: tuple[float, float, float] = (0.05, 0.5, 0.95),
        title: str | None = None,
        figsize: tuple[int, int] = (16, 10),
        prediction_events: pd.DataFrame | None = None,
    ) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
        surface = result["surface"].copy()
        scenarios = result["scenarios"].copy()
        current_t2 = float(result["current_t2"])
        cutoff = float(result["cutoff"])
        obs = self._as_array(self.observations)

        last_mask = obs[0, :] <= current_t2
        obs_plot = obs[:, last_mask]
        time_dates = self.values_to_timestamps(obs_plot[0, :])
        price_vals = np.exp(obs_plot[1, :])

        tc_vals = np.array(sorted(surface["tc"].unique()), dtype=float)
        ws_vals = np.array(sorted(surface["window_size"].unique()), dtype=int)
        pivot_rm = surface.pivot(index="window_size", columns="tc", values="rm").reindex(index=ws_vals, columns=tc_vals)
        pivot_q = surface.pivot(index="window_size", columns="tc", values=use_qualified).reindex(index=ws_vals, columns=tc_vals)
        rm_arr = pivot_rm.to_numpy(dtype=float)
        q_arr = pivot_q.to_numpy(dtype=bool)

        fig = plt.figure(figsize=figsize, facecolor="#f8fafc")
        gs = fig.add_gridspec(2, 1, height_ratios=[1.22, 1.0], hspace=0.14)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax1.set_facecolor("#fbfdff")
        ax2.set_facecolor("#fbfdff")
        for ax in (ax1, ax2):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax1.plot(time_dates, price_vals, linewidth=1.35, color="#2563eb", label="Price", zorder=2)
        ax1.axvline(
            self.value_to_timestamp(current_t2),
            linestyle="--",
            linewidth=1.2,
            color="#0f172a",
            alpha=0.75,
            label="Analysis time t2",
            zorder=3,
        )

        price_max = float(np.nanmax(price_vals))
        price_min = float(np.nanmin(price_vals))
        price_pad = max((price_max - price_min) * 0.06, price_max * 0.03, 1.0)
        scenario_y = price_max + price_pad * 0.35
        ax1.set_ylim(price_min - price_pad * 0.25, price_max + price_pad)
        scenario_x: list[float] = []
        scenario_hover_rows: list[dict[str, Any]] = []
        scenario_label_y: list[float] = []
        scenario_colors = plt.cm.Set2(np.linspace(0.15, 0.85, max(len(scenarios), 1)))
        for idx, (_, row) in enumerate(scenarios.iterrows()):
            lo_dt = pd.Timestamp(row["interval_lo_time"])
            hi_dt = pd.Timestamp(row["interval_hi_time"])
            peak_dt = pd.Timestamp(row["peak_tc_time"])
            color = scenario_colors[idx]
            label_y = scenario_y + (idx % 2) * price_pad * 0.18
            ax1.axvspan(lo_dt, hi_dt, color=color, alpha=0.10, zorder=1)
            ax1.axvline(peak_dt, linestyle="-.", linewidth=1.2, color=color, alpha=0.85, zorder=3)
            ax1.text(
                peak_dt,
                label_y,
                row["scenario_id"],
                ha="center",
                va="bottom",
                fontsize=9,
                color="#111827",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color, alpha=0.95),
                zorder=6,
            )
            scenario_x.append(mdates.date2num(peak_dt.to_pydatetime()))
            scenario_label_y.append(label_y)
            scenario_hover_rows.append(row.to_dict())
        scenario_scatter = None
        if scenario_x:
            scenario_scatter = ax1.scatter(
                [mdates.num2date(x) for x in scenario_x],
                scenario_label_y,
                s=44,
                alpha=0.0,
                zorder=5,
            )

        prediction_scatter = None
        prediction_hover_rows: list[dict[str, Any]] = []
        if prediction_events is not None and len(prediction_events) > 0:
            pred_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(prediction_events), 1)))
            pred_x: list[float] = []
            pred_y: list[float] = []
            for idx, (signal_time, row) in enumerate(prediction_events.iterrows(), start=1):
                color = pred_colors[idx - 1]
                tc_time = pd.Timestamp(row["tc_median_time"])
                lo_time = pd.Timestamp(row["tc_q10_time"])
                hi_time = pd.Timestamp(row["tc_q90_time"])
                pred_label_y = price_max + price_pad * (0.56 + 0.10 * ((idx - 1) % 2))
                ax1.axvspan(lo_time, hi_time, color=color, alpha=0.08, zorder=0)
                ax1.axvline(tc_time, color=color, linestyle="--", linewidth=1.15, alpha=0.95, zorder=3)
                if pd.notna(signal_time) and "signal_price" in row and pd.notna(row["signal_price"]):
                    ax1.scatter(
                        pd.Timestamp(signal_time),
                        float(row["signal_price"]),
                        marker="s",
                        s=38,
                        color=color,
                        edgecolors="white",
                        linewidths=0.75,
                        zorder=5,
                    )
                ax1.text(
                    tc_time,
                    pred_label_y,
                    f"P{int(row.get('event_id', idx))}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=color,
                    zorder=6,
                )
                pred_x.append(mdates.date2num(tc_time.to_pydatetime()))
                pred_y.append(pred_label_y)
                prediction_hover_rows.append(
                    {
                        "signal_time": pd.Timestamp(signal_time),
                        "tc_median_time": tc_time,
                        "tc_q10_time": lo_time,
                        "tc_q90_time": hi_time,
                        "signal_price": float(row["signal_price"]) if "signal_price" in row and pd.notna(row["signal_price"]) else np.nan,
                        "confidence": float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else np.nan,
                        "event_id": int(row["event_id"]) if "event_id" in row and pd.notna(row["event_id"]) else idx,
                    }
                )
            prediction_scatter = ax1.scatter(
                [mdates.num2date(x) for x in pred_x],
                pred_y,
                s=46,
                alpha=0.0,
                zorder=5,
            )

        ax1.text(
            0.01,
            0.98,
            "Hover the lower panel to inspect Rm(tc, Δt). Hover scenario labels above to inspect candidate intervals.",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color="0.35",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.85),
            zorder=10,
        )
        ax1.text(
            0.01,
            0.98,
            "Hover the lower panel to inspect Rm(tc, window). Hover scenario or prediction labels above for details.",
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="#475569",
            bbox=dict(boxstyle="round", fc="white", ec="#cbd5e1", alpha=0.92),
            zorder=11,
        )
        ax1.set_ylabel("Price")
        ax1.grid(True, linestyle="--", alpha=0.25, color="#94a3b8")
        ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#cbd5e1")

        tc_dates = self.values_to_timestamps(tc_vals)
        x_nums = mdates.date2num(tc_dates.to_pydatetime())
        extent = [x_nums.min(), x_nums.max(), ws_vals.min(), ws_vals.max()]
        rm_plot = np.ma.masked_where(~np.isfinite(rm_arr) | (rm_arr <= 0.0), rm_arr)
        cmap = plt.get_cmap("plasma").copy()
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        norm = mcolors.PowerNorm(gamma=0.65, vmin=0.0, vmax=max(float(np.nanmax(rm_arr)), 1e-6))
        img = ax2.imshow(
            rm_plot,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        unqualified = np.where(q_arr, np.nan, 1.0)
        ax2.imshow(
            unqualified,
            aspect="auto",
            origin="lower",
            extent=extent,
            interpolation="nearest",
            cmap="Greys",
            alpha=0.10,
        )
        if rm_arr.shape[0] > 1 and rm_arr.shape[1] > 1:
            X, Y = np.meshgrid(x_nums, ws_vals)
            try:
                ax2.contour(X, Y, rm_arr, levels=contour_levels, linewidths=0.95, colors="#f8fafc", alpha=0.9)
            except Exception:
                pass
        ax2.axvline(
            mdates.date2num(self.value_to_timestamp(current_t2).to_pydatetime()),
            linestyle="--",
            linewidth=1.2,
            color="#0f172a",
            alpha=0.75,
        )
        if len(scenarios):
            for idx, (_, row) in enumerate(scenarios.iterrows()):
                color = scenario_colors[idx]
                lo_num = mdates.date2num(pd.Timestamp(row["interval_lo_time"]).to_pydatetime())
                hi_num = mdates.date2num(pd.Timestamp(row["interval_hi_time"]).to_pydatetime())
                peak_num = mdates.date2num(pd.Timestamp(row["peak_tc_time"]).to_pydatetime())
                ax2.axvspan(lo_num, hi_num, color=color, alpha=0.08, zorder=0)
                ax2.axvline(peak_num, color=color, linestyle="-.", linewidth=1.0, alpha=0.8, zorder=2)
        if prediction_events is not None and len(prediction_events) > 0:
            pred_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(prediction_events), 1)))
            for idx, (_, row) in enumerate(prediction_events.iterrows()):
                color = pred_colors[idx]
                lo_num = mdates.date2num(pd.Timestamp(row["tc_q10_time"]).to_pydatetime())
                hi_num = mdates.date2num(pd.Timestamp(row["tc_q90_time"]).to_pydatetime())
                tc_num = mdates.date2num(pd.Timestamp(row["tc_median_time"]).to_pydatetime())
                ax2.axvspan(lo_num, hi_num, color=color, alpha=0.06, zorder=0)
                ax2.axvline(tc_num, color=color, linestyle="--", linewidth=1.0, alpha=0.9, zorder=2)
        ax2.set_ylabel("Window size")
        ax2.set_xlabel("Candidate t_c")
        cbar = fig.colorbar(img, ax=ax2, pad=0.01)
        cbar.set_label("Relative modified likelihood Rm(tc, window)")
        ax2.grid(False)
        cbar.set_label("Relative modified likelihood R_m(t_c, Δt)")
        cbar.set_label("Relative modified likelihood Rm(tc, window)")
        ax2.xaxis_date()
        ax1.xaxis_date()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

        if title is None:
            title = f"LPPLS modified-profile-likelihood t_c structure | t2={result['current_t2_time']} | cutoff={cutoff:.2f}"
        fig.suptitle(title)

        annot = ax2.annotate(
            "",
            xy=(0, 0),
            xytext=(18, 18),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="white", alpha=0.92),
            arrowprops=dict(arrowstyle="->", alpha=0.75),
        )
        annot.set_visible(False)

        def _show_annotation(ax: plt.Axes, x: float, y: float, text: str) -> None:
            annot.xy = (x, y)
            annot.set_text(text)
            annot.set_visible(True)
            annot.axes = ax
            fig.canvas.draw_idle()

        def _hide_annotation() -> None:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()

        surface_index = surface.set_index(["window_size", "tc"]).sort_index()

        def _on_move(event: Any) -> None:
            if event.inaxes == ax2 and event.xdata is not None and event.ydata is not None:
                tc_idx = int(np.argmin(np.abs(x_nums - event.xdata)))
                ws_idx = int(np.argmin(np.abs(ws_vals - event.ydata)))
                tc_val = float(tc_vals[tc_idx])
                ws_val = int(ws_vals[ws_idx])
                try:
                    row = surface_index.loc[(ws_val, tc_val)]
                    text = (
                        f"Δt={ws_val}\n"
                        f"t_c={pd.Timestamp(row['tc_time']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Rm={row['rm']:.3f}  Rp={row['rp']:.3f}\n"
                        f"qualified={bool(row[use_qualified])}\n"
                        f"m={row['m']:.3f}  ω={row['w']:.3f}\n"
                        f"D={row['D']:.3f}  B={row['b']:.3g}\n"
                        f"5% LI(m)=({row['m_lo']:.3f}, {row['m_hi']:.3f})\n"
                        f"5% LI(ω)=({row['w_lo']:.3f}, {row['w_hi']:.3f})\n"
                        f"5% LI(D)=({row['D_lo']:.3f}, {row['D_hi']:.3f})"
                    )
                    text = (
                        f"window={ws_val}\n"
                        f"t_c={pd.Timestamp(row['tc_time']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Rm={row['rm']:.3f}  Rp={row['rp']:.3f}\n"
                        f"qualified={bool(row[use_qualified])}\n"
                        f"m={row['m']:.3f}  omega={row['w']:.3f}\n"
                        f"D={row['D']:.3f}  B={row['b']:.3g}\n"
                        f"5% LI(m)=({row['m_lo']:.3f}, {row['m_hi']:.3f})\n"
                        f"5% LI(omega)=({row['w_lo']:.3f}, {row['w_hi']:.3f})\n"
                        f"5% LI(D)=({row['D_lo']:.3f}, {row['D_hi']:.3f})"
                    )
                    _show_annotation(ax2, x_nums[tc_idx], ws_vals[ws_idx], text)
                    return
                except Exception:
                    pass
            if scenario_scatter is not None and event.inaxes == ax1:
                contains, info = scenario_scatter.contains(event)
                if contains:
                    idx = int(info["ind"][0])
                    row = scenario_hover_rows[idx]
                    text = (
                        f"{row['scenario_id']}\n"
                        f"peak t_c={pd.Timestamp(row['peak_tc_time']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"5% interval=[{pd.Timestamp(row['interval_lo_time']).strftime('%Y-%m-%d %H:%M:%S')},\n"
                        f"            {pd.Timestamp(row['interval_hi_time']).strftime('%Y-%m-%d %H:%M:%S')}]\n"
                        f"horizon={row['horizon_days']:.0f} days\n"
                        f"support={row['support_windows']} windows ({row['support_share']:.0%})\n"
                        f"Δt range={row['window_min']}–{row['window_max']}\n"
                        f"Rm(max)={row['rm_max']:.3f}\n"
                        f"m={row['m']:.3f}  ω={row['w']:.3f}  D={row['D']:.3f}"
                    )
                    text = (
                        f"{row['scenario_id']}\n"
                        f"peak t_c={pd.Timestamp(row['peak_tc_time']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"5% interval=[{pd.Timestamp(row['interval_lo_time']).strftime('%Y-%m-%d %H:%M:%S')},\n"
                        f"            {pd.Timestamp(row['interval_hi_time']).strftime('%Y-%m-%d %H:%M:%S')}]\n"
                        f"horizon={row['horizon_days']:.0f} days\n"
                        f"support={row['support_windows']} windows ({row['support_share']:.0%})\n"
                        f"window range={row['window_min']}..{row['window_max']}\n"
                        f"Rm(max)={row['rm_max']:.3f}\n"
                        f"m={row['m']:.3f}  omega={row['w']:.3f}  D={row['D']:.3f}"
                    )
                    _show_annotation(ax1, scenario_x[idx], scenario_label_y[idx], text)
                    return
            if prediction_scatter is not None and event.inaxes == ax1:
                contains, info = prediction_scatter.contains(event)
                if contains:
                    idx = int(info["ind"][0])
                    row = prediction_hover_rows[idx]
                    conf_text = "nan" if not np.isfinite(row["confidence"]) else f"{row['confidence']:.3f}"
                    price_text = "nan" if not np.isfinite(row["signal_price"]) else f"{row['signal_price']:.4f}"
                    text = (
                        f"P{row['event_id']}\n"
                        f"signal={pd.Timestamp(row['signal_time']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"signal price={price_text}  conf={conf_text}\n"
                        f"tc={pd.Timestamp(row['tc_median_time']).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"band=[{pd.Timestamp(row['tc_q10_time']).strftime('%Y-%m-%d %H:%M:%S')},\n"
                        f"      {pd.Timestamp(row['tc_q90_time']).strftime('%Y-%m-%d %H:%M:%S')}]"
                    )
                    _show_annotation(ax1, prediction_scatter.get_offsets()[idx, 0], prediction_scatter.get_offsets()[idx, 1], text)
                    return
            _hide_annotation()

        fig.canvas.mpl_connect("motion_notify_event", _on_move)
        fig.tight_layout()
        return fig, (ax1, ax2)


__all__ = ["FitConstraints", "LPPLSModifiedTC"]
