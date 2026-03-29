from __future__ import annotations

import base64
import io
import json
import math
import re
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from jinja2 import Template
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy import integrate, optimize, stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
REPORT_DIR = PROJECT_DIR / "reports"
HTML_PATH = REPORT_DIR / "negative_tail_powerlaw_report.html"
JSON_PATH = REPORT_DIR / "negative_tail_powerlaw_results.json"

CLASSIC_XMIN = 2.0
CI_BOOTSTRAP_REPS = 800
GOF_BOOTSTRAP_REPS = 300
MAX_XMIN_CANDIDATES = 180
RANDOM_SEED = 20260325

FILE_RE = re.compile(r"sample_(?P<instrument>.+)_(?P<scale>\d+[A-Za-z]+)\.csv$")


@dataclass
class DatasetMeta:
    dataset_id: str
    instrument: str
    scale: str
    file_name: str
    start_time: str
    end_time: str
    rows: int
    returns_count: int
    negative_count: int
    mean_return: float
    std_return: float
    tail_mean: float
    tail_max: float


def format_scale_for_sort(scale: str) -> tuple[int, int, str]:
    match = re.match(r"(?P<num>\d+)(?P<unit>[A-Za-z]+)", scale)
    if not match:
        return (99, 0, scale)
    num = int(match.group("num"))
    unit = match.group("unit").lower()
    order = {"s": 0, "m": 1, "h": 2, "d": 3, "w": 4}
    return (order.get(unit, 98), num, scale)


def load_datasets() -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        instrument = match.group("instrument").upper()
        scale = match.group("scale")
        df = pd.read_csv(
            path,
            header=None,
            names=["time", "open", "high", "low", "close", "volume"],
        )
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        for column in ["open", "high", "low", "close", "volume"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
        if len(df) < 10:
            continue
        close = df["close"].astype(float)
        log_returns = np.log(close).diff()
        simple_returns = close.pct_change()
        valid_mask = log_returns.notna()
        returns = log_returns[valid_mask].reset_index(drop=True)
        return_times = df.loc[valid_mask, "time"].reset_index(drop=True)
        simple_returns = simple_returns[valid_mask].reset_index(drop=True)
        mean_return = float(returns.mean())
        std_return = float(returns.std(ddof=1))
        standardized = ((returns - mean_return) / std_return).reset_index(drop=True)
        tail_table = pd.DataFrame(
            {
                "time": return_times,
                "log_return": returns,
                "simple_return": simple_returns,
                "standardized_return": standardized,
            }
        )
        negative_tail_table = tail_table[tail_table["standardized_return"] < 0].copy()
        negative_tail_table["tail_x"] = -negative_tail_table["standardized_return"]
        negative_tail_table = negative_tail_table.sort_values("tail_x").reset_index(drop=True)
        negative_tail = negative_tail_table["tail_x"].to_numpy()
        dataset_id = f"{instrument}_{scale}"
        meta = DatasetMeta(
            dataset_id=dataset_id,
            instrument=instrument,
            scale=scale,
            file_name=path.name,
            start_time=str(df["time"].min()),
            end_time=str(df["time"].max()),
            rows=int(len(df)),
            returns_count=int(len(returns)),
            negative_count=int(len(negative_tail)),
            mean_return=mean_return,
            std_return=std_return,
            tail_mean=float(np.mean(negative_tail)),
            tail_max=float(np.max(negative_tail)),
        )
        datasets.append(
            {
                "meta": meta,
                "df": df,
                "returns": returns.to_numpy(),
                "standardized": standardized.to_numpy(),
                "negative_tail": np.sort(negative_tail),
                "negative_tail_table": negative_tail_table,
            }
        )
    datasets.sort(key=lambda item: (item["meta"].instrument, format_scale_for_sort(item["meta"].scale)))
    return datasets


def empirical_ccdf(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(sample, dtype=float))
    if values.size == 0:
        return np.array([]), np.array([])
    unique, counts = np.unique(values, return_counts=True)
    survivors = np.cumsum(counts[::-1])[::-1]
    ccdf = survivors / values.size
    return unique, ccdf


def fit_classic_ccdf(sample: np.ndarray, xmin: float = CLASSIC_XMIN) -> dict[str, Any]:
    tail = np.sort(sample[sample >= xmin])
    x_plot, ccdf_plot = empirical_ccdf(tail)
    result: dict[str, Any] = {
        "xmin": xmin,
        "n_tail": int(tail.size),
        "alpha_ccdf": np.nan,
        "alpha_pdf": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
        "tail": tail,
        "plot_x": x_plot,
        "plot_ccdf": ccdf_plot,
    }
    if tail.size < 5 or x_plot.size < 2:
        return result
    log_x = np.log(x_plot)
    log_y = np.log(ccdf_plot)
    slope, intercept = np.polyfit(log_x, log_y, deg=1)
    fitted = intercept + slope * log_x
    ss_res = np.sum((log_y - fitted) ** 2)
    ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
    alpha_ccdf = -float(slope)
    result["alpha_ccdf"] = alpha_ccdf
    result["alpha_pdf"] = alpha_ccdf + 1.0
    result["intercept"] = float(intercept)
    result["r_squared"] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return result


def estimate_power_law_tail(tail: np.ndarray, xmin: float) -> dict[str, Any]:
    values = np.sort(np.asarray(tail, dtype=float))
    if values.size < 2:
        return {
            "xmin": xmin,
            "n_tail": int(values.size),
            "alpha_ccdf": np.nan,
            "alpha_pdf": np.nan,
            "ks": np.nan,
            "loglik": np.nan,
            "aic": np.nan,
            "sorted_tail": values,
        }
    log_ratio = np.log(values / xmin)
    denom = float(np.sum(log_ratio))
    if not np.isfinite(denom) or denom <= 0:
        return {
            "xmin": xmin,
            "n_tail": int(values.size),
            "alpha_ccdf": np.nan,
            "alpha_pdf": np.nan,
            "ks": np.nan,
            "loglik": np.nan,
            "aic": np.nan,
            "sorted_tail": values,
        }
    alpha_ccdf = float(values.size / denom)
    logpdf = np.log(alpha_ccdf) + alpha_ccdf * np.log(xmin) - (alpha_ccdf + 1.0) * np.log(values)
    loglik = float(np.sum(logpdf))
    ecdf = np.arange(1, values.size + 1) / values.size
    model_cdf = 1.0 - np.power(values / xmin, -alpha_ccdf)
    ks = float(np.max(np.abs(ecdf - model_cdf)))
    return {
        "xmin": float(xmin),
        "n_tail": int(values.size),
        "alpha_ccdf": alpha_ccdf,
        "alpha_pdf": alpha_ccdf + 1.0,
        "ks": ks,
        "loglik": loglik,
        "aic": float(2.0 - 2.0 * loglik),
        "sorted_tail": values,
        "logpdf": logpdf,
    }


def build_xmin_candidates(sample: np.ndarray, min_tail: int, max_candidates: int = MAX_XMIN_CANDIDATES) -> np.ndarray:
    values = np.sort(np.asarray(sample, dtype=float))
    unique = np.unique(values)
    if unique.size == 0:
        return unique
    tail_counts = values.size - np.searchsorted(values, unique, side="left")
    valid = unique[tail_counts >= min_tail]
    if valid.size == 0:
        fallback_min_tail = max(8, min(values.size, min_tail // 2))
        valid = unique[tail_counts >= fallback_min_tail]
    if valid.size == 0:
        valid = unique[:-1] if unique.size > 1 else unique
    if valid.size > max_candidates:
        picks = np.linspace(0, valid.size - 1, max_candidates, dtype=int)
        valid = valid[picks]
        valid = np.unique(valid)
    return valid


def fit_modern_power_law(sample: np.ndarray, max_candidates: int = MAX_XMIN_CANDIDATES) -> dict[str, Any]:
    values = np.sort(np.asarray(sample, dtype=float))
    min_tail = max(10, min(50, values.size // 10))
    candidates = build_xmin_candidates(values, min_tail=min_tail, max_candidates=max_candidates)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for xmin in candidates:
        tail = values[values >= xmin]
        fit = estimate_power_law_tail(tail, float(xmin))
        if not np.isfinite(fit["ks"]):
            continue
        rows.append(
            {
                "xmin": float(xmin),
                "n_tail": int(fit["n_tail"]),
                "alpha_ccdf": float(fit["alpha_ccdf"]),
                "alpha_pdf": float(fit["alpha_pdf"]),
                "ks": float(fit["ks"]),
            }
        )
        if best is None or fit["ks"] < best["ks"] or (
            math.isclose(fit["ks"], best["ks"], rel_tol=1e-12, abs_tol=1e-12)
            and fit["xmin"] > best["xmin"]
        ):
            best = fit
    if best is None:
        best = estimate_power_law_tail(values, float(np.min(values)))
        rows.append(
            {
                "xmin": float(best["xmin"]),
                "n_tail": int(best["n_tail"]),
                "alpha_ccdf": float(best["alpha_ccdf"]),
                "alpha_pdf": float(best["alpha_pdf"]),
                "ks": float(best["ks"]),
            }
        )
    best["xmin_scan"] = rows
    best["min_tail_rule"] = int(min_tail)
    return best


def bootstrap_alpha_ci(tail: np.ndarray, xmin: float, reps: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.asarray(tail, dtype=float)
    estimates = np.empty(reps, dtype=float)
    for idx in range(reps):
        sample = rng.choice(values, size=values.size, replace=True)
        estimates[idx] = estimate_power_law_tail(sample, xmin)["alpha_ccdf"]
    return {
        "bootstrap_alpha_ccdf": estimates,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "std": float(np.std(estimates, ddof=1)),
    }


def sample_power_law_tail(size: int, xmin: float, alpha_ccdf: float, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(size)
    return xmin * np.power(1.0 - u, -1.0 / alpha_ccdf)


def bootstrap_gof_p_value(
    sample: np.ndarray,
    alpha_ccdf: float,
    xmin: float,
    empirical_ks: float,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    values = np.sort(np.asarray(sample, dtype=float))
    body = values[values < xmin]
    p_body = float(body.size / values.size)
    ks_values = np.empty(reps, dtype=float)
    for idx in range(reps):
        draw_body_mask = rng.random(values.size) < p_body
        n_body = int(np.sum(draw_body_mask))
        n_tail = int(values.size - n_body)
        sampled_body = rng.choice(body, size=n_body, replace=True) if body.size and n_body else np.array([], dtype=float)
        sampled_tail = sample_power_law_tail(n_tail, xmin, alpha_ccdf, rng) if n_tail else np.array([], dtype=float)
        bootstrap_sample = np.sort(np.concatenate([sampled_body, sampled_tail]))
        fit = fit_modern_power_law(bootstrap_sample, max_candidates=120)
        ks_values[idx] = fit["ks"]
    p_value = float(np.mean(ks_values >= empirical_ks))
    return {
        "bootstrap_ks": ks_values,
        "p_value": p_value,
    }


def fit_exponential_tail(tail: np.ndarray, xmin: float) -> dict[str, Any]:
    values = np.sort(np.asarray(tail, dtype=float))
    excess = values - xmin
    mean_excess = float(np.mean(excess))
    if not np.isfinite(mean_excess) or mean_excess <= 0:
        return {"lambda": np.nan, "loglik": np.nan, "aic": np.nan, "ks": np.nan, "logpdf": np.full(values.size, np.nan)}
    lam = 1.0 / mean_excess
    logpdf = np.log(lam) - lam * excess
    loglik = float(np.sum(logpdf))
    ecdf = np.arange(1, values.size + 1) / values.size
    model_cdf = 1.0 - np.exp(-lam * excess)
    ks = float(np.max(np.abs(ecdf - model_cdf)))
    return {
        "lambda": float(lam),
        "loglik": loglik,
        "aic": float(2.0 - 2.0 * loglik),
        "ks": ks,
        "logpdf": logpdf,
    }


def fit_lognormal_tail(tail: np.ndarray, xmin: float) -> dict[str, Any]:
    values = np.sort(np.asarray(tail, dtype=float))
    log_values = np.log(values)

    def objective(params: np.ndarray) -> float:
        mu, log_sigma = params
        sigma = float(np.exp(log_sigma))
        zmin = (np.log(xmin) - mu) / sigma
        sf = float(stats.norm.sf(zmin))
        if not np.isfinite(sf) or sf <= 0:
            return np.inf
        logpdf = (
            -np.log(values)
            - np.log(sigma)
            - 0.5 * np.log(2.0 * np.pi)
            - 0.5 * np.square((log_values - mu) / sigma)
            - np.log(sf)
        )
        return float(-np.sum(logpdf))

    mu0 = float(np.mean(log_values))
    sigma0 = float(np.std(log_values, ddof=1))
    sigma0 = max(sigma0, 0.1)
    result = optimize.minimize(
        objective,
        x0=np.array([mu0, np.log(sigma0)]),
        method="L-BFGS-B",
        bounds=[(-10.0, 10.0), (np.log(1e-3), np.log(10.0))],
    )
    mu, log_sigma = result.x
    sigma = float(np.exp(log_sigma))
    zmin = (np.log(xmin) - mu) / sigma
    sf = float(stats.norm.sf(zmin))
    logpdf = (
        -np.log(values)
        - np.log(sigma)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * np.square((log_values - mu) / sigma)
        - np.log(sf)
    )
    loglik = float(np.sum(logpdf))
    ecdf = np.arange(1, values.size + 1) / values.size
    cdf_xmin = float(stats.norm.cdf(zmin))
    model_cdf = (stats.norm.cdf((log_values - mu) / sigma) - cdf_xmin) / max(sf, 1e-12)
    model_cdf = np.clip(model_cdf, 0.0, 1.0)
    ks = float(np.max(np.abs(ecdf - model_cdf)))
    return {
        "mu": float(mu),
        "sigma": sigma,
        "loglik": loglik,
        "aic": float(4.0 - 2.0 * loglik),
        "ks": ks,
        "logpdf": logpdf,
    }


def fit_truncated_power_law_tail(tail: np.ndarray, xmin: float, alpha_guess: float) -> dict[str, Any]:
    values = np.sort(np.asarray(tail, dtype=float))
    sum_log = float(np.sum(np.log(values)))
    sum_values = float(np.sum(values))
    n = int(values.size)

    def normalization(alpha_ccdf: float, lam: float) -> float:
        integral, _ = integrate.quad(
            lambda x: np.power(x, -(alpha_ccdf + 1.0)) * np.exp(-lam * x),
            xmin,
            np.inf,
            limit=200,
        )
        return float(integral)

    def objective(params: np.ndarray) -> float:
        log_alpha, log_lam = params
        alpha_ccdf = float(np.exp(log_alpha))
        lam = float(np.exp(log_lam))
        z = normalization(alpha_ccdf, lam)
        if not np.isfinite(z) or z <= 0:
            return np.inf
        return float((alpha_ccdf + 1.0) * sum_log + lam * sum_values + n * np.log(z))

    lam0 = max(1e-3, 1.0 / max(float(np.mean(values - xmin)), 1e-3))
    result = optimize.minimize(
        objective,
        x0=np.array([np.log(max(alpha_guess, 0.1)), np.log(lam0)]),
        method="L-BFGS-B",
        bounds=[(np.log(1e-3), np.log(20.0)), (np.log(1e-4), np.log(50.0))],
    )
    log_alpha, log_lam = result.x
    alpha_ccdf = float(np.exp(log_alpha))
    lam = float(np.exp(log_lam))
    z = normalization(alpha_ccdf, lam)
    logpdf = -(alpha_ccdf + 1.0) * np.log(values) - lam * values - np.log(z)
    loglik = float(np.sum(logpdf))
    return {
        "alpha_ccdf": alpha_ccdf,
        "alpha_pdf": alpha_ccdf + 1.0,
        "lambda": lam,
        "loglik": loglik,
        "aic": float(4.0 - 2.0 * loglik),
        "ks": np.nan,
        "logpdf": logpdf,
    }


def vuong_from_logpdf(reference_logpdf: np.ndarray, alt_logpdf: np.ndarray) -> tuple[float, float]:
    diff = np.asarray(reference_logpdf - alt_logpdf, dtype=float)
    lr = float(np.sum(diff))
    if diff.size < 2:
        return lr, np.nan
    variance = float(np.var(diff, ddof=1))
    if not np.isfinite(variance) or variance <= 0:
        return lr, np.nan
    z_score = lr / math.sqrt(diff.size * variance)
    p_value = float(2.0 * stats.norm.sf(abs(z_score)))
    return lr, p_value


def analyze_dataset(dataset: dict[str, Any], seed_offset: int) -> dict[str, Any]:
    meta: DatasetMeta = dataset["meta"]
    negative_tail = dataset["negative_tail"]
    negative_tail_table: pd.DataFrame = dataset["negative_tail_table"]

    classic = fit_classic_ccdf(negative_tail, xmin=CLASSIC_XMIN)
    modern = fit_modern_power_law(negative_tail)
    modern_tail_table = negative_tail_table[negative_tail_table["tail_x"] >= modern["xmin"]].copy().sort_values("tail_x").reset_index(drop=True)
    if len(modern_tail_table):
        modern_tail_table["survivors"] = np.arange(len(modern_tail_table), 0, -1, dtype=int)
        modern_tail_table["conditional_ccdf"] = modern_tail_table["survivors"] / len(modern_tail_table)
    else:
        modern_tail_table["survivors"] = []
        modern_tail_table["conditional_ccdf"] = []

    ci = bootstrap_alpha_ci(
        modern["sorted_tail"],
        xmin=modern["xmin"],
        reps=CI_BOOTSTRAP_REPS,
        seed=RANDOM_SEED + seed_offset,
    )
    gof = bootstrap_gof_p_value(
        negative_tail,
        alpha_ccdf=modern["alpha_ccdf"],
        xmin=modern["xmin"],
        empirical_ks=modern["ks"],
        reps=GOF_BOOTSTRAP_REPS,
        seed=RANDOM_SEED + 10_000 + seed_offset,
    )
    exponential = fit_exponential_tail(modern["sorted_tail"], modern["xmin"])
    lognormal = fit_lognormal_tail(modern["sorted_tail"], modern["xmin"])
    truncated = fit_truncated_power_law_tail(modern["sorted_tail"], modern["xmin"], modern["alpha_ccdf"])

    power_law_logpdf = modern["logpdf"]
    exp_lr, exp_p = vuong_from_logpdf(power_law_logpdf, exponential["logpdf"])
    logn_lr, logn_p = vuong_from_logpdf(power_law_logpdf, lognormal["logpdf"])

    comparisons = [
        {
            "dataset_id": meta.dataset_id,
            "instrument": meta.instrument,
            "scale": meta.scale,
            "model": "power_law",
            "loglik": float(modern["loglik"]),
            "aic": float(modern["aic"]),
            "ks": float(modern["ks"]),
            "llr_vs_power_law": 0.0,
            "p_value": np.nan,
        },
        {
            "dataset_id": meta.dataset_id,
            "instrument": meta.instrument,
            "scale": meta.scale,
            "model": "exponential",
            "loglik": float(exponential["loglik"]),
            "aic": float(exponential["aic"]),
            "ks": float(exponential["ks"]),
            "llr_vs_power_law": float(exp_lr),
            "p_value": float(exp_p),
        },
        {
            "dataset_id": meta.dataset_id,
            "instrument": meta.instrument,
            "scale": meta.scale,
            "model": "lognormal",
            "loglik": float(lognormal["loglik"]),
            "aic": float(lognormal["aic"]),
            "ks": float(lognormal["ks"]),
            "llr_vs_power_law": float(logn_lr),
            "p_value": float(logn_p),
        },
        {
            "dataset_id": meta.dataset_id,
            "instrument": meta.instrument,
            "scale": meta.scale,
            "model": "truncated_power_law",
            "loglik": float(truncated["loglik"]),
            "aic": float(truncated["aic"]),
            "ks": np.nan,
            "llr_vs_power_law": float(modern["loglik"] - truncated["loglik"]),
            "p_value": np.nan,
        },
    ]

    return {
        "meta": asdict(meta),
        "classic": {
            "xmin": float(classic["xmin"]),
            "n_tail": int(classic["n_tail"]),
            "alpha_ccdf": float(classic["alpha_ccdf"]),
            "alpha_pdf": float(classic["alpha_pdf"]),
            "intercept": float(classic["intercept"]),
            "r_squared": float(classic["r_squared"]),
            "plot_x": classic["plot_x"].tolist(),
            "plot_ccdf": classic["plot_ccdf"].tolist(),
        },
        "modern": {
            "xmin": float(modern["xmin"]),
            "n_tail": int(modern["n_tail"]),
            "alpha_ccdf": float(modern["alpha_ccdf"]),
            "alpha_pdf": float(modern["alpha_pdf"]),
            "ks": float(modern["ks"]),
            "loglik": float(modern["loglik"]),
            "aic": float(modern["aic"]),
            "min_tail_rule": int(modern["min_tail_rule"]),
            "xmin_scan": modern["xmin_scan"],
            "tail_x": modern["sorted_tail"].tolist(),
            "tail_observations": [
                {
                    "time": row.time.strftime("%Y-%m-%d %H:%M:%S"),
                    "tail_x": float(row.tail_x),
                    "standardized_return": float(row.standardized_return),
                    "log_return": float(row.log_return),
                    "simple_return_pct": float(row.simple_return * 100.0),
                    "survivors": int(row.survivors),
                    "conditional_ccdf": float(row.conditional_ccdf),
                }
                for row in modern_tail_table.itertuples(index=False)
            ],
        },
        "bootstrap": {
            "ci_low": float(ci["ci_low"]),
            "ci_high": float(ci["ci_high"]),
            "std": float(ci["std"]),
        },
        "gof": {
            "p_value": float(gof["p_value"]),
        },
        "comparisons": comparisons,
    }


def build_pooled_classic_results(results: list[dict[str, Any]], original_datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    instruments = {item["meta"]["instrument"] for item in results}
    if len(instruments) < 2:
        return []
    grouped: dict[str, list[np.ndarray]] = {}
    for dataset in original_datasets:
        scale = dataset["meta"].scale
        grouped.setdefault(scale, []).append(dataset["negative_tail"])
    pooled_results: list[dict[str, Any]] = []
    for scale, samples in grouped.items():
        pooled_sample = np.sort(np.concatenate(samples))
        fit = fit_classic_ccdf(pooled_sample, xmin=CLASSIC_XMIN)
        pooled_results.append(
            {
                "scale": scale,
                "n_tail": int(fit["n_tail"]),
                "alpha_ccdf": float(fit["alpha_ccdf"]),
                "alpha_pdf": float(fit["alpha_pdf"]),
                "r_squared": float(fit["r_squared"]),
            }
        )
    pooled_results.sort(key=lambda item: format_scale_for_sort(item["scale"]))
    return pooled_results


def save_figure(fig: plt.Figure, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_classic_ccdf(results: list[dict[str, Any]], asset_dir: Path) -> str:
    cols = 2
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, result in zip(axes, results):
        classic = result["classic"]
        x = np.asarray(classic["plot_x"], dtype=float)
        y = np.asarray(classic["plot_ccdf"], dtype=float)
        if x.size:
            ax.loglog(x, y, "o", color="#1d4ed8", alpha=0.75, label="Empirical CCDF")
        ax.axvline(CLASSIC_XMIN, color="#ef4444", linestyle="--", linewidth=1.2, label="xmin = 2")
        meta = result["meta"]
        ax.set_title(f"{meta['instrument']} {meta['scale']}")
        ax.set_xlabel("Negative tail magnitude x")
        ax.set_ylabel("P(X >= x)")
        ax.grid(True, which="both", linestyle=":", alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle("Classical empirical CCDF of negative standardized returns", fontsize=15, y=1.02)
    fig.tight_layout()
    return save_figure(fig, asset_dir / "classic_ccdf.png")


def plot_fitted_loglog(results: list[dict[str, Any]], asset_dir: Path) -> str:
    cols = 2
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, result in zip(axes, results):
        meta = result["meta"]
        x_all, y_all = empirical_ccdf(np.asarray(result["modern"]["tail_x"], dtype=float))
        x_classic = np.asarray(result["classic"]["plot_x"], dtype=float)
        modern_xmin = result["modern"]["xmin"]
        modern_alpha = result["modern"]["alpha_ccdf"]
        classic_alpha = result["classic"]["alpha_ccdf"]

        if x_all.size:
            ax.loglog(x_all, y_all, "o", color="#0f766e", alpha=0.6, markersize=3.5, label="Tail sample")
        if x_classic.size and np.isfinite(classic_alpha):
            grid = np.linspace(max(CLASSIC_XMIN, np.min(x_classic)), np.max(x_classic), 300)
            classic_fit = np.exp(result["classic"]["intercept"]) * np.power(grid, -classic_alpha)
            ax.loglog(grid, classic_fit, color="#ef4444", linewidth=2.0, label=f"Classic fit alpha={classic_alpha:.2f}")
        if np.isfinite(modern_alpha):
            x_modern = np.asarray(result["modern"]["tail_x"], dtype=float)
            grid = np.linspace(np.min(x_modern), np.max(x_modern), 300)
            modern_fit = np.power(grid / modern_xmin, -modern_alpha)
            ax.loglog(grid, modern_fit, color="#1d4ed8", linewidth=2.0, label=f"Modern fit alpha={modern_alpha:.2f}")
            ax.axvline(modern_xmin, color="#1d4ed8", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.axvline(CLASSIC_XMIN, color="#ef4444", linestyle=":", linewidth=1.1, alpha=0.8)
        ax.set_title(f"{meta['instrument']} {meta['scale']}")
        ax.set_xlabel("Negative tail magnitude x")
        ax.set_ylabel("Conditional tail CCDF")
        ax.grid(True, which="both", linestyle=":", alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle("Log-log tail fit: classical threshold and modern automatic threshold", fontsize=15, y=1.02)
    fig.tight_layout()
    return save_figure(fig, asset_dir / "fitted_loglog.png")


def plot_alpha_by_scale(results: list[dict[str, Any]], asset_dir: Path) -> str:
    ordered = sorted(results, key=lambda item: (item["meta"]["instrument"], format_scale_for_sort(item["meta"]["scale"])))
    labels = [item["meta"]["scale"] for item in ordered]
    x = np.arange(len(labels))
    classic = np.array([item["classic"]["alpha_ccdf"] for item in ordered], dtype=float)
    modern = np.array([item["modern"]["alpha_ccdf"] for item in ordered], dtype=float)
    ci_low = np.array([item["bootstrap"]["ci_low"] for item in ordered], dtype=float)
    ci_high = np.array([item["bootstrap"]["ci_high"] for item in ordered], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, classic, marker="o", linewidth=2, color="#ef4444", label="Classic alpha (CCDF)")
    ax.errorbar(
        x,
        modern,
        yerr=np.vstack([modern - ci_low, ci_high - modern]),
        fmt="s-",
        linewidth=2,
        capsize=5,
        color="#1d4ed8",
        label="Modern alpha with 95% bootstrap CI",
    )
    ax.axhline(3.0, color="#111827", linestyle="--", linewidth=1.2, label="Inverse cubic benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CCDF tail exponent alpha")
    ax.set_xlabel("Time scale")
    ax.set_title("Tail exponent comparison across time scales")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, asset_dir / "alpha_by_scale.png")


def plot_alpha_by_instrument(results: list[dict[str, Any]], asset_dir: Path) -> str:
    instruments = sorted({item["meta"]["instrument"] for item in results})
    scales = sorted({item["meta"]["scale"] for item in results}, key=format_scale_for_sort)
    x = np.arange(len(instruments))
    width = 0.14 if scales else 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, scale in enumerate(scales):
        subset = {item["meta"]["instrument"]: item for item in results if item["meta"]["scale"] == scale}
        y = np.array([subset.get(inst, {}).get("modern", {}).get("alpha_ccdf", np.nan) for inst in instruments], dtype=float)
        ci_low = np.array([subset.get(inst, {}).get("bootstrap", {}).get("ci_low", np.nan) for inst in instruments], dtype=float)
        ci_high = np.array([subset.get(inst, {}).get("bootstrap", {}).get("ci_high", np.nan) for inst in instruments], dtype=float)
        offset = (idx - (len(scales) - 1) / 2.0) * width
        ax.errorbar(
            x + offset,
            y,
            yerr=np.vstack([y - ci_low, ci_high - y]),
            fmt="o",
            capsize=4,
            linewidth=1.2,
            label=scale,
        )
    ax.axhline(3.0, color="#111827", linestyle="--", linewidth=1.2, label="Inverse cubic benchmark")
    ax.set_xticks(x)
    ax.set_xticklabels(instruments)
    ax.set_xlabel("Instrument")
    ax.set_ylabel("Modern alpha (CCDF)")
    ax.set_title("Tail exponent comparison across instruments")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, asset_dir / "alpha_by_instrument.png")


def plot_xmin_selection(results: list[dict[str, Any]], asset_dir: Path) -> str:
    cols = 2
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, result in zip(axes, results):
        scan = pd.DataFrame(result["modern"]["xmin_scan"])
        meta = result["meta"]
        ax.plot(scan["xmin"], scan["ks"], color="#7c3aed", linewidth=1.8)
        chosen_xmin = result["modern"]["xmin"]
        chosen_ks = result["modern"]["ks"]
        ax.scatter([chosen_xmin], [chosen_ks], color="#ef4444", s=50, zorder=3)
        ax.set_title(f"{meta['instrument']} {meta['scale']}")
        ax.set_xlabel("Candidate xmin")
        ax.set_ylabel("KS distance")
        ax.grid(True, linestyle=":", alpha=0.3)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.suptitle("Automatic xmin selection by KS minimization", fontsize=15, y=1.02)
    fig.tight_layout()
    return save_figure(fig, asset_dir / "xmin_selection.png")


def plot_bootstrap_ci(results: list[dict[str, Any]], asset_dir: Path) -> str:
    ordered = sorted(results, key=lambda item: (item["meta"]["instrument"], format_scale_for_sort(item["meta"]["scale"])))
    labels = [f"{item['meta']['instrument']} {item['meta']['scale']}" for item in ordered]
    alpha = np.array([item["modern"]["alpha_ccdf"] for item in ordered], dtype=float)
    ci_low = np.array([item["bootstrap"]["ci_low"] for item in ordered], dtype=float)
    ci_high = np.array([item["bootstrap"]["ci_high"] for item in ordered], dtype=float)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 4.5 + 0.6 * len(labels)))
    ax.errorbar(
        alpha,
        y,
        xerr=np.vstack([alpha - ci_low, ci_high - alpha]),
        fmt="o",
        color="#1d4ed8",
        capsize=5,
        linewidth=2,
    )
    ax.axvline(3.0, color="#111827", linestyle="--", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Modern alpha (CCDF)")
    ax.set_ylabel("Dataset")
    ax.set_title("Bootstrap 95% confidence interval of alpha")
    ax.grid(True, axis="x", linestyle=":", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, asset_dir / "bootstrap_ci.png")


def format_number(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return ""
    return f"{float(value):.{digits}f}"


def dataframe_to_html(df: pd.DataFrame, digits: int = 4) -> str:
    rendered = df.copy()
    for column in rendered.columns:
        if pd.api.types.is_numeric_dtype(rendered[column]):
            rendered[column] = rendered[column].map(lambda x: format_number(x, digits=digits) if pd.notna(x) else "")
    return rendered.to_html(index=False, classes="report-table", border=0, justify="left", escape=False)


def build_report_tables(results: list[dict[str, Any]], pooled_classic: list[dict[str, Any]]) -> dict[str, str]:
    summary_rows = []
    classic_rows = []
    modern_rows = []
    comparison_rows = []
    xmin_rows = []
    for result in results:
        meta = result["meta"]
        summary_rows.append(
            {
                "Instrument": meta["instrument"],
                "Scale": meta["scale"],
                "Rows": meta["rows"],
                "Returns": meta["returns_count"],
                "Negative tail sample": meta["negative_count"],
                "Mean(r)": meta["mean_return"],
                "Std(r)": meta["std_return"],
                "Tail max": meta["tail_max"],
                "Range start": meta["start_time"],
                "Range end": meta["end_time"],
            }
        )
        classic_rows.append(
            {
                "Instrument": meta["instrument"],
                "Scale": meta["scale"],
                "xmin": result["classic"]["xmin"],
                "Tail n": result["classic"]["n_tail"],
                "alpha CCDF": result["classic"]["alpha_ccdf"],
                "alpha PDF": result["classic"]["alpha_pdf"],
                "R^2": result["classic"]["r_squared"],
            }
        )
        modern_rows.append(
            {
                "Instrument": meta["instrument"],
                "Scale": meta["scale"],
                "Selected xmin": result["modern"]["xmin"],
                "Tail n": result["modern"]["n_tail"],
                "alpha CCDF": result["modern"]["alpha_ccdf"],
                "alpha PDF": result["modern"]["alpha_pdf"],
                "KS": result["modern"]["ks"],
                "95% CI low": result["bootstrap"]["ci_low"],
                "95% CI high": result["bootstrap"]["ci_high"],
                "GOF p-value": result["gof"]["p_value"],
            }
        )
        xmin_rows.append(
            {
                "Instrument": meta["instrument"],
                "Scale": meta["scale"],
                "Selected xmin": result["modern"]["xmin"],
                "Min tail rule": result["modern"]["min_tail_rule"],
                "Tail n": result["modern"]["n_tail"],
                "KS": result["modern"]["ks"],
            }
        )
        for row in result["comparisons"]:
            comparison_rows.append(
                {
                    "Instrument": row["instrument"],
                    "Scale": row["scale"],
                    "Model": row["model"],
                    "Log-likelihood": row["loglik"],
                    "AIC": row["aic"],
                    "KS": row["ks"],
                    "LLR (PL-alt)": row["llr_vs_power_law"],
                    "Vuong p-value": row["p_value"],
                }
            )

    tables = {
        "summary": dataframe_to_html(pd.DataFrame(summary_rows), digits=6),
        "classic": dataframe_to_html(pd.DataFrame(classic_rows), digits=4),
        "modern": dataframe_to_html(pd.DataFrame(modern_rows), digits=4),
        "comparisons": dataframe_to_html(pd.DataFrame(comparison_rows), digits=4),
        "xmin": dataframe_to_html(pd.DataFrame(xmin_rows), digits=4),
    }
    if pooled_classic:
        pooled_rows = []
        for item in pooled_classic:
            pooled_rows.append(
                {
                    "Scale": item["scale"],
                    "Tail n": item["n_tail"],
                    "alpha CCDF": item["alpha_ccdf"],
                    "alpha PDF": item["alpha_pdf"],
                    "R^2": item["r_squared"],
                }
            )
        tables["pooled_classic"] = dataframe_to_html(pd.DataFrame(pooled_rows), digits=4)
    else:
        tables["pooled_classic"] = ""
    return tables


def build_conclusions(results: list[dict[str, Any]]) -> list[str]:
    statements: list[str] = []
    for result in results:
        meta = result["meta"]
        classic_alpha = result["classic"]["alpha_ccdf"]
        modern_alpha = result["modern"]["alpha_ccdf"]
        ci_low = result["bootstrap"]["ci_low"]
        ci_high = result["bootstrap"]["ci_high"]
        gof_p = result["gof"]["p_value"]
        if ci_low <= 3.0 <= ci_high:
            cubic_text = "3 落在现代估计的 95% bootstrap 区间内。"
        else:
            cubic_text = "3 没有落在现代估计的 95% bootstrap 区间内。"
        if gof_p >= 0.1:
            gof_text = "幂律假设通过了较宽松的 goodness-of-fit 检查。"
        elif gof_p >= 0.05:
            gof_text = "幂律假设处在边界区域。"
        else:
            gof_text = "幂律假设的 goodness-of-fit 支持力度较弱。"
        statements.append(
            f"{meta['instrument']} {meta['scale']} 的经典 CCDF 指数为 {classic_alpha:.2f}，现代 CCDF 指数为 {modern_alpha:.2f}，区间为 [{ci_low:.2f}, {ci_high:.2f}]；{cubic_text}{gof_text}"
        )
    return statements


def render_html(
    results: list[dict[str, Any]],
    pooled_classic: list[dict[str, Any]],
    figures: dict[str, str],
    tables: dict[str, str],
) -> str:
    conclusions = build_conclusions(results)
    only_one_instrument = len({item["meta"]["instrument"] for item in results}) == 1

    template = Template(
        r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Negative Tail Power Law Report</title>
  <style>
    :root { --bg: #f6f7fb; --card: #ffffff; --ink: #172033; --muted: #5b6474; --line: #d9dfeb; --accent: #1d4ed8; --accent2: #ef4444; --accent3: #7c3aed; --shadow: 0 14px 40px rgba(23, 32, 51, 0.10); }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; background: radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 28%), radial-gradient(circle at top right, rgba(124, 58, 237, 0.09), transparent 26%), linear-gradient(180deg, #fbfcff 0%, #f2f4fa 100%); color: var(--ink); line-height: 1.75; }
    .page { width: min(1180px, calc(100vw - 32px)); margin: 28px auto 48px; }
    .hero { background: linear-gradient(135deg, rgba(29, 78, 216, 0.94), rgba(12, 74, 110, 0.90)); color: #fff; padding: 34px 36px; border-radius: 28px; box-shadow: var(--shadow); position: relative; overflow: hidden; }
    .hero::after { content: ""; position: absolute; inset: auto -80px -120px auto; width: 260px; height: 260px; border-radius: 50%; background: rgba(255, 255, 255, 0.12); filter: blur(4px); }
    .hero h1 { margin: 0 0 10px; font-size: 34px; line-height: 1.25; letter-spacing: 0.01em; }
    .hero p { margin: 8px 0; max-width: 920px; color: rgba(255, 255, 255, 0.92); font-size: 16px; }
    .hero-meta { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .pill { padding: 8px 14px; border-radius: 999px; background: rgba(255, 255, 255, 0.14); border: 1px solid rgba(255, 255, 255, 0.18); font-size: 14px; }
    .grid { display: grid; gap: 18px; margin-top: 18px; }
    .grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .card { background: var(--card); border: 1px solid rgba(217, 223, 235, 0.9); border-radius: 24px; box-shadow: var(--shadow); padding: 24px 26px; }
    h2 { margin: 0 0 12px; font-size: 24px; line-height: 1.3; }
    h3 { margin: 0 0 10px; font-size: 18px; line-height: 1.35; }
    p { margin: 10px 0; color: var(--ink); }
    .muted { color: var(--muted); }
    .formula-box { display: grid; gap: 10px; margin-top: 14px; }
    .formula { background: #f8faff; border: 1px solid #d7e3ff; border-radius: 18px; padding: 14px 16px; font-family: "Consolas", "SFMono-Regular", monospace; font-size: 15px; color: #173067; }
    .note { padding: 14px 16px; border-radius: 18px; background: #fff8ec; border: 1px solid #f8db9f; color: #7a4b00; }
    .result-list { margin: 0; padding-left: 20px; }
    .result-list li { margin: 10px 0; }
    .figure { margin-top: 14px; }
    .figure img { display: block; width: 100%; border-radius: 18px; border: 1px solid var(--line); background: #fff; }
    .caption { margin-top: 8px; color: var(--muted); font-size: 14px; }
    .report-table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 14px; overflow: hidden; border-radius: 18px; }
    .report-table thead th { background: #edf4ff; color: #1f356d; text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--line); }
    .report-table tbody td { padding: 10px 12px; border-bottom: 1px solid #e8ecf5; vertical-align: top; }
    .report-table tbody tr:nth-child(even) { background: #fbfcff; }
    .section-lead { font-size: 15px; color: var(--muted); }
    .footer { margin-top: 18px; color: var(--muted); font-size: 14px; text-align: center; }
    @media (max-width: 900px) { .grid.two { grid-template-columns: 1fr; } .hero { padding: 28px 24px; } .card { padding: 20px 18px; } }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>标准化收益率负尾幂律复现报告</h1>
      <p>报告基于项目现有样本数据，围绕负尾经验 CCDF、固定阈值经典复现、自动阈值现代统计检验，以及模型比较四个部分展开。报告中的尾指数主记号采用 CCDF 形式：P(X &gt; x) ∝ x<sup>-α</sup>。</p>
      <div class="hero-meta">
        <div class="pill">Data files: {{ results|length }}</div>
        <div class="pill">Classic xmin = {{ classic_xmin }}</div>
        <div class="pill">Bootstrap CI reps = {{ ci_reps }}</div>
        <div class="pill">GOF bootstrap reps = {{ gof_reps }}</div>
      </div>
    </section>
    <div class="grid two">
      <section class="card">
        <h2>研究问题</h2>
        <p class="section-lead">目标分为两个层次：一个层次面向经典文献复现，另一个层次面向更严格的统计验证。</p>
        <p>我们检验标准化收益率负尾是否近似服从幂律分布，并检验 CCDF 尾指数 α 是否接近 3。若按 PDF 记号书写，指数会多出 1，因此对应关系是 β = α + 1。</p>
        <div class="formula-box">
          <div class="formula">r_t = ln(P_t) - ln(P_{t-1})</div>
          <div class="formula">g_t = (r_t - mean(r)) / std(r)</div>
          <div class="formula">x_t = -g_t, with g_t &lt; 0</div>
          <div class="formula">CCDF: P(X &gt; x) ∝ x^(-α) ; PDF: f(x) ∝ x^(-(α+1))</div>
        </div>
      </section>
      <section class="card">
        <h2>数据摘要</h2>
        <p class="section-lead">每个文件单独计算对数收益率、全样本均值和标准差，再提取负标准化收益率并转成正的尾部样本。</p>
        {{ tables.summary | safe }}
      </section>
    </div>
    <section class="card">
      <h2>经典复现</h2>
      <p>经典版本固定使用 xmin = 2，只观察 2 sigma 以上的负尾部。经验 CCDF 在 log-log 坐标中近似呈线性时，常被视为幂律尾部的视觉证据。这里的 α 指 CCDF 指数，PDF 指数等于 α + 1。</p>
      {{ tables.classic | safe }}
      {% if tables.pooled_classic %}
      <h3>Classical pooled result</h3>
      {{ tables.pooled_classic | safe }}
      {% else %}
      <div class="note">当前数据集中只有一个品种，因此“各个品种分别标准化后再合并”的 pooled 版本没有额外信息量，报告里保留单品种结果。</div>
      {% endif %}
      <div class="figure"><img src="data:image/png;base64,{{ figures.classic_ccdf }}" alt="Classic CCDF"><div class="caption">图 1. 各个数据集的负尾经验 CCDF。红色虚线对应经典阈值 xmin = 2。</div></div>
      <div class="figure"><img src="data:image/png;base64,{{ figures.fitted_loglog }}" alt="Fitted loglog"><div class="caption">图 2. 经典阈值拟合与现代自动阈值拟合的 log-log 对照图。</div></div>
    </section>
    <div class="grid two">
      <section class="card">
        <h2>现代统计检验</h2>
        <p>现代版本沿用 Clauset-Shalizi-Newman 一类思路：先自动选择 xmin，再对尾部样本做极大似然估计，并用 KS 距离评估拟合偏差。bootstrap 给出 α 的区间估计，半参数 bootstrap 给出 goodness-of-fit p 值。</p>
        {{ tables.modern | safe }}
        <div class="note">GOF p-value 越大，幂律模型与数据相容的程度越高。数值接近 0.1 或 0.05 时，通常需要更谨慎地解读结论。</div>
      </section>
      <section class="card">
        <h2>自动 xmin 结果</h2>
        <p>自动阈值通过扫描候选 xmin 并最小化 KS 距离获得。候选点数量较多时，程序会做均匀抽样，以免无谓地拉长计算时间。</p>
        {{ tables.xmin | safe }}
        <div class="figure"><img src="data:image/png;base64,{{ figures.xmin_selection }}" alt="xmin selection"><div class="caption">图 3. KS 距离随 xmin 变化的扫描结果，红点是最终选中的 xmin。</div></div>
      </section>
    </div>
    <section class="card">
      <h2>模型比较</h2>
      <p>报告将 power law 与 exponential、lognormal、truncated power law 放在同一条尾部样本上比较。对于 exponential 与 lognormal，表中给出相对幂律模型的对数似然差与 Vuong p 值；对于 truncated power law，表中重点呈现 log-likelihood 与 AIC。</p>
      {{ tables.comparisons | safe }}
      <p class="muted">LLR (PL-alt) 为正时，幂律模型的对数似然更高；AIC 越小，模型在拟合与参数复杂度之间的平衡越好。</p>
    </section>
    <div class="grid two">
      <section class="card">
        <h2>尾指数对比</h2>
        <p>这部分集中展示 α 的横向差异。图中基准线取 α = 3，也就是本报告采用的 inverse cubic law 基准。</p>
        <div class="figure"><img src="data:image/png;base64,{{ figures.alpha_by_scale }}" alt="alpha by scale"><div class="caption">图 4. 不同时间尺度下的 α 对比。蓝线带有 bootstrap 误差条。</div></div>
        <div class="figure"><img src="data:image/png;base64,{{ figures.alpha_by_instrument }}" alt="alpha by instrument"><div class="caption">图 5. 不同品种下的 α 对比。当前样本只有 XAGUSD，因此图中只有一个品种。</div></div>
      </section>
      <section class="card">
        <h2>Bootstrap 区间</h2>
        <p>现代 α 的 95% bootstrap 区间直接展示了估计不确定性。区间若覆盖 3，则 inverse cubic law 的基准与样本没有明显冲突。</p>
        <div class="figure"><img src="data:image/png;base64,{{ figures.bootstrap_ci }}" alt="bootstrap ci"><div class="caption">图 6. 各个数据集现代 α 的 95% bootstrap 区间。</div></div>
      </section>
    </div>
    <section class="card">
      <h2>结论</h2>
      <ul class="result-list">{% for line in conclusions %}<li>{{ line }}</li>{% endfor %}</ul>
      <div class="note">在本报告中，inverse cubic law 的“3”指 CCDF 指数 α。若切换到 PDF 记号，尾指数应写成 β = α + 1，因此对应的 PDF 指数基准是 4。</div>
    </section>
    <section class="card">
      <h2>方法细节</h2>
      <p>经典版本的 α 来自固定 xmin = 2 的 log-log 线性拟合。现代版本的 α 来自自动 xmin 下的连续型 Pareto 极大似然估计。两者服务于不同的问题：经典版本强调与旧文献的直观对照，现代版本强调统计诊断与模型辨别。</p>
      <p>若后续补充更多品种或更高频数据，脚本会自动纳入新的 `sample_*.csv` 文件，并继续输出 pooled 结果与更新后的比较图。</p>
    </section>
    <div class="footer">Generated from {{ results|length }} dataset(s).{% if only_one_instrument %} Current instrument count: 1.{% endif %}</div>
  </div>
</body>
</html>
"""
    )
    return template.render(
        results=results,
        pooled_classic=pooled_classic,
        figures=figures,
        tables=tables,
        conclusions=conclusions,
        classic_xmin=CLASSIC_XMIN,
        ci_reps=CI_BOOTSTRAP_REPS,
        gof_reps=GOF_BOOTSTRAP_REPS,
        only_one_instrument=only_one_instrument,
    )


def empirical_ccdf_details(sample: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.sort(np.asarray(sample, dtype=float))
    if values.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty
    unique, counts = np.unique(values, return_counts=True)
    survivors = np.cumsum(counts[::-1])[::-1]
    ccdf = survivors / values.size
    return unique, ccdf, survivors, counts


def format_count(value: Any) -> str:
    return f"{int(value):,}"


def format_value(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return ""
    return f"{float(value):.{digits}f}"


def model_label(model: str) -> str:
    labels = {
        "power_law": "幂律 (power law)",
        "exponential": "指数分布 (exponential)",
        "lognormal": "对数正态 (lognormal)",
        "truncated_power_law": "截断幂律 (truncated power law)",
    }
    return labels.get(model, model)


def comparison_note(delta_aic: float, model: str) -> str:
    if model == "power_law":
        return "基准模型"
    if delta_aic <= -6:
        return "替代模型优势明显"
    if delta_aic <= -2:
        return "替代模型略占优势"
    if delta_aic < 2:
        return "差异接近"
    if delta_aic < 6:
        return "幂律略占优势"
    return "幂律优势明显"


def ordered_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=lambda item: (item["meta"]["instrument"], format_scale_for_sort(item["meta"]["scale"])))


def plotly_fragment(fig: go.Figure, include_js: bool = False) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs="inline" if include_js else False,
        config={
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": False,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        },
    )


def apply_plot_theme(fig: go.Figure, title_text: str, height: int, legend_y: float | None = None) -> None:
    legend: dict[str, Any] = {
        "orientation": "h",
        "x": 0.0,
        "bgcolor": "rgba(255, 255, 255, 0.92)",
        "bordercolor": "rgba(191, 219, 254, 0.92)",
        "borderwidth": 1,
        "font": {"size": 13, "color": "#173067"},
    }
    if legend_y is not None:
        legend["y"] = legend_y
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 34, "r": 22, "t": 78, "b": 46},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"family": '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif', "size": 14, "color": "#173067"},
        title={"text": title_text, "x": 0.5, "xanchor": "center", "font": {"size": 26, "color": "#173067"}},
        legend=legend,
        hoverlabel={
            "bgcolor": "rgba(255, 255, 255, 0.97)",
            "bordercolor": "#bfdbfe",
            "font": {"family": '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif', "size": 13, "color": "#173067"},
        },
        hovermode="closest",
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(96, 165, 250, 0.18)",
        zeroline=False,
        linecolor="rgba(29, 78, 216, 0.24)",
        tickcolor="rgba(29, 78, 216, 0.24)",
        ticks="outside",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(96, 165, 250, 0.18)",
        zeroline=False,
        linecolor="rgba(29, 78, 216, 0.24)",
        tickcolor="rgba(29, 78, 216, 0.24)",
        ticks="outside",
    )


def select_focus_result(results: list[dict[str, Any]], preferred_scale: str = "2h") -> dict[str, Any]:
    items = ordered_results(results)
    for item in items:
        if item["meta"]["scale"].lower() == preferred_scale.lower():
            return item
    return items[0]


def select_comparison_results(results: list[dict[str, Any]], preferred_scale: str = "2h") -> list[dict[str, Any]]:
    focus = select_focus_result(results, preferred_scale=preferred_scale)
    return [item for item in ordered_results(results) if item["meta"]["dataset_id"] != focus["meta"]["dataset_id"]]


def make_modern_tail_fit_figure(results: list[dict[str, Any]]) -> go.Figure:
    items = [select_focus_result(results, preferred_scale="2h")]
    fig = make_subplots(
        rows=1,
        cols=len(items),
        subplot_titles=[f"{item['meta']['instrument']} {item['meta']['scale']}" for item in items],
        horizontal_spacing=0.08,
    )
    for col, result in enumerate(items, start=1):
        meta = result["meta"]
        modern = result["modern"]
        observations = pd.DataFrame(modern["tail_observations"])
        x = observations["tail_x"].to_numpy(dtype=float)
        y = observations["conditional_ccdf"].to_numpy(dtype=float)
        custom = np.column_stack(
            [
                observations["time"].to_numpy(),
                observations["simple_return_pct"].to_numpy(dtype=float),
                observations["log_return"].to_numpy(dtype=float),
                observations["standardized_return"].to_numpy(dtype=float),
                observations["survivors"].to_numpy(dtype=float),
                np.full(len(observations), modern["n_tail"], dtype=float),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="经验尾部点",
                showlegend=col == 1,
                marker={"size": 7.2, "color": "#2f9e91", "opacity": 0.76, "line": {"width": 0.7, "color": "#c7f5ee"}},
                customdata=custom,
                hovertemplate=(
                    f"{meta['instrument']} {meta['scale']}<br>"
                    "时间 (time): %{customdata[0]}<br>"
                    "尾部幅度 x: %{x:.4f}<br>"
                    "条件 CCDF: %{y:.4f}<br>"
                    "真实跌幅: %{customdata[1]:.4f}%<br>"
                    "对数收益率 (log return): %{customdata[2]:.6f}<br>"
                    "标准化收益率 g: %{customdata[3]:.4f}<br>"
                    "右侧剩余样本: %{customdata[4]:.0f} / %{customdata[5]:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        grid = np.geomspace(max(modern["xmin"], x.min()), x.max() * 1.04, 240)
        fitted = np.power(grid / modern["xmin"], -modern["alpha_ccdf"])
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=fitted,
                mode="lines",
                name="幂律拟合",
                showlegend=col == 1,
                line={"color": "#1d4ed8", "width": 2.1},
                hovertemplate=(
                    f"{meta['instrument']} {meta['scale']}<br>"
                    "拟合 x: %{x:.4f}<br>"
                    "拟合 CCDF: %{y:.4f}<br>"
                    f"xmin = {modern['xmin']:.4f}<br>"
                    f"alpha = {modern['alpha_ccdf']:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.add_vline(
            x=modern["xmin"],
            line_width=1.8,
            line_dash="dash",
            line_color="#60a5fa",
            row=1,
            col=col,
        )
        fig.add_annotation(
            x=modern["xmin"],
            y=1.02,
            xref=f"x{col}" if col > 1 else "x",
            yref=f"y{col} domain" if col > 1 else "y domain",
            text=f"xmin={modern['xmin']:.3f}",
            showarrow=False,
            font={"size": 11, "color": "#1d4ed8"},
        )
        fig.update_xaxes(type="log", title_text="负尾幅度 x", row=1, col=col)
        fig.update_yaxes(type="log", title_text="条件 CCDF" if col == 1 else "", row=1, col=col)
    focus = items[0]
    apply_plot_theme(
        fig,
        f"{focus['meta']['instrument']} {focus['meta']['scale']}：负尾经验 CCDF 与现代幂律拟合",
        height=560,
        legend_y=1.14,
    )
    return fig


def make_tail_comparison_figure(results: list[dict[str, Any]]) -> go.Figure:
    items = select_comparison_results(results, preferred_scale="2h")
    compare_title = " 与 ".join(item["meta"]["scale"] for item in items) + " 对照图"
    fig = make_subplots(
        rows=1,
        cols=len(items),
        subplot_titles=[f"{item['meta']['instrument']} {item['meta']['scale']}" for item in items],
        horizontal_spacing=0.08,
    )
    for col, result in enumerate(items, start=1):
        meta = result["meta"]
        modern = result["modern"]
        observations = pd.DataFrame(modern["tail_observations"])
        x = observations["tail_x"].to_numpy(dtype=float)
        y = observations["conditional_ccdf"].to_numpy(dtype=float)
        custom = np.column_stack(
            [
                observations["time"].to_numpy(),
                observations["simple_return_pct"].to_numpy(dtype=float),
                observations["log_return"].to_numpy(dtype=float),
                observations["standardized_return"].to_numpy(dtype=float),
                observations["survivors"].to_numpy(dtype=float),
                np.full(len(observations), modern["n_tail"], dtype=float),
            ]
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="经验尾部点",
                showlegend=col == 1,
                marker={"size": 6.5, "color": "#2f9e91", "opacity": 0.76, "line": {"width": 0.7, "color": "#c7f5ee"}},
                customdata=custom,
                hovertemplate=(
                    f"{meta['instrument']} {meta['scale']}<br>"
                    "时间 (time): %{customdata[0]}<br>"
                    "尾部幅度 x: %{x:.4f}<br>"
                    "条件 CCDF: %{y:.4f}<br>"
                    "真实跌幅: %{customdata[1]:.4f}%<br>"
                    "对数收益率 (log return): %{customdata[2]:.6f}<br>"
                    "标准化收益率 g: %{customdata[3]:.4f}<br>"
                    "右侧剩余样本: %{customdata[4]:.0f} / %{customdata[5]:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        grid = np.geomspace(max(modern["xmin"], x.min()), x.max() * 1.04, 240)
        fitted = np.power(grid / modern["xmin"], -modern["alpha_ccdf"])
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=fitted,
                mode="lines",
                name="幂律拟合",
                showlegend=col == 1,
                line={"color": "#1d4ed8", "width": 2.0},
                hovertemplate=(
                    f"{meta['instrument']} {meta['scale']}<br>"
                    "拟合 x: %{x:.4f}<br>"
                    "拟合 CCDF: %{y:.4f}<br>"
                    f"xmin = {modern['xmin']:.4f}<br>"
                    f"alpha = {modern['alpha_ccdf']:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.add_vline(
            x=modern["xmin"],
            line_width=1.5,
            line_dash="dash",
            line_color="#60a5fa",
            row=1,
            col=col,
        )
        fig.add_annotation(
            x=modern["xmin"],
            y=1.02,
            xref=f"x{col}" if col > 1 else "x",
            yref=f"y{col} domain" if col > 1 else "y domain",
            text=f"xmin={modern['xmin']:.3f}",
            showarrow=False,
            font={"size": 10, "color": "#1d4ed8"},
        )
        fig.update_xaxes(type="log", title_text="负尾幅度 x", row=1, col=col)
        fig.update_yaxes(type="log", title_text="条件 CCDF" if col == 1 else "", row=1, col=col)
    apply_plot_theme(fig, compare_title, height=390, legend_y=1.13)
    return fig


def make_xmin_scan_figure(results: list[dict[str, Any]]) -> go.Figure:
    items = ordered_results(results)
    fig = make_subplots(
        rows=1,
        cols=len(items),
        subplot_titles=[f"{item['meta']['instrument']} {item['meta']['scale']}" for item in items],
        horizontal_spacing=0.08,
    )
    for col, result in enumerate(items, start=1):
        scan = pd.DataFrame(result["modern"]["xmin_scan"])
        custom = np.column_stack([scan["n_tail"], scan["alpha_ccdf"], scan["alpha_pdf"]])
        fig.add_trace(
            go.Scatter(
                x=scan["xmin"],
                y=scan["ks"],
                mode="lines+markers",
                name="候选 xmin",
                showlegend=col == 1,
                line={"color": "#60a5fa", "width": 2.3},
                marker={"size": 7, "color": "#93c5fd", "line": {"width": 0.8, "color": "#2563eb"}},
                customdata=custom,
                hovertemplate=(
                    "xmin = %{x:.4f}<br>"
                    "KS 距离: %{y:.4f}<br>"
                    "尾部样本数 Tail n: %{customdata[0]:.0f}<br>"
                    "CCDF 指数 alpha: %{customdata[1]:.4f}<br>"
                    "PDF 指数: %{customdata[2]:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[result["modern"]["xmin"]],
                y=[result["modern"]["ks"]],
                mode="markers",
                name="选中的 xmin",
                showlegend=col == 1,
                marker={"size": 12, "color": "#1d4ed8", "symbol": "diamond", "line": {"width": 1.0, "color": "#dbeafe"}},
                hovertemplate=(
                    f"{result['meta']['instrument']} {result['meta']['scale']}<br>"
                    f"选中的 xmin = {result['modern']['xmin']:.4f}<br>"
                    f"KS 距离 = {result['modern']['ks']:.4f}<br>"
                    f"尾部样本数 = {result['modern']['n_tail']}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="候选 xmin", row=1, col=col)
        fig.update_yaxes(title_text="KS 距离" if col == 1 else "", row=1, col=col)
    apply_plot_theme(fig, "xmin 自动选择与 KS 路径", height=410, legend_y=1.15)
    return fig


def make_alpha_overview_figure(results: list[dict[str, Any]]) -> go.Figure:
    items = ordered_results(results)
    labels = [f"{item['meta']['instrument']} {item['meta']['scale']}" for item in items]
    alpha = np.array([item["modern"]["alpha_ccdf"] for item in items], dtype=float)
    ci_low = np.array([item["bootstrap"]["ci_low"] for item in items], dtype=float)
    ci_high = np.array([item["bootstrap"]["ci_high"] for item in items], dtype=float)
    custom = np.column_stack(
        [
            np.array([item["bootstrap"]["ci_low"] for item in items], dtype=float),
            np.array([item["bootstrap"]["ci_high"] for item in items], dtype=float),
            np.array([item["modern"]["xmin"] for item in items], dtype=float),
            np.array([item["modern"]["n_tail"] for item in items], dtype=float),
            np.array([item["modern"]["ks"] for item in items], dtype=float),
            np.array([item["gof"]["p_value"] for item in items], dtype=float),
        ]
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=alpha,
            mode="markers+lines",
            marker={"size": 11, "color": "#2563eb", "line": {"width": 1.0, "color": "#dbeafe"}},
            line={"width": 2.6, "color": "#2563eb"},
            error_y={"type": "data", "symmetric": False, "array": ci_high - alpha, "arrayminus": alpha - ci_low, "visible": True},
            customdata=custom,
            hovertemplate=(
                "数据集: %{x}<br>"
                "CCDF 指数 alpha = %{y:.4f}<br>"
                "95% 置信区间 = [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>"
                "xmin = %{customdata[2]:.4f}<br>"
                "尾部样本数 Tail n = %{customdata[3]:.0f}<br>"
                "KS 距离 = %{customdata[4]:.4f}<br>"
                "GOF p 值 = %{customdata[5]:.4f}<extra></extra>"
            ),
            name="alpha 与 95% 区间",
        )
    )
    fig.add_hline(y=3.0, line_dash="dash", line_color="#60a5fa", line_width=1.8)
    fig.add_annotation(
        x=labels[-1],
        y=3.0,
        text="inverse cubic 基准：alpha = 3",
        showarrow=False,
        yshift=12,
        font={"size": 11, "color": "#1d4ed8"},
    )
    fig.update_yaxes(title_text="CCDF 尾指数 alpha")
    fig.update_xaxes(title_text="数据集")
    apply_plot_theme(fig, "alpha 与 bootstrap 区间", height=390, legend_y=1.13)
    return fig


def make_model_comparison_figure(results: list[dict[str, Any]]) -> go.Figure:
    items = ordered_results(results)
    fig = make_subplots(
        rows=1,
        cols=len(items),
        subplot_titles=[f"{item['meta']['instrument']} {item['meta']['scale']}" for item in items],
        horizontal_spacing=0.08,
    )
    colors = {
        "power_law": "#1d4ed8",
        "exponential": "#93c5fd",
        "lognormal": "#60a5fa",
        "truncated_power_law": "#2563eb",
    }
    for col, result in enumerate(items, start=1):
        rows = result["comparisons"]
        baseline_aic = next(row["aic"] for row in rows if row["model"] == "power_law")
        x = [model_label(row["model"]) for row in rows]
        y = [row["aic"] - baseline_aic for row in rows]
        custom = np.column_stack(
            [
                np.array([row["loglik"] for row in rows], dtype=float),
                np.array([row["aic"] for row in rows], dtype=float),
                np.array([row["ks"] for row in rows], dtype=float),
                np.array([row["llr_vs_power_law"] for row in rows], dtype=float),
                np.array([row["p_value"] for row in rows], dtype=float),
            ]
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=y,
                name="Delta AIC",
                showlegend=False,
                marker={"color": [colors[row["model"]] for row in rows]},
                customdata=custom,
                hovertemplate=(
                    "模型: %{x}<br>"
                    "delta AIC = %{y:.4f}<br>"
                    "对数似然 loglik = %{customdata[0]:.4f}<br>"
                    "AIC = %{customdata[1]:.4f}<br>"
                    "KS 距离 = %{customdata[2]:.4f}<br>"
                    "LLR(PL-alt) = %{customdata[3]:.4f}<br>"
                    "Vuong p 值 = %{customdata[4]:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="模型", row=1, col=col)
        fig.update_yaxes(title_text="Delta AIC" if col == 1 else "", row=1, col=col)
    fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8", line_width=1.5)
    apply_plot_theme(fig, "模型比较：相对幂律的 Delta AIC", height=430, legend_y=1.12)
    return fig


def metric_html(label: str, value: str, hint: str) -> str:
    return f'<div class="metric-item"><div class="metric-label">{escape(label)}</div><div class="metric-value">{value}</div><div class="metric-hint">{escape(hint)}</div></div>'


def build_dataset_cards_html(results: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for result in ordered_results(results):
        meta = result["meta"]
        modern = result["modern"]
        bootstrap = result["bootstrap"]
        gof = result["gof"]
        tail_share = modern["n_tail"] / meta["negative_count"] if meta["negative_count"] else np.nan
        card = f"""
        <article class="dataset-card">
          <div class="dataset-top">
            <div>
              <h3>{escape(meta['instrument'])} <span>{escape(meta['scale'])}</span></h3>
              <p class="dataset-range">{escape(meta['start_time'])} 至 {escape(meta['end_time'])}</p>
            </div>
            <div class="dataset-badge">现代检验</div>
          </div>
          <p class="dataset-summary">选中的 xmin = {format_value(modern['xmin'], 4)}，尾部样本数 = {format_count(modern['n_tail'])}，CCDF 尾指数 alpha = {format_value(modern['alpha_ccdf'], 4)}。</p>
          <div class="metric-grid">
            {metric_html("样本行数", format_count(meta["rows"]), "原始价格记录个数")}
            {metric_html("收益率个数", format_count(meta["returns_count"]), "对数收益率 r 的个数")}
            {metric_html("负收益样本", format_count(meta["negative_count"]), "全部 g < 0 的标准化收益率")}
            {metric_html("尾部样本数", format_count(modern["n_tail"]), "满足 x >= xmin 的样本个数")}
            {metric_html("尾部占比", format_value(tail_share, 4), "Tail n / 负收益样本")}
            {metric_html("CCDF 指数 alpha", format_value(modern["alpha_ccdf"], 4), "P(X>x) ~ x^-alpha")}
            {metric_html("PDF 指数", format_value(modern["alpha_pdf"], 4), "alpha PDF = alpha CCDF + 1")}
            {metric_html("95% 置信区间", f"[{format_value(bootstrap['ci_low'], 3)}, {format_value(bootstrap['ci_high'], 3)}]", "bootstrap 区间估计")}
            {metric_html("KS 距离", format_value(modern["ks"], 4), "经验分布与拟合分布的最大距离")}
            {metric_html("GOF p 值", format_value(gof["p_value"], 4), "半参数 bootstrap 拟合优度")}
            {metric_html("mean(r)", format_value(meta["mean_return"], 6), "全样本对数收益率均值")}
            {metric_html("std(r)", format_value(meta["std_return"], 6), "全样本对数收益率标准差")}
            {metric_html("最大尾部 x", format_value(meta["tail_max"], 4), "观测到的最大负尾幅度")}
          </div>
        </article>
        """
        cards.append(card)
    return "\n".join(cards)


def build_metric_glossary_html() -> str:
    items = [
        ("xmin", "幂律尾部的起点阈值。报告会扫描候选值，并保留 KS 距离最小的 xmin。"),
        ("Tail n", "满足 x >= xmin 的样本个数。它就是尾部拟合真正使用的样本数。"),
        ("alpha CCDF", "报告的主指数，满足 P(X>x) ~ x^-alpha。alpha 越小，尾部衰减越慢，极端事件的概率质量越厚。"),
        ("alpha PDF", "若 PDF 写成 f(x) ~ x^-beta，则 beta = alpha + 1。报告会同时列出两种写法，方便和文献口径对照。"),
        ("KS", "经验条件分布与拟合 Pareto 分布之间的最大垂直距离。KS 越小，拟合贴合度越高。"),
        ("95% CI", "对尾部样本重复 bootstrap 抽样以后得到的 alpha 区间估计。区间宽度可以反映估计稳定性。"),
        ("GOF p-value", "半参数 bootstrap 拟合优度检验的 p 值。较大的 p 值表示当前样本与幂律模型之间没有明显冲突。"),
        ("AIC / Delta AIC", "AIC 同时考察拟合优度与参数个数。Delta AIC 以幂律为基准，负值表示替代模型的 AIC 较小。"),
        ("LLR / Vuong p", "LLR 是幂律与替代模型的对数似然差。Vuong p 值常用于幂律与 exponential、lognormal 之间的比较。"),
    ]
    html_parts = []
    for title, body in items:
        html_parts.append(f'<div class="glossary-item"><h4>{escape(title)}</h4><p>{escape(body)}</p></div>')
    return "\n".join(html_parts)


def build_tail_chart_reading_html(results: list[dict[str, Any]]) -> str:
    focus = select_focus_result(results, preferred_scale="2h")
    compare = {item["meta"]["scale"]: item for item in ordered_results(results)}
    one_h = compare.get("1h")
    one_d = compare.get("1d")
    focus_best = min(focus["comparisons"], key=lambda row: row["aic"])
    alpha_summary_parts = [
        f"2h 的 alpha = {focus['modern']['alpha_ccdf']:.3f}",
    ]
    if one_h is not None:
        alpha_summary_parts.append(f"1h = {one_h['modern']['alpha_ccdf']:.3f}")
    if one_d is not None:
        alpha_summary_parts.append(f"1d = {one_d['modern']['alpha_ccdf']:.3f}")
    xmin_summary_parts = [
        f"2h 的 xmin = {focus['modern']['xmin']:.3f}",
    ]
    if one_h is not None:
        xmin_summary_parts.append(f"1h = {one_h['modern']['xmin']:.3f}")
    if one_d is not None:
        xmin_summary_parts.append(f"1d = {one_d['modern']['xmin']:.3f}")

    items = [
        (
            "alpha 的意义",
            "alpha 控制尾部衰减速度。alpha 较小，远端极端负收益的相对频率较高；alpha 较大，尾部收敛得较快。当前样本里，"
            + "，".join(alpha_summary_parts)
            + "。从这组结果看，1h 的尾部收敛速度最快，2h 与 1d 的尾部厚度接近。"
        ),
        (
            "点在线上方或下方",
            "某一段点若连续落在线上方，表示经验 CCDF 高于幂律拟合，样本在这一段的超额概率高于模型给出的水平。某一段点若连续落在线下方，表示经验 CCDF 低于拟合线，模型在这一段给出的尾部概率偏大。若上下摆动主要集中在最右端，有限样本波动通常会参与其中。"
        ),
        (
            "为何主图选 2h",
            f"2h 的尾部样本数 Tail n = {focus['modern']['n_tail']}。这个样本量高于 1h 和 1d，因此图形信息最完整，适合作为主图。下方对照图继续保留 1h 与 1d，用来观察斜率、弯曲方向和 xmin 位置的差异。"
        ),
        (
            "xmin 的图上含义",
            "xmin 是幂律尾部开始的位置。图里所有点都已经位于 x >= xmin 的拟合区间。"
            + "，".join(xmin_summary_parts)
            + "。1h 的 xmin 最深，说明 1h 需要进入较大的标准差区间以后，幂律尾部才显得稳定。"
        ),
        (
            "从当前图形可读出的结论",
            f"若点沿拟合线排布得较整齐，说明幂律近似在该区间内较稳定。2h 主图右端若干点位于拟合线下方，表示最远端极端事件的经验概率低于纯幂律外推。当前 2h 的 AIC 首位模型是 {model_label(focus_best['model'])}，这与右端下弯的图形直觉是一致的。"
        ),
    ]
    return "\n".join(
        f'<div class="glossary-item"><h4>{escape(title)}</h4><p>{escape(body)}</p></div>'
        for title, body in items
    )


def build_model_table_html(results: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for result in ordered_results(results):
        baseline_aic = next(row["aic"] for row in result["comparisons"] if row["model"] == "power_law")
        for row in result["comparisons"]:
            delta_aic = row["aic"] - baseline_aic
            rows.append(
                "<tr>"
                f"<td>{escape(result['meta']['instrument'])}</td>"
                f"<td>{escape(result['meta']['scale'])}</td>"
                f"<td>{escape(model_label(row['model']))}</td>"
                f"<td>{format_value(row['loglik'], 4)}</td>"
                f"<td>{format_value(row['aic'], 4)}</td>"
                f"<td>{format_value(delta_aic, 4)}</td>"
                f"<td>{format_value(row['ks'], 4)}</td>"
                f"<td>{format_value(row['llr_vs_power_law'], 4)}</td>"
                f"<td>{format_value(row['p_value'], 4)}</td>"
                f"<td>{escape(comparison_note(delta_aic, row['model']))}</td>"
                "</tr>"
            )
    return (
        '<div class="table-wrap"><table class="report-table"><thead><tr>'
        "<th>品种</th><th>时间尺度</th><th>模型</th><th>LogLik</th><th>AIC</th><th>Delta AIC</th><th>KS</th><th>LLR</th><th>Vuong p</th><th>解释</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def build_modern_takeaways(results: list[dict[str, Any]]) -> list[str]:
    notes: list[str] = []
    for result in ordered_results(results):
        meta = result["meta"]
        modern = result["modern"]
        bootstrap = result["bootstrap"]
        gof = result["gof"]
        best_alt = min(result["comparisons"], key=lambda row: row["aic"])
        if bootstrap["ci_low"] <= 3.0 <= bootstrap["ci_high"]:
            benchmark_text = "区间覆盖 alpha = 3。"
        else:
            benchmark_text = "区间没有覆盖 alpha = 3。"
        if best_alt["model"] == "power_law":
            model_text = "按 AIC 口径，幂律位于首位。"
        else:
            model_text = f"AIC 最小值出现在 {model_label(best_alt['model'])}。"
        notes.append(
            f"{meta['instrument']} {meta['scale']}：xmin = {modern['xmin']:.4f}，alpha = {modern['alpha_ccdf']:.4f}，95% 区间 = [{bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f}]，GOF p 值 = {gof['p_value']:.4f}。{benchmark_text}{model_text}"
        )
    return notes


def render_modern_report_html(results: list[dict[str, Any]]) -> str:
    focus_tail_fig = plotly_fragment(make_modern_tail_fit_figure(results), include_js=True)
    compare_tail_fig = plotly_fragment(make_tail_comparison_figure(results), include_js=False)
    xmin_fig = plotly_fragment(make_xmin_scan_figure(results), include_js=False)
    alpha_fig = plotly_fragment(make_alpha_overview_figure(results), include_js=False)
    model_fig = plotly_fragment(make_model_comparison_figure(results), include_js=False)
    dataset_cards = build_dataset_cards_html(results)
    glossary_html = build_metric_glossary_html()
    tail_chart_reading_html = build_tail_chart_reading_html(results)
    model_table = build_model_table_html(results)
    takeaways = build_modern_takeaways(results)

    template = Template(
        r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>负尾幂律现代检验报告</title>
  <style>
    :root {
      --bg: #f6f8fc;
      --card: #ffffff;
      --ink: #172033;
      --muted: #5b6474;
      --line: #d9dfeb;
      --accent: #1d4ed8;
      --accent-soft: #60a5fa;
      --accent-faint: #dbeafe;
      --accent-deep: #173067;
      --shadow: 0 14px 40px rgba(23, 32, 51, 0.10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(29, 78, 216, 0.10), transparent 28%),
        radial-gradient(circle at top right, rgba(96, 165, 250, 0.10), transparent 26%),
        linear-gradient(180deg, #fbfcff 0%, #f2f6fd 100%);
      color: var(--ink);
      line-height: 1.75;
    }
    .page { width: min(1240px, calc(100vw - 32px)); margin: 28px auto 48px; }
    .hero {
      background: linear-gradient(135deg, rgba(29, 78, 216, 0.96), rgba(12, 74, 110, 0.92));
      color: #ffffff;
      border-radius: 28px;
      padding: 34px 36px;
      box-shadow: var(--shadow);
      position: relative;
      overflow: hidden;
    }
    .hero::after {
      content: "";
      position: absolute;
      inset: auto -80px -120px auto;
      width: 260px;
      height: 260px;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.12);
      filter: blur(4px);
    }
    .hero h1 { margin: 0 0 10px; font-size: 34px; line-height: 1.25; letter-spacing: 0.01em; }
    .hero p { margin: 8px 0; max-width: 980px; color: rgba(255, 255, 255, 0.92); font-size: 16px; }
    .hero-pills { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
    .pill {
      padding: 8px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.14);
      border: 1px solid rgba(255, 255, 255, 0.18);
      font-size: 14px;
    }
    .section { margin-top: 18px; background: var(--card); border: 1px solid rgba(217, 223, 235, 0.92); border-radius: 24px; box-shadow: var(--shadow); padding: 24px 26px; }
    .section h2 { margin: 0 0 10px; font-size: 25px; line-height: 1.3; }
    .lead { margin: 0; color: var(--muted); font-size: 15px; }
    .dataset-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin-top: 18px; }
    .dataset-card { border: 1px solid var(--line); border-radius: 22px; padding: 20px; background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%); }
    .dataset-top { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
    .dataset-top h3 { margin: 0; font-size: 24px; }
    .dataset-top h3 span { color: var(--accent); font-size: 18px; }
    .dataset-range { margin: 8px 0 0; color: var(--muted); font-size: 14px; }
    .dataset-badge { padding: 7px 12px; border-radius: 999px; background: #eef5ff; color: #173b8f; font-size: 13px; white-space: nowrap; }
    .dataset-summary { margin: 14px 0 0; color: var(--ink); }
    .metric-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }
    .metric-item { border: 1px solid var(--line); border-radius: 18px; padding: 12px 14px; background: #fff; min-height: 92px; }
    .metric-label { color: var(--muted); font-size: 13px; }
    .metric-value { margin-top: 4px; font-size: 22px; line-height: 1.15; font-weight: 700; color: var(--accent-deep); font-family: "JetBrains Mono", "Consolas", monospace; }
    .metric-hint { margin-top: 6px; color: var(--muted); font-size: 12px; }
    .two-col { display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px; margin-top: 18px; }
    .formula-grid { display: grid; gap: 10px; margin-top: 16px; }
    .formula { border: 1px solid #d7e3ff; border-radius: 18px; padding: 13px 15px; background: #f8faff; color: #173067; font-family: "JetBrains Mono", "Consolas", monospace; }
    .glossary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin-top: 16px; }
    .glossary-item { border: 1px solid var(--line); border-radius: 18px; padding: 14px 15px; background: #fff; }
    .glossary-item h4 { margin: 0 0 6px; font-size: 15px; color: var(--accent-deep); }
    .glossary-item p { margin: 0; color: var(--muted); font-size: 14px; }
    .plot-wrap { margin-top: 16px; border: 1px solid #d7e3ff; border-radius: 22px; padding: 12px; background: #ffffff; overflow: hidden; }
    .hint { margin-top: 10px; color: var(--muted); font-size: 14px; }
    details { margin-top: 14px; border: 1px solid var(--line); border-radius: 18px; padding: 12px 14px; background: #fcfdff; }
    summary { cursor: pointer; font-weight: 700; }
    .table-wrap { margin-top: 14px; overflow-x: auto; border: 1px solid var(--line); border-radius: 20px; }
    .report-table { width: 100%; border-collapse: collapse; min-width: 980px; }
    .report-table thead th { position: sticky; top: 0; background: #eef5ff; color: #16357a; padding: 11px 12px; text-align: left; border-bottom: 1px solid var(--line); }
    .report-table tbody td { padding: 11px 12px; border-bottom: 1px solid #ebeff5; font-size: 14px; vertical-align: top; }
    .report-table tbody tr:nth-child(even) { background: #fbfdff; }
    .takeaway-list { margin: 12px 0 0; padding-left: 20px; }
    .takeaway-list li { margin: 10px 0; }
    .footer { margin-top: 16px; color: var(--muted); text-align: center; font-size: 14px; }
    @media (max-width: 980px) {
      .page { width: min(100vw - 20px, 1240px); }
      .hero { padding: 26px 22px; }
      .section { padding: 20px 18px; }
      .two-col { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>负尾幂律现代检验报告</h1>
      <p>报告使用自动 xmin 选择、极大似然估计（maximum-likelihood estimation）、KS 距离、bootstrap 区间、拟合优度检验（goodness-of-fit）与替代分布比较，展示 XAGUSD 在 1h、2h、1d 三个时间尺度上的负尾行为。</p>
      <div class="hero-pills">
        <div class="pill">数据集 = {{ results|length }}</div>
        <div class="pill">bootstrap 区间重采样 = {{ ci_reps }}</div>
        <div class="pill">GOF 检验重采样 = {{ gof_reps }}</div>
        <div class="pill">方法 = 现代尾部统计检验</div>
      </div>
    </section>

    <section class="section">
      <h2>交互尾部图</h2>
      <p class="lead">上图单独放大 2h 主图，下图保留 1h 与 1d 作为对照。图上的每一个点都对应一个真实尾部事件。鼠标悬停时会显示出现时间、真实跌幅、对数收益率、标准化收益率、尾部幅度 x，以及阈值右侧剩余样本数。虚线对应自动选中的 xmin。</p>
      <div class="plot-wrap">{{ focus_tail_fig | safe }}</div>
      <p class="hint">主图使用 2h，原因是 2h 的尾部样本数最多，图形信息最完整。纵轴使用条件 CCDF，也就是样本已经进入尾部区域 x &gt;= xmin 以后，继续超过某个阈值的概率。</p>
      <div class="plot-wrap">{{ compare_tail_fig | safe }}</div>
      <p class="hint">对照图用于观察 1h 与 1d 在斜率、xmin 位置和右端弯曲方向上的差异。</p>
    </section>

    <section class="section">
      <h2>如何读这张图</h2>
      <p class="lead">下面这几条解释专门回答 alpha 的意义、点在线上方或下方的意义，以及当前三组数据可以从图上直接读出的结论。</p>
      <div class="glossary-grid">{{ tail_chart_reading_html | safe }}</div>
    </section>

    <section class="section">
      <h2>名词解释</h2>
      <p class="lead">这些指标会反复出现在图形、数据摘要和模型比较表里。先统一口径，后续阅读会轻松许多。</p>
      <div class="glossary-grid">{{ glossary_html | safe }}</div>
    </section>

    <section class="section">
      <h2>方法与 xmin 选择</h2>
      <div class="two-col">
        <div>
          <p class="lead">每一个时间尺度都单独处理。我们先计算对数收益率，再完成全样本标准化，随后提取负标准化收益率并转成正的尾部样本。幂律拟合只在 x &gt;= xmin 的区间里进行。</p>
          <div class="formula-grid">
            <div class="formula">r_t = ln(P_t) - ln(P_{t-1})</div>
            <div class="formula">g_t = (r_t - mean(r)) / std(r)</div>
            <div class="formula">x_t = -g_t, when g_t &lt; 0</div>
            <div class="formula">P(X &gt; x | X &gt;= xmin) = (x / xmin)^(-alpha)</div>
            <div class="formula">alpha_hat = n / Σ log(x_i / xmin)</div>
          </div>
        </div>
        <div>
          <p class="lead">xmin 的选择规则是明确可复核的。</p>
          <div class="glossary-grid">
            <div class="glossary-item"><h4>步骤 1</h4><p>从排序后的负尾样本里生成候选 xmin。候选值需要满足最小尾部样本规则 min_tail = max(10, min(50, n_negative // 10))。</p></div>
            <div class="glossary-item"><h4>步骤 2</h4><p>针对每一个候选 xmin，只使用 x &gt;= xmin 的样本拟合连续 Pareto 尾部，并用极大似然估计 alpha。</p></div>
            <div class="glossary-item"><h4>步骤 3</h4><p>计算经验条件 CDF 与拟合 Pareto CDF 之间的 KS 距离。这个距离衡量经验尾部形状与理论曲线的贴合程度。</p></div>
            <div class="glossary-item"><h4>步骤 4</h4><p>保留 KS 距离最小的 xmin。若两个候选值的 KS 一样，规则会保留较大的 xmin，使尾部定义更克制。</p></div>
            <div class="glossary-item"><h4>步骤 5</h4><p>选定 xmin 以后，再进行 bootstrap 置信区间、GOF 检验，以及幂律与 exponential、lognormal、truncated power law 的比较。</p></div>
          </div>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>xmin 路径与 alpha 区间</h2>
      <p class="lead">左图展示候选 xmin 对应的 KS 路径，右图展示最终 alpha 的 bootstrap 区间。虚线基准 alpha = 3 对应 inverse cubic law 的 CCDF 口径。</p>
      <div class="two-col">
        <div class="plot-wrap">{{ xmin_fig | safe }}</div>
        <div class="plot-wrap">{{ alpha_fig | safe }}</div>
      </div>
    </section>

    <section class="section">
      <h2>数据摘要</h2>
      <p class="lead">每张卡片都汇总同一个时间尺度的样本规模、尾部规模与现代检验结果。阅读图以后，再看这里会非常顺手。</p>
      <div class="dataset-grid">{{ dataset_cards | safe }}</div>
    </section>

    <section class="section">
      <h2>模型比较</h2>
      <p class="lead">图里使用 Delta AIC 比较幂律与替代模型。数值落在 0 以下，表示替代模型的 AIC 较小。鼠标悬停时可以直接读取 log-likelihood、KS、LLR 与 Vuong p 值。</p>
      <div class="plot-wrap">{{ model_fig | safe }}</div>
      <details>
        <summary>如何阅读 Delta AIC</summary>
        <p>Delta AIC = AIC(model) - AIC(power law)。幂律的 Delta AIC 恒等于 0。显著的负值说明替代模型在拟合效果与参数复杂度之间取得了较好的平衡。绝对值很小的时候，两边都可以解释这段尾部。</p>
      </details>
      {{ model_table | safe }}
    </section>

    <section class="section">
      <h2>结论</h2>
      <p class="lead">以下结论全部使用现代检验流程，并统一采用 CCDF 尾指数 alpha 的口径。</p>
      <ul class="takeaway-list">
        {% for note in takeaways %}
        <li>{{ note }}</li>
        {% endfor %}
      </ul>
      <details>
        <summary>CCDF 指数与 PDF 指数的关系</summary>
        <p>报告正文统一使用 CCDF 指数 alpha。若文献采用 PDF 口径，指数会多 1，因此 alpha PDF = alpha CCDF + 1。按照 CCDF 口径，inverse cubic law 的基准值是 alpha = 3；按照 PDF 口径，对应的基准值是 4。</p>
      </details>
    </section>

    <div class="footer">基于项目样本数据生成</div>
  </div>
</body>
</html>
"""
    )
    return template.render(
        results=results,
        ci_reps=CI_BOOTSTRAP_REPS,
        gof_reps=GOF_BOOTSTRAP_REPS,
        dataset_cards=dataset_cards,
        glossary_html=glossary_html,
        focus_tail_fig=focus_tail_fig,
        compare_tail_fig=compare_tail_fig,
        tail_chart_reading_html=tail_chart_reading_html,
        xmin_fig=xmin_fig,
        alpha_fig=alpha_fig,
        model_fig=model_fig,
        model_table=model_table,
        takeaways=takeaways,
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = load_datasets()
    if not datasets:
        raise SystemExit("No sample data found in data directory.")

    results = [analyze_dataset(dataset, seed_offset=idx * 1000) for idx, dataset in enumerate(datasets)]
    html = render_modern_report_html(results)

    payload_results = []
    for result in results:
        payload_results.append(
            {
                "meta": result["meta"],
                "modern": result["modern"],
                "bootstrap": result["bootstrap"],
                "gof": result["gof"],
                "comparisons": result["comparisons"],
            }
        )
    payload = {
        "report_mode": "modern_only",
        "ci_bootstrap_reps": CI_BOOTSTRAP_REPS,
        "gof_bootstrap_reps": GOF_BOOTSTRAP_REPS,
        "results": payload_results,
    }
    HTML_PATH.write_text(html, encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved HTML report to: {HTML_PATH}")
    print(f"Saved machine-readable results to: {JSON_PATH}")


if __name__ == "__main__":
    main()
