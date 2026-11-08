"""Evaluation metrics for the thermal twin project."""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error in degrees C."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error in degrees C."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def skill_score_vs_persistence(mae_model: float, mae_persistence: float) -> float:
    """Skill score: 1 - MAE_model / MAE_persistence.

    Positive = model beats persistence; 0 = equal; negative = worse.
    """
    if mae_persistence == 0:
        return 0.0
    return 1.0 - mae_model / mae_persistence


def persistence_predictions(y_series: pd.Series, horizon_steps: int) -> pd.Series:
    """Generate persistence baseline: prediction = value `horizon_steps` ago."""
    return y_series.shift(horizon_steps)


def per_server_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    server_col: str = "Series",
    y_persist_col: str | None = None,
) -> pd.DataFrame:
    """Compute MAE, RMSE, and optional skill score per server.

    Returns DataFrame with columns: server, mae, rmse, n_samples, [skill_score].
    """
    records = []
    for server, group in df.groupby(server_col):
        valid = group[[y_true_col, y_pred_col]].dropna()
        if len(valid) == 0:
            continue
        yt = valid[y_true_col].values
        yp = valid[y_pred_col].values
        m = mae(yt, yp)
        r = rmse(yt, yp)
        record = {"server": server, "mae": m, "rmse": r, "n_samples": len(valid)}

        if y_persist_col and y_persist_col in group.columns:
            valid_p = group[[y_true_col, y_persist_col]].dropna()
            if len(valid_p) > 0:
                mae_p = mae(valid_p[y_true_col].values, valid_p[y_persist_col].values)
                record["mae_persistence"] = mae_p
                record["skill_score"] = skill_score_vs_persistence(m, mae_p)

        records.append(record)

    return pd.DataFrame(records)


def summarize_per_server(per_server_df: pd.DataFrame, metric_col: str = "mae") -> dict:
    """Compute summary statistics across servers for a metric."""
    vals = per_server_df[metric_col].dropna()
    if len(vals) == 0:
        return {}
    return {
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(vals.median()),
        "q25": float(vals.quantile(0.25)),
        "q75": float(vals.quantile(0.75)),
        "iqr": float(vals.quantile(0.75) - vals.quantile(0.25)),
        "worst_decile": float(vals.quantile(0.9)),
        "max": float(vals.max()),
        "min": float(vals.min()),
        "n_servers": len(vals),
    }
