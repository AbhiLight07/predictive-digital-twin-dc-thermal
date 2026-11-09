"""Few-shot / limited-data adaptation for the thermal twin project (E4).

Strategies:
  - Direct transfer: apply source model to target server as-is (zero-shot)
  - Fine-tune LightGBM: retrain on N adaptation days, using source as prior
  - Feature normalisation adaptation: re-fit scaler on target data, reuse weights
  - Warm-start Ridge: fit on small target data only
"""

import numpy as np
import pandas as pd
from typing import Optional, List
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def direct_transfer(source_model, X_target: np.ndarray) -> np.ndarray:
    """Zero-shot: apply source model directly to target server.

    Args:
        source_model: fitted model with .predict() method
        X_target: feature matrix for target server
    Returns:
        Predictions array
    """
    return source_model.predict(X_target)


def adapt_ridge(
    source_model,
    X_adapt: np.ndarray,
    y_adapt: np.ndarray,
    X_target: np.ndarray,
    alpha: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Fit a fresh Ridge on adaptation data only.

    For small N (days), Ridge is more stable than tree models.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_adapt)
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(X_scaled, y_adapt)
    return model.predict(scaler.transform(X_target))


def adapt_lightgbm_finetune(
    source_model: lgb.Booster,
    X_adapt: np.ndarray,
    y_adapt: np.ndarray,
    X_target: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_rounds: int = 50,
    learning_rate: float = 0.01,
) -> np.ndarray:
    """Fine-tune LightGBM by continuing training from source model checkpoint.

    Uses init_model to warm-start from the source model's trees.
    """
    dtrain = lgb.Dataset(X_adapt, label=y_adapt, feature_name=feature_names or "auto")
    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": learning_rate,
        "verbosity": -1,
        "num_leaves": 31,  # smaller for fine-tuning
    }
    adapted = lgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        init_model=source_model,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return adapted.predict(X_target)


def run_adaptation_experiment(
    df_train: pd.DataFrame,
    df_target_server: pd.DataFrame,
    source_model,
    feature_cols: List[str],
    target_col: str,
    adapt_days: List[int],
    time_col: str = "Time",
    seed: int = 42,
) -> pd.DataFrame:
    """Run E4: evaluate adaptation with varying amounts of target-server data.

    Args:
        df_train: training data for fitting source model (already fitted externally)
        df_target_server: all data for the new (unseen) target server
        source_model: fitted model
        feature_cols: feature column names
        target_col: target column name (e.g. 'CPU_Temperature_target_1')
        adapt_days: list of adaptation day counts to evaluate [1, 3, 7, 14, 30]
        time_col: timestamp column

    Returns:
        DataFrame with columns: adapt_days, strategy, mae, rmse, n_test
    """
    from ttwin.metrics import mae, rmse

    # Sort by time
    df_server = df_target_server.sort_values(time_col).reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(df_server[time_col]):
        df_server[time_col] = pd.to_datetime(df_server[time_col])

    t_min = df_server[time_col].min()
    results = []

    # Zero-shot baseline (no adaptation data)
    test_data = df_server[feature_cols + [target_col]].dropna()
    if len(test_data) > 0:
        X_test = test_data[feature_cols].values
        y_test = test_data[target_col].values
        y_pred_zero = source_model.predict(X_test)
        results.append({
            "adapt_days": 0,
            "strategy": "zero_shot",
            "mae": mae(y_test, y_pred_zero),
            "rmse": rmse(y_test, y_pred_zero),
            "n_adapt": 0,
            "n_test": len(y_test),
        })

    for n_days in adapt_days:
        adapt_end = t_min + pd.Timedelta(days=n_days)
        adapt_df = df_server[df_server[time_col] < adapt_end]
        # Embargo: skip 6h after adapt period
        test_start = adapt_end + pd.Timedelta(hours=6)
        test_df = df_server[df_server[time_col] >= test_start]

        adapt_clean = adapt_df[feature_cols + [target_col]].dropna()
        test_clean = test_df[feature_cols + [target_col]].dropna()

        if len(adapt_clean) < 5 or len(test_clean) < 5:
            continue

        X_adapt = adapt_clean[feature_cols].values
        y_adapt = adapt_clean[target_col].values
        X_test = test_clean[feature_cols].values
        y_test = test_clean[target_col].values

        # Strategy 1: direct transfer (zero-shot but evaluated on post-adapt test)
        y_pred_direct = source_model.predict(X_test)
        results.append({
            "adapt_days": n_days,
            "strategy": "direct_transfer",
            "mae": mae(y_test, y_pred_direct),
            "rmse": rmse(y_test, y_pred_direct),
            "n_adapt": len(X_adapt),
            "n_test": len(X_test),
        })

        # Strategy 2: Ridge fine-tune
        try:
            y_pred_ridge = adapt_ridge(source_model, X_adapt, y_adapt, X_test)
            results.append({
                "adapt_days": n_days,
                "strategy": "ridge_finetune",
                "mae": mae(y_test, y_pred_ridge),
                "rmse": rmse(y_test, y_pred_ridge),
                "n_adapt": len(X_adapt),
                "n_test": len(X_test),
            })
        except Exception as e:
            print(f"  [adapt] Ridge failed for {n_days}d: {e}")

        # Strategy 3: LightGBM fine-tune (only if source is LightGBM)
        if hasattr(source_model, "predict") and isinstance(source_model, lgb.Booster):
            try:
                y_pred_lgb = adapt_lightgbm_finetune(
                    source_model, X_adapt, y_adapt, X_test, feature_names=feature_cols
                )
                results.append({
                    "adapt_days": n_days,
                    "strategy": "lgbm_finetune",
                    "mae": mae(y_test, y_pred_lgb),
                    "rmse": rmse(y_test, y_pred_lgb),
                    "n_adapt": len(X_adapt),
                    "n_test": len(X_test),
                })
            except Exception as e:
                print(f"  [adapt] LightGBM fine-tune failed for {n_days}d: {e}")

        # Strategy 4: LightGBM trained strictly from scratch on adaptation data
        try:
            dtrain_scratch = lgb.Dataset(X_adapt, label=y_adapt, feature_name=feature_cols)
            params_scratch = {
                "objective": "regression",
                "metric": "mae",
                "learning_rate": 0.05,
                "verbosity": -1,
                "num_leaves": 15,
                "min_child_samples": 10,
                "seed": seed,
            }
            n_rounds = min(50, max(10, len(X_adapt) // 10))
            scratch_model = lgb.train(
                params_scratch,
                dtrain_scratch,
                num_boost_round=n_rounds,
                callbacks=[lgb.log_evaluation(period=0)],
            )
            y_pred_scratch = scratch_model.predict(X_test)
            results.append({
                "adapt_days": n_days,
                "strategy": "scratch_lgbm",
                "mae": mae(y_test, y_pred_scratch),
                "rmse": rmse(y_test, y_pred_scratch),
                "n_adapt": len(X_adapt),
                "n_test": len(X_test),
            })
        except Exception as e:
            print(f"  [adapt] Scratch LightGBM failed for {n_days}d: {e}")

    return pd.DataFrame(results)
