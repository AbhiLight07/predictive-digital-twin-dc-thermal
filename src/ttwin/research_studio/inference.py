"""Model Inference Engine for the Research Studio.

Executes genuine offline inference on actual telemetry observations,
validating input schema, feature alignment, target alignment, and persistence comparisons.
Zero heuristics, zero synthetic noise, zero faking.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import pandas as pd
import numpy as np

from ttwin.research_studio.model_loader import ModelLoader
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.features import build_features, get_feature_set
from ttwin.splits import split_chronological


class ModelInferenceEngine:
    """Manages observation selection, feature vector preparation, and genuine model prediction."""

    def __init__(self, model_loader: Optional[ModelLoader] = None, dataset_loader: Optional[DatasetLoader] = None):
        self.model_loader = model_loader or ModelLoader()
        self.dataset_loader = dataset_loader or DatasetLoader()

    def run_inference(
        self,
        dataset_name: str,
        server_name: str,
        timestamp: pd.Timestamp,
        checkpoint_name: Optional[str] = None,
        horizon_steps: int = 6,  # default 1h = 6 steps
        feature_set: str = "S1",
    ) -> Dict[str, Any]:
        """Execute genuine model prediction for a specific server and timestamp.

        If no checkpoint exists, returns available=False with a clear reproducible training prompt.
        """
        # 1. Check model availability
        model_obj, model_desc, status_msg = self.model_loader.load_checkpoint(checkpoint_name)
        if model_obj is None:
            return {
                "available": False,
                "error": status_msg,
                "reproducible_command": self.model_loader.get_reproducible_training_command(),
                "dataset": dataset_name,
                "server": server_name,
                "timestamp": str(timestamp),
            }

        # 2. Load server observations with sufficient context for lag & rolling features
        df_srv = self.dataset_loader.load_interim_server(dataset_name, server_name)
        df_srv = df_srv.sort_values("Time").reset_index(drop=True)

        # Build verified features for this server
        df_features = build_features(
            df_srv,
            horizons=[horizon_steps],
            lags=[1, 2, 3, 6, 12],
            rolling_windows=[6, 12, 36],
        )

        # Locate exact row
        matching_rows = df_features[df_features["Time"] == timestamp]
        if matching_rows.empty:
            return {
                "available": False,
                "error": f"Timestamp '{timestamp}' not found in observations for server '{server_name}'.",
            }

        obs_row = matching_rows.iloc[0]

        # 3. Check future target availability
        target_abs_col = f"CPU_Temperature_target_{horizon_steps}"
        target_delta_col = f"CPU_Temperature_delta_{horizon_steps}"

        if target_abs_col not in obs_row or pd.isna(obs_row[target_abs_col]):
            return {
                "available": False,
                "error": (
                    f"Future observed target at horizon step {horizon_steps} is unobserved/missing. "
                    "Cannot evaluate prediction accuracy without verified ground truth."
                ),
            }

        cur_temp = float(obs_row["CPU_Temperature"])
        actual_future_temp = float(obs_row[target_abs_col])

        # 4. Prepare feature vector in exact order
        expected_features = model_desc.feature_names if model_desc.feature_names else get_feature_set(feature_set, df_features.columns.tolist())
        missing_feats = [f for f in expected_features if f not in obs_row or pd.isna(obs_row[f])]
        if missing_feats:
            return {
                "available": False,
                "error": f"Missing required input features in selected observation: {missing_feats[:5]}...",
            }

        feat_vector = np.array([[obs_row[f] for f in expected_features]], dtype=float)

        # 5. Predict using actual trained model
        raw_pred = float(model_obj.predict(feat_vector)[0])

        if model_desc.target_type == "delta":
            pred_delta = raw_pred
            pred_future_temp = cur_temp + pred_delta
        else:
            pred_future_temp = raw_pred
            pred_delta = pred_future_temp - cur_temp

        # 6. Evaluate errors and baseline
        model_abs_error = abs(pred_future_temp - actual_future_temp)
        persistence_pred = cur_temp
        persistence_abs_error = abs(persistence_pred - actual_future_temp)
        improvement_over_persistence = persistence_abs_error - model_abs_error

        # 7. Identify partition membership (Split A chronological 70/10/20)
        t_min = df_features["Time"].min()
        t_max = df_features["Time"].max()
        duration = t_max - t_min
        train_end = t_min + duration * 0.70
        val_end = train_end + duration * 0.10
        test_start = val_end + pd.Timedelta(minutes=10 * 36)  # 6h embargo

        if timestamp < train_end:
            partition = "TRAIN (In-Sample)"
        elif timestamp < val_end:
            partition = "VALIDATION (Model Selection)"
        elif timestamp < test_start:
            partition = "EMBARGO GAP (Excluded)"
        else:
            partition = "TEST (Held-Out Future)"

        return {
            "available": True,
            "dataset": dataset_name,
            "server": server_name,
            "timestamp": str(timestamp),
            "partition": partition,
            "horizon_steps": horizon_steps,
            "horizon_label": f"{horizon_steps * 10}min",
            "model_name": model_desc.name,
            "checkpoint_path": str(model_desc.path),
            "feature_set": model_desc.feature_set,
            "target_type": model_desc.target_type,
            "current_temp": cur_temp,
            "actual_future_temp": actual_future_temp,
            "predicted_future_temp": pred_future_temp,
            "predicted_delta": pred_delta,
            "model_abs_error": model_abs_error,
            "persistence_pred": persistence_pred,
            "persistence_abs_error": persistence_abs_error,
            "improvement_over_persistence": improvement_over_persistence,
            "feature_summary": {f: float(obs_row[f]) for f in expected_features[:10]},
        }
