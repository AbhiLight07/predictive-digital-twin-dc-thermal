"""Model definitions: persistence, linear, LightGBM, XGBoost.

All models follow a consistent interface:
  - fit(X_train, y_train, X_val=None, y_val=None) -> self
  - predict(X) -> np.ndarray
  - name -> str (model identifier)

Pooled models use cluster/hardware indicators and static capacity features
(cores, RAM) but NOT server ID for unseen-server tests.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb


class PersistenceModel:
    """Baseline: predict the last known value (shift by horizon)."""

    name = "persistence"

    def __init__(self):
        pass

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """No-op for persistence."""
        return self

    def predict(self, X):
        """Persistence predictions are computed externally via shift.
        This method raises an error to prevent misuse.
        """
        raise RuntimeError(
            "PersistenceModel.predict() should not be called directly. "
            "Use metrics.persistence_predictions() on the target column."
        )


class RidgeModel:
    """Ridge regression with standardisation."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0, seed: int = 42):
        self.alpha = alpha
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=seed)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class LightGBMModel:
    """LightGBM gradient boosting model."""

    name = "lightgbm"

    def __init__(self, params: Optional[Dict[str, Any]] = None, seed: int = 42,
                 n_estimators: int = 500, early_stopping_rounds: int = 50):
        self.seed = seed
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.params = params or {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbosity": -1,
            "seed": seed,
            "n_jobs": -1,
        }
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None):
        callbacks = [lgb.log_evaluation(period=100)]

        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))

        train_set = lgb.Dataset(X_train, label=y_train,
                                feature_name=feature_names or "auto")

        valid_sets = [train_set]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            val_set = lgb.Dataset(X_val, label=y_val, reference=train_set,
                                  feature_name=feature_names or "auto")
            valid_sets.append(val_set)
            valid_names.append("valid")

        self.model = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importances as a pandas Series."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        imp = self.model.feature_importance(importance_type=importance_type)
        names = self.model.feature_name()
        return pd.Series(imp, index=names).sort_values(ascending=False)


class XGBoostModel:
    """XGBoost gradient boosting model."""

    name = "xgboost"

    def __init__(self, params: Optional[Dict[str, Any]] = None, seed: int = 42,
                 n_estimators: int = 500, early_stopping_rounds: int = 50):
        self.seed = seed
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.params = params or {
            "objective": "reg:squarederror",
            "eval_metric": "mae",
            "learning_rate": 0.05,
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "seed": seed,
            "nthread": -1,
            "verbosity": 0,
        }
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[list] = None):
        dtrain = xgb.DMatrix(X_train, label=y_train,
                              feature_names=feature_names)
        evals = [(dtrain, "train")]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val,
                                feature_names=feature_names)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if X_val is not None else None,
            verbose_eval=100,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(X)
        return self.model.predict(dmat)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importances as a pandas Series."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        scores = self.model.get_score(importance_type=importance_type)
        return pd.Series(scores).sort_values(ascending=False)


def get_model(name: str, seed: int = 42, **kwargs):
    """Factory function to get a model by name.

    Args:
        name: One of 'persistence', 'ridge', 'lightgbm', 'xgboost'.
        seed: Random seed.
        **kwargs: Additional arguments passed to the model constructor.
    """
    models = {
        "persistence": PersistenceModel,
        "ridge": RidgeModel,
        "lightgbm": LightGBMModel,
        "xgboost": XGBoostModel,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name](seed=seed, **kwargs) if name != "persistence" else models[name]()
