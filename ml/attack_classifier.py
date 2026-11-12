"""
Attack classifier: XGBoost / Random Forest / SVM-RBF (spec Section 5).

The classifier owns its StandardScaler: `train` fits the scaler on the raw
training features and `predict` always applies `scaler.transform` first, so
callers pass raw 63-feature vectors everywhere.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import settings as S

CLASS_LABELS = dict(S.CLASS_LABELS)
N_CLASSES = len(CLASS_LABELS)
MODEL_TYPES = ("xgboost", "random_forest", "svm_rbf")


def _make_estimator(model_type: str, random_state: int = S.RANDOM_STATE, **params):
    if model_type == "xgboost":
        from xgboost import XGBClassifier
        base = dict(objective="multi:softprob", num_class=N_CLASSES, tree_method="hist",
                    eval_metric="mlogloss", n_jobs=-1, random_state=random_state)
        base.update(params)
        return XGBClassifier(**base)
    if model_type == "random_forest":
        base = dict(n_jobs=-1, random_state=random_state)
        base.update(params)
        return RandomForestClassifier(**base)
    if model_type == "svm_rbf":
        params = dict(params)
        probability = params.pop("probability", True)
        svc = SVC(kernel="rbf", random_state=random_state, **params)
        if not probability:
            return svc
        # Platt-scaled probabilities (replacement for the deprecated SVC(probability=True))
        from sklearn.calibration import CalibratedClassifierCV
        return CalibratedClassifierCV(svc, method="sigmoid", cv=5, ensemble=False)
    raise ValueError(f"model_type must be one of {MODEL_TYPES}")


def _param_grid(model_type: str) -> dict:
    return {"xgboost": S.XGB_PARAM_GRID, "random_forest": S.RF_PARAM_GRID,
            "svm_rbf": S.SVM_PARAM_GRID}[model_type]


class AttackClassifier:
    def __init__(self, model_type: str = "xgboost"):
        """model_type: 'xgboost', 'random_forest', or 'svm_rbf'"""
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}")
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.best_params: dict = {}

    # ------------------------------------------------------------------
    def train(self, X_train: np.ndarray, y_train: np.ndarray, cv_folds: int = S.CV_FOLDS,
              search: str = "grid", n_iter: int = 20, tune_subsample: int | None = None,
              params: dict | None = None, random_state: int = S.RANDOM_STATE,
              verbose: int = 0) -> dict:
        """
        Train with hyper-parameter tuning via stratified k-fold CV.

        search:   'grid' (exhaustive, as specified), 'random' (n_iter samples of
                  the same grid) or 'none' (use `params` directly).
        tune_subsample: tune on a stratified subsample of this size (useful for
                  the O(n^2) SVM), then refit the best model on all data.
        Returns {'cv_accuracy', 'cv_f1', 'best_params', ...}
        """
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=int)
        Xs = self.scaler.fit_transform(X_train)

        if search == "none":
            self.best_params = dict(params or {})
            self.model = _make_estimator(self.model_type, random_state, **self.best_params)
            self.model.fit(Xs, y_train)
            return {"cv_accuracy": None, "cv_f1": None, "best_params": self.best_params}

        Xt, yt = Xs, y_train
        if tune_subsample and tune_subsample < len(y_train):
            from sklearn.model_selection import train_test_split
            Xt, _, yt, _ = train_test_split(Xs, y_train, train_size=tune_subsample,
                                            stratify=y_train, random_state=random_state)
        # probability calibration is not needed while searching (much faster for SVM)
        est = _make_estimator(self.model_type, random_state,
                              **({"probability": False} if self.model_type == "svm_rbf" else {}))
        if self.model_type == "xgboost":
            est.set_params(n_jobs=1)          # parallelise across CV fits instead
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
        common = dict(scoring=scoring, refit="accuracy", cv=cv, n_jobs=-1, verbose=verbose)
        if search == "grid":
            gs = GridSearchCV(est, _param_grid(self.model_type), **common)
        elif search == "random":
            gs = RandomizedSearchCV(est, _param_grid(self.model_type), n_iter=n_iter,
                                    random_state=random_state, **common)
        else:
            raise ValueError("search must be 'grid', 'random' or 'none'")
        gs.fit(Xt, yt)
        best = gs.best_index_
        self.best_params = dict(gs.best_params_)
        self.model = _make_estimator(self.model_type, random_state, **self.best_params)
        self.model.fit(Xs, y_train)
        return {
            "cv_accuracy": float(gs.cv_results_["mean_test_accuracy"][best]),
            "cv_accuracy_std": float(gs.cv_results_["std_test_accuracy"][best]),
            "cv_f1": float(gs.cv_results_["mean_test_f1_macro"][best]),
            "cv_f1_std": float(gs.cv_results_["std_test_f1_macro"][best]),
            "best_params": self.best_params,
            "n_candidates": int(len(gs.cv_results_["params"])),
            "tuned_on": int(len(yt)),
        }

    # ------------------------------------------------------------------
    def _check(self):
        if self.model is None:
            raise RuntimeError("Classifier is not trained/loaded")

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != S.N_FEATURES:
            raise ValueError(f"expected {S.N_FEATURES} features, got {X.shape[1]}")
        return self.scaler.transform(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._check()
        return self.model.predict_proba(self.transform(X))

    def predict(self, X: np.ndarray):
        """
        X of shape (63,)          -> (predicted_class: int, probabilities: (8,))
        X of shape (n, 63)        -> (classes: (n,), probabilities: (n, 8))
        """
        single = np.asarray(X).ndim == 1
        proba = self.predict_proba(X)
        cls = proba.argmax(axis=1)
        if single:
            return int(cls[0]), proba[0]
        return cls, proba

    # ------------------------------------------------------------------
    def save(self, path: str, scaler_path: str | None = None):
        """XGBoost -> native JSON at `path`; sklearn models -> joblib pickle.
        The scaler is stored next to the model (scaler_v1.pkl by default)."""
        self._check()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path = Path(scaler_path) if scaler_path else path.parent / S.SCALER_PATH.name
        if self.model_type == "xgboost":
            self.model.save_model(str(path))
        else:
            joblib.dump({"model": self.model, "scaler": self.scaler,
                         "best_params": self.best_params}, path)
        joblib.dump(self.scaler, scaler_path)
        meta = {"model_type": self.model_type, "best_params": self.best_params,
                "n_features": S.N_FEATURES, "class_labels": CLASS_LABELS}
        with open(path.with_suffix(".meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def load(self, path: str, scaler_path: str | None = None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}. Run `python main.py train` first.")
        if path.suffix == ".json":
            from xgboost import XGBClassifier
            self.model_type = "xgboost"
            self.model = XGBClassifier()
            self.model.load_model(str(path))
            sp = Path(scaler_path) if scaler_path else path.parent / S.SCALER_PATH.name
            self.scaler = joblib.load(sp)
        else:
            blob = joblib.load(path)
            self.model, self.scaler = blob["model"], blob["scaler"]
            self.best_params = blob.get("best_params", {})
            self.model_type = ("random_forest" if isinstance(self.model, RandomForestClassifier)
                               else "svm_rbf")
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            with open(meta_path) as fh:
                self.best_params = json.load(fh).get("best_params", self.best_params)
        return self

    @classmethod
    def from_file(cls, path: str) -> "AttackClassifier":
        c = cls("xgboost")
        c.load(path)
        return c


def default_model_path(model_type: str = "xgboost") -> str:
    return str({"xgboost": S.XGBOOST_MODEL_PATH, "random_forest": S.RF_MODEL_PATH,
                "svm_rbf": S.SVM_MODEL_PATH}[model_type])


def model_available(model_type: str = "xgboost") -> bool:
    return os.path.exists(default_model_path(model_type))
