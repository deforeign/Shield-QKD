"""
SHAP explainability for the attack classifier (spec Section 8.1).

Uses shap.TreeExplainer for tree models.  If the installed shap / xgboost pair
is incompatible, it falls back to XGBoost's built-in exact TreeSHAP
(`pred_contribs=True`), which computes the same Shapley values.
SHAP values are computed in the *scaled* feature space the model sees, while
the reported `actual_value` is the raw (physical) feature value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml.attack_classifier import CLASS_LABELS, AttackClassifier
from ml.feature_engine import FEATURE_CATEGORY, FEATURE_NAMES

# Human-readable physics interpretation used in forensics reports.
FEATURE_EXPLANATIONS = {
    "qber_mean": "average error rate across the key",
    "qber_std": "window-to-window QBER fluctuation",
    "spike_density": "fraction of windows with abnormal error spikes",
    "burst_duration_max": "longest burst of high-error windows",
    "jump_energy": "energy of abrupt QBER jumps between windows",
    "autocorr_lag1": "temporal correlation of errors (sequenced probing)",
    "basis_asymmetry": "difference between rectilinear and diagonal error rates",
    "basis_ratio": "ratio of per-basis error rates",
    "key_gen_rate": "sifted-key yield (drops when photons are removed)",
    "photon_loss_rate": "fraction of photons that never reached Bob",
    "transmission_efficiency": "fraction of photons that reached Bob",
    "qber_loss_correlation": "co-occurrence of errors and photon loss in time",
    "effective_key_fraction": "key fraction left after estimated leakage",
    "qber_cv": "relative QBER variability",
    "hurst_exponent": "long-range dependence of the error process",
}


class SHAPExplainer:
    def __init__(self, classifier: AttackClassifier):
        self.classifier = classifier
        self.model = classifier.model
        self._mode = None
        self.explainer = None
        if classifier.model_type in ("xgboost", "random_forest"):
            try:
                import shap
                self.explainer = shap.TreeExplainer(classifier.model)
                # probe once - some shap/xgboost combinations fail lazily
                self.explainer.shap_values(np.zeros((1, len(FEATURE_NAMES))))
                self._mode = "shap"
            except Exception:
                self.explainer = None
                self._mode = "xgb_native" if classifier.model_type == "xgboost" else None
        if self._mode is None:
            import shap
            background = np.zeros((1, len(FEATURE_NAMES)))
            self.explainer = shap.KernelExplainer(classifier.model.predict_proba, background)
            self._mode = "kernel"

    # ------------------------------------------------------------------
    def _raw_shap(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (values (n, features, classes), base (classes,))."""
        if self._mode == "xgb_native":
            import xgboost as xgb
            contrib = self.model.get_booster().predict(xgb.DMatrix(Xs), pred_contribs=True)
            # shape (n, classes, features + 1); last column = bias
            vals = np.transpose(contrib[:, :, :-1], (0, 2, 1))
            base = contrib[0, :, -1]
            return vals, base
        sv = self.explainer.shap_values(Xs)
        ev = np.atleast_1d(self.explainer.expected_value)
        if isinstance(sv, list):                       # old shap: list per class
            vals = np.stack(sv, axis=-1)
        else:
            vals = np.asarray(sv)
            if vals.ndim == 2:                         # binary / single output
                vals = vals[..., None]
        return vals, ev

    def explain(self, features: np.ndarray, top_k: int = 5, class_idx: int | None = None) -> dict:
        """
        Explain one 63-feature vector for the PREDICTED class (or class_idx).
        Returns {'top_features': [(name, shap_value, actual_value), ...],
                 'full_shap_values': (63,), 'base_value': float, ...}
        """
        x = np.asarray(features, dtype=float).reshape(1, -1)
        Xs = self.classifier.transform(x)
        pred, proba = self.classifier.predict(x[0])
        cls = pred if class_idx is None else int(class_idx)
        vals, base = self._raw_shap(Xs)
        sv = vals[0, :, cls] if vals.shape[-1] > 1 else vals[0, :, 0]
        base_val = float(base[cls] if base.size > 1 else base[0])
        order = np.argsort(-np.abs(sv))[:top_k]
        top = [(FEATURE_NAMES[i], float(sv[i]), float(x[0, i])) for i in order]
        return {
            "top_features": top,
            "full_shap_values": sv.astype(float),
            "base_value": base_val,
            "explained_class": CLASS_LABELS[cls],
            "explained_class_idx": cls,
            "output_space": "log-odds (margin)" if self._mode != "kernel" else "probability",
            "method": self._mode,
            "interpretations": [
                {"feature": n, "category": FEATURE_CATEGORY.get(n, ""),
                 "shap_value": v, "actual_value": a,
                 "direction": "pushes TOWARD" if v > 0 else "pushes AWAY from",
                 "meaning": FEATURE_EXPLANATIONS.get(n, n.replace("_", " "))}
                for n, v, a in top
            ],
        }

    def global_importance(self, X_test: np.ndarray, max_samples: int = 2000) -> pd.DataFrame:
        """mean |SHAP| for each feature across the test set (all classes)."""
        X = np.asarray(X_test, dtype=float)
        if len(X) > max_samples:
            idx = np.random.default_rng(0).choice(len(X), max_samples, replace=False)
            X = X[idx]
        vals, _ = self._raw_shap(self.classifier.transform(X))
        imp = np.abs(vals).mean(axis=(0, 2))
        per_class = np.abs(vals).mean(axis=0)              # (features, classes)
        df = pd.DataFrame({"feature": FEATURE_NAMES, "mean_abs_shap": imp,
                           "category": [FEATURE_CATEGORY[n] for n in FEATURE_NAMES]})
        for c in range(per_class.shape[1]):
            df[CLASS_LABELS.get(c, str(c))] = per_class[:, c]
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
