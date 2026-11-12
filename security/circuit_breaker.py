"""
Hybrid circuit breaker - the unified security decision engine (spec Section 7).

    1. PQC authentication failed            -> ABORT_MITM
    2. temporal QBER -> 63 features -> XGBoost
    3. Normal  with confidence  > 0.85      -> PROCEED
       Normal  with confidence <= 0.85      -> CAUTION
       Attack  with confidence  > 0.70      -> ABORT_ATTACK
       Attack  with confidence <= 0.70      -> WARN
    4. SHAP explanation for ABORT_ATTACK / WARN
    5. attack-specific countermeasure + forensics report

`evaluate_with_resampling` implements "if uncertain -> increase sample size,
re-classify": extra independent sessions of the SAME size are collected (so
the feature distribution matches training) and their evidence is fused with
Bayes' rule - the posterior over classes is the normalised product of the
per-session class probabilities (uniform prior, independent sessions).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from config import settings as S
from ml.attack_classifier import CLASS_LABELS, AttackClassifier
from ml.feature_engine import FEATURE_NAMES, FeatureEngine
from ml.shap_explainer import SHAPExplainer
from ml.temporal_qber import TemporalQBER
from security.forensics import Forensics
from security.response_router import ResponseRouter

DECISIONS = ("PROCEED", "CAUTION", "ABORT_MITM", "ABORT_ATTACK", "WARN")


def threshold_verdict(mean_qber: float, threshold: float = S.FIXED_QBER_THRESHOLD) -> dict:
    return {"mean_qber": float(mean_qber), "threshold": threshold,
            "verdict": "ABORT (QBER above threshold)" if mean_qber > threshold
            else "ALL CLEAR (QBER below threshold)",
            "flags_attack": bool(mean_qber > threshold)}


class CircuitBreaker:
    """Checks PQC auth, runs the ML classifier, routes the response."""

    def __init__(self, classifier_path: str = str(S.XGBOOST_MODEL_PATH),
                 confidence_threshold: float = S.ATTACK_CONFIDENCE_THRESHOLD,
                 normal_threshold: float = S.NORMAL_CONFIDENCE_THRESHOLD,
                 save_reports: bool = True):
        self.classifier = AttackClassifier("xgboost")
        self.classifier.load(classifier_path)
        self.temporal_qber = TemporalQBER(window_size=S.WINDOW_SIZE)
        self.feature_engine = FeatureEngine()
        self.confidence_threshold = confidence_threshold
        self.normal_threshold = normal_threshold
        self.shap_explainer = SHAPExplainer(self.classifier)
        self.save_reports = save_reports

    # ------------------------------------------------------------------
    def features_for(self, session_result: dict) -> tuple[np.ndarray, dict]:
        td = self.temporal_qber.compute(session_result["sifted_alice"], session_result["sifted_bob"],
                                        session_result["sifted_bases"],
                                        matching_indices=session_result.get("matching_indices"),
                                        lost_mask=session_result.get("lost_mask"))
        x = self.feature_engine.extract(td, session_result["metadata"])
        return x, td

    def decide(self, pred_class: int, max_confidence: float) -> str:
        if pred_class == 0 and max_confidence > self.normal_threshold:
            return "PROCEED"
        if pred_class == 0:
            return "CAUTION"
        if max_confidence > self.confidence_threshold:
            return "ABORT_ATTACK"
        return "WARN"

    @staticmethod
    def qber_stats(x: np.ndarray, td: dict) -> dict:
        f = dict(zip(FEATURE_NAMES, x))
        return {"n_windows": int(td["n_windows"]), "window_size": int(td["window_size"]),
                "mean_qber": float(f["qber_mean"]), "std_qber": float(f["qber_std"]),
                "max_qber": float(f["qber_max"]), "spike_density": float(f["spike_density"]),
                "jump_energy": float(f["jump_energy"]), "autocorr_lag1": float(f["autocorr_lag1"]),
                "qber_rect_mean": float(f["qber_rect_mean"]), "qber_diag_mean": float(f["qber_diag_mean"]),
                "basis_asymmetry": float(f["basis_asymmetry"]), "key_gen_rate": float(f["key_gen_rate"]),
                "photon_loss_rate": float(f["photon_loss_rate"])}

    # ------------------------------------------------------------------
    def _mitm_result(self, session_result: dict | None, pqc_log) -> dict:
        response = ResponseRouter.get_response("Classical MITM")
        report = Forensics.generate(
            decision="ABORT_MITM", attack_class="Classical MITM", confidence=1.0,
            probabilities=None, shap_values=None, qber_stats=None, response=response,
            session_metadata=(session_result or {}).get("metadata"), pqc_verified=False,
            pqc_log=pqc_log, save=self.save_reports)
        return {"decision": "ABORT_MITM", "attack_class": "Classical MITM", "confidence": 1.0,
                "all_probabilities": {}, "shap_values": None, "forensics_report": report["text"],
                "report_paths": {k: report.get(k) for k in ("path_json", "path_txt")},
                "recommended_action": response["action"], "response": response,
                "qber_stats": {}, "pqc_verified": False, "threshold_verdict": None,
                "qber_sequence": None, "features": None}

    def evaluate(self, session_result: dict, pqc_verified: bool, pqc_log: list | None = None,
                 probabilities_override: np.ndarray | None = None, extra: dict | None = None,
                 pdf_report: bool = False) -> dict:
        """Full security evaluation of a completed QKD session."""
        # Step 1: PQC authentication
        if not pqc_verified:
            return self._mitm_result(session_result, pqc_log)

        # Step 2: temporal QBER features
        features, td = self.features_for(session_result)

        # Step 3: classifier
        if probabilities_override is None:
            pred_class, probabilities = self.classifier.predict(features)
        else:
            probabilities = np.asarray(probabilities_override)
            pred_class = int(probabilities.argmax())

        # Step 4: decision
        max_confidence = float(probabilities.max())
        decision = self.decide(pred_class, max_confidence)
        attack_class = CLASS_LABELS[pred_class]

        # Step 5: SHAP explanation when an attack is suspected
        shap_values = None
        if decision in ("ABORT_ATTACK", "WARN"):
            shap_values = self.shap_explainer.explain(features, top_k=5, class_idx=pred_class)

        # Step 6: response routing
        response = ResponseRouter.get_response(attack_class)
        stats = self.qber_stats(features, td)
        tv = threshold_verdict(stats["mean_qber"])
        probs = {CLASS_LABELS[i]: float(p) for i, p in enumerate(probabilities)}

        # Step 7: forensics
        report = Forensics.generate(
            decision=decision, attack_class=None if pred_class == 0 else attack_class,
            confidence=max_confidence, probabilities=probs, shap_values=shap_values,
            qber_stats=stats, response=response, session_metadata=session_result.get("metadata"),
            pqc_verified=True, pqc_log=pqc_log, threshold_verdict=tv,
            qber_sequence=td["qber_sequence"], save=self.save_reports, pdf=pdf_report)
        out = {
            "decision": decision,
            "attack_class": None if pred_class == 0 else attack_class,
            "predicted_label": attack_class,
            "confidence": max_confidence,
            "all_probabilities": probs,
            "shap_values": ({"top_features": shap_values["top_features"],
                             "base_value": shap_values["base_value"],
                             "explained_class": shap_values["explained_class"],
                             "interpretations": shap_values["interpretations"],
                             "full_shap_values": shap_values["full_shap_values"]}
                            if shap_values else None),
            "forensics_report": report["text"],
            "report_paths": {k: report.get(k) for k in ("path_json", "path_txt", "path_pdf")},
            "recommended_action": response["action"],
            "response": response,
            "qber_stats": stats,
            "pqc_verified": True,
            "threshold_verdict": tv,
            "qber_sequence": td["qber_sequence"],
            "qber_rect_sequence": td["qber_rect_sequence"],
            "qber_diag_sequence": td["qber_diag_sequence"],
            "features": features,
        }
        if extra:
            out.update(extra)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def fuse(prob_list: list[np.ndarray]) -> np.ndarray:
        """Bayesian fusion of independent sessions: normalised product."""
        logp = np.sum(np.log(np.clip(np.asarray(prob_list), 1e-9, 1.0)), axis=0)
        p = np.exp(logp - logp.max())
        return p / p.sum()

    def evaluate_with_resampling(self, session_factory: Callable[[int], dict], pqc_verified: bool,
                                 pqc_log: list | None = None,
                                 max_rounds: int = S.MAX_RECLASSIFY_ROUNDS,
                                 pdf_report: bool = False) -> dict:
        """Evaluate; while the verdict is CAUTION/WARN, collect another
        independent session (`session_factory(round_index)`) and re-classify on
        the fused class posterior."""
        first = session_factory(0)
        if not pqc_verified:
            return self._mitm_result(first, pqc_log)
        x, _ = self.features_for(first)
        probs = [self.classifier.predict(x)[1]]
        sessions = [first]
        fused = probs[0]
        decision = self.decide(int(fused.argmax()), float(fused.max()))
        r = 0
        while decision in ("CAUTION", "WARN") and r < max_rounds:
            r += 1
            s = session_factory(r)
            sessions.append(s)
            probs.append(self.classifier.predict(self.features_for(s)[0])[1])
            fused = self.fuse(probs)
            decision = self.decide(int(fused.argmax()), float(fused.max()))
        return self.evaluate(sessions[-1], True, pqc_log, probabilities_override=fused,
                             pdf_report=pdf_report,
                             extra={"resampling_rounds": r, "n_sessions_used": len(sessions),
                                    "per_round_probabilities": [p.tolist() for p in probs]})
