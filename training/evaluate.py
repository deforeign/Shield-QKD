"""
Evaluation metrics, baselines and figures (spec Section 5.2).

For every classifier:
  (1) accuracy +/- std over 10 runs   (2) macro-F1 +/- std   (3) per-class F1
  (4) 8x8 confusion matrix            (5) one-vs-rest ROC + AUC per class
  (6) macro ROC-AUC                   (7) binary attack-vs-normal FNR
  (8) comparison with the fixed 11 % QBER threshold (and a CUSUM detector)

Run standalone to re-evaluate saved models and regenerate all figures:
    python training/evaluate.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from config import settings as S

LABEL_NAMES = [S.CLASS_LABELS[i] for i in range(len(S.CLASS_LABELS))]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def binary_metrics(is_attack_true: np.ndarray, is_attack_pred: np.ndarray) -> dict:
    t = np.asarray(is_attack_true, bool)
    p = np.asarray(is_attack_pred, bool)
    tp, fn = int(np.sum(t & p)), int(np.sum(t & ~p))
    tn, fp = int(np.sum(~t & ~p)), int(np.sum(~t & p))
    return {
        "accuracy": (tp + tn) / max(len(t), 1),
        "fnr": fn / max(fn + tp, 1),
        "fpr": fp / max(fp + tn, 1),
        "tpr": tp / max(fn + tp, 1),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, roc_points: int = 101) -> dict:
    y_true = np.asarray(y_true, int)
    y_pred = proba.argmax(axis=1)
    classes = list(range(len(LABEL_NAMES)))
    y_bin = label_binarize(y_true, classes=classes)
    roc = {}
    aucs = {}
    grid = np.linspace(0, 1, roc_points)
    for c in classes:
        fpr, tpr, _ = roc_curve(y_bin[:, c], proba[:, c])
        aucs[LABEL_NAMES[c]] = float(auc(fpr, tpr))
        roc[LABEL_NAMES[c]] = {"fpr": grid.tolist(),
                               "tpr": np.interp(grid, fpr, tpr).round(5).tolist()}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_f1": dict(zip(LABEL_NAMES, f1_score(y_true, y_pred, average=None,
                                                        labels=classes).round(4).tolist())),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "roc_auc_per_class": aucs,
        "macro_roc_auc": float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro")),
        "roc_curves": roc,
        "binary": binary_metrics(y_true != 0, y_pred != 0),
    }


def breaker_policy_metrics(y_true: np.ndarray, proba: np.ndarray,
                           normal_thr: float = S.NORMAL_CONFIDENCE_THRESHOLD,
                           attack_thr: float = S.ATTACK_CONFIDENCE_THRESHOLD) -> dict:
    """Apply the circuit-breaker decision rules to classifier probabilities.

    operational_fnr  = attack sessions that end in PROCEED (key actually released)
    false_abort_rate = normal sessions that end in ABORT_ATTACK
    key_yield_normal = normal sessions that end in PROCEED
    (single-session decisions, i.e. before any re-classification rounds)"""
    y_true = np.asarray(y_true)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    decision = np.where(pred == 0, np.where(conf > normal_thr, "PROCEED", "CAUTION"),
                        np.where(conf > attack_thr, "ABORT_ATTACK", "WARN"))
    attack = y_true != 0
    dist = {}
    for c in range(len(LABEL_NAMES)):
        sel = y_true == c
        dist[LABEL_NAMES[c]] = {d: float(np.mean(decision[sel] == d))
                                for d in ("PROCEED", "CAUTION", "WARN", "ABORT_ATTACK")}
    return {
        "operational_fnr": float(np.mean(decision[attack] == "PROCEED")),
        "false_abort_rate": float(np.mean(decision[~attack] == "ABORT_ATTACK")),
        "key_yield_normal": float(np.mean(decision[~attack] == "PROCEED")),
        "decision_distribution": dist,
    }


def threshold_baseline(X: np.ndarray, y: np.ndarray, threshold: float = S.FIXED_QBER_THRESHOLD) -> dict:
    """Conventional rule: attack iff mean QBER (feature 1) > 11 %."""
    pred_attack = np.asarray(X)[:, 0] > threshold
    out = binary_metrics(np.asarray(y) != 0, pred_attack)
    out["threshold"] = threshold
    # per-attack detection rate
    out["detection_rate_per_class"] = {
        LABEL_NAMES[c]: float(pred_attack[np.asarray(y) == c].mean()) for c in range(1, len(LABEL_NAMES))
    }
    return out


def cusum_statistic(seq: np.ndarray, mu0: float, k: float) -> float:
    s, smax = 0.0, 0.0
    for q in seq[~np.isnan(seq)]:
        s = max(0.0, s + q - mu0 - k)
        smax = max(smax, s)
    return smax


def fit_cusum(train_sequences: np.ndarray, train_labels: np.ndarray) -> dict:
    """One-sided CUSUM on the QBER series.  The reference level mu0 and slack k
    come from NORMAL training windows; the alarm level h is chosen on the
    training split to maximise Youden's J (TPR - FPR) - a generous baseline."""
    normal = train_sequences[train_labels == 0]
    vals = normal[~np.isnan(normal)]
    mu0, sigma = float(vals.mean()), float(vals.std())
    k = 0.5 * sigma
    stat = np.array([cusum_statistic(s, mu0, k) for s in train_sequences])
    attack = train_labels != 0
    best_h, best_j = 0.0, -1.0
    for h in np.unique(np.quantile(stat, np.linspace(0, 1, 401))):
        pred = stat > h
        j = pred[attack].mean() - pred[~attack].mean()
        if j > best_j:
            best_h, best_j = float(h), float(j)
    return {"mu0": mu0, "k": k, "h": best_h, "train_youden_j": best_j}


def cusum_baseline(sequences: np.ndarray, y: np.ndarray, params: dict) -> dict:
    stat = np.array([cusum_statistic(s, params["mu0"], params["k"]) for s in sequences])
    out = binary_metrics(np.asarray(y) != 0, stat > params["h"])
    out.update({"params": params})
    return out


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(metrics: dict, out_dir: str = str(S.FIGURES_DIR)) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    short = ["Normal", "I-R", "PNS", "B-S", "T-S", "Trojan", "Clone", "Fake"]
    for name, m in metrics.get("models", {}).items():
        test = m["test"]
        cm = np.array(test["confusion_matrix"])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cmn, annot=cm, fmt="d", cmap="Blues", xticklabels=short, yticklabels=short,
                    cbar_kws={"label": "row-normalised"}, ax=ax, vmin=0, vmax=1)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} - confusion matrix (acc {test['accuracy']:.3f})")
        fig.tight_layout()
        p = os.path.join(out_dir, f"confusion_{name}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

        fig, ax = plt.subplots(figsize=(7, 6))
        for cls, curve in test["roc_curves"].items():
            ax.plot(curve["fpr"], curve["tpr"],
                    label=f"{cls} (AUC {test['roc_auc_per_class'][cls]:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(f"{name} - one-vs-rest ROC (macro AUC {test['macro_roc_auc']:.3f})")
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        p = os.path.join(out_dir, f"roc_{name}.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    # detector comparison
    comp = metrics.get("comparison")
    if comp:
        names = list(comp.keys())
        acc = [comp[n]["accuracy"] * 100 for n in names]
        fnr = [comp[n]["fnr"] * 100 for n in names]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        b1 = ax.bar(x - 0.2, acc, 0.4, label="Detection accuracy (%)", color="#2b7bba")
        b2 = ax.bar(x + 0.2, fnr, 0.4, label="False negative rate (%)", color="#d1495b")
        ax.bar_label(b1, fmt="%.1f", fontsize=8)
        ax.bar_label(b2, fmt="%.1f", fontsize=8)
        ax.set_xticks(x, names)
        ax.set_ylim(0, 110)
        ax.set_title("Attack-vs-normal detection on the held-out test split")
        ax.legend()
        fig.tight_layout()
        p = os.path.join(out_dir, "detector_comparison.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    imp = metrics.get("global_shap_importance")
    if imp:
        df = pd.DataFrame(imp).head(20)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(data=df, y="feature", x="mean_abs_shap", hue="category", dodge=False, ax=ax)
        ax.set_title("Global feature importance (mean |SHAP|, XGBoost)")
        fig.tight_layout()
        p = os.path.join(out_dir, "shap_global_importance.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)

    runs = {n: m.get("runs") for n, m in metrics.get("models", {}).items() if m.get("runs")}
    if runs:
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(runs)
        means = [runs[n]["accuracy_mean"] * 100 for n in names]
        stds = [runs[n]["accuracy_std"] * 100 for n in names]
        bars = ax.bar(names, means, yerr=stds, capsize=6, color=["#2b7bba", "#59a14f", "#f28e2b"][:len(names)])
        ax.bar_label(bars, labels=[f"{m:.2f}±{s:.2f}" for m, s in zip(means, stds)], fontsize=8)
        ax.set_ylabel("8-class accuracy (%)")
        ax.set_ylim(min(means) - 10, 100)
        ax.set_title(f"Accuracy over {runs[names[0]]['n_runs']} independent runs")
        fig.tight_layout()
        p = os.path.join(out_dir, "model_accuracy_runs.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
def evaluate_saved_models(data_path: str = str(S.TRAINING_DATA_PATH)) -> dict:
    """Re-evaluate saved models on the test split and the independent test set."""
    from sklearn.model_selection import train_test_split
    from ml.attack_classifier import AttackClassifier, default_model_path, model_available

    df = pd.read_csv(data_path)
    X, y = df.iloc[:, :63].values, df["label"].values
    _, idx_te = train_test_split(np.arange(len(y)), test_size=S.TEST_SIZE, stratify=y,
                                 random_state=S.RANDOM_STATE)
    out = {}
    indep = None
    if os.path.exists(S.TEST_DATA_PATH):
        tdf = pd.read_csv(S.TEST_DATA_PATH)
        indep = (tdf.iloc[:, :63].values, tdf["label"].values)
    for mt in ("xgboost", "random_forest", "svm_rbf"):
        if not model_available(mt):
            continue
        clf = AttackClassifier(mt).load(default_model_path(mt))
        res = {"test_split": compute_metrics(y[idx_te], clf.predict_proba(X[idx_te]))}
        if indep is not None:
            res["independent_test"] = compute_metrics(indep[1], clf.predict_proba(indep[0]))
        out[mt] = res
        print(f"{mt:14s} split acc={res['test_split']['accuracy']:.4f} "
              f"FNR={res['test_split']['binary']['fnr']:.4f}"
              + (f" | independent acc={res['independent_test']['accuracy']:.4f}" if indep else ""))
    base = threshold_baseline(X[idx_te], y[idx_te])
    print(f"fixed-11% threshold: acc={base['accuracy']:.4f} FNR={base['fnr']:.4f}")
    out["threshold_baseline"] = base
    return out


def add_policy_metrics(metrics: dict, data_path: str = str(S.TRAINING_DATA_PATH)) -> dict:
    """Attach circuit-breaker policy metrics for the deployed XGBoost model."""
    from sklearn.model_selection import train_test_split
    from ml.attack_classifier import AttackClassifier, model_available
    if not model_available("xgboost"):
        return metrics
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, :63].values, df["label"].values
    _, idx_te = train_test_split(np.arange(len(y)), test_size=S.TEST_SIZE, stratify=y,
                                 random_state=S.RANDOM_STATE)
    clf = AttackClassifier("xgboost").load(str(S.XGBOOST_MODEL_PATH))
    pol = breaker_policy_metrics(y[idx_te], clf.predict_proba(X[idx_te]))
    metrics["circuit_breaker_policy"] = pol
    print(f"circuit breaker (XGBoost): operational FNR={pol['operational_fnr']:.4f} "
          f"false-abort={pol['false_abort_rate']:.4f} normal key yield={pol['key_yield_normal']:.4f}")
    return metrics


def simulate_breaker_with_resampling(n_per_class: int = 150, seed: int = 4242) -> dict:
    """End-to-end policy check on FRESH sessions drawn from the dataset
    distribution: each channel configuration (noise + attack parameters) is kept
    fixed while the breaker collects up to MAX_RECLASSIFY_ROUNDS extra sessions."""
    from quantum.bbm92_simulator import BBM92Session
    from security.circuit_breaker import CircuitBreaker
    from training.generate_dataset import sample_session_config

    breaker = CircuitBreaker(str(S.XGBOOST_MODEL_PATH), save_reports=False)
    breaker.shap_explainer.explain = lambda *a, **k: {            # SHAP not needed here
        "top_features": [], "base_value": 0.0, "explained_class": "", "interpretations": [],
        "full_shap_values": np.zeros(63)}
    master = np.random.default_rng(seed)
    dist, rounds = {}, {}
    for c in range(len(LABEL_NAMES)):
        counts = {d: 0 for d in ("PROCEED", "CAUTION", "WARN", "ABORT_ATTACK")}
        used = []
        for _ in range(n_per_class):
            rng = np.random.default_rng(int(master.integers(0, 2**62)))
            cfg = sample_session_config(c, rng)
            base_seed = int(rng.integers(0, 2**31 - 1000))

            def factory(r, cfg=cfg, base_seed=base_seed):
                return BBM92Session(**cfg, seed=base_seed + r).run()

            ev = breaker.evaluate_with_resampling(factory, True)
            counts[ev["decision"]] += 1
            used.append(ev["n_sessions_used"])
        dist[LABEL_NAMES[c]] = {k: v / n_per_class for k, v in counts.items()}
        rounds[LABEL_NAMES[c]] = float(np.mean(used))
    attack_proceed = np.mean([dist[n]["PROCEED"] for n in LABEL_NAMES[1:]])
    out = {"n_per_class": n_per_class, "max_extra_rounds": S.MAX_RECLASSIFY_ROUNDS,
           "decision_distribution": dist, "mean_sessions_used": rounds,
           "operational_fnr": float(attack_proceed),
           "false_abort_rate": dist["Normal"]["ABORT_ATTACK"],
           "key_yield_normal": dist["Normal"]["PROCEED"]}
    print(f"circuit breaker + resampling: operational FNR={out['operational_fnr']:.4f} "
          f"false-abort={out['false_abort_rate']:.4f} normal key yield={out['key_yield_normal']:.4f}")
    return out


def main():
    if not os.path.exists(S.METRICS_PATH):
        print("No metrics file yet - run `python main.py train` first.")
        return
    evaluate_saved_models()
    with open(S.METRICS_PATH) as fh:
        metrics = json.load(fh)
    metrics = add_policy_metrics(metrics)
    metrics["circuit_breaker_resampled"] = simulate_breaker_with_resampling()
    with open(S.METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=1, default=float)
    for p in make_figures(metrics):
        print("figure:", p)


if __name__ == "__main__":
    main()
