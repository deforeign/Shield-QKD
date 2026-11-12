"""
Train XGBoost, Random Forest and SVM-RBF (spec Section 9.2).

  1. load dataset                      5. evaluate all Section 5.2 metrics
  2. stratified 80/20 split            6. 10 independent runs -> mean +/- std
  3. StandardScaler fit on train only  7. save models + scaler
  4. 5-fold stratified CV tuning       8. fixed 11 % QBER threshold (+ CUSUM) baseline

    python training/train_models.py                    # full grid search (spec)
    python training/train_models.py --search random    # faster randomised search
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import settings as S
from ml.attack_classifier import AttackClassifier
from training.evaluate import (binary_metrics, compute_metrics, cusum_baseline, fit_cusum,
                               make_figures, threshold_baseline)

MODEL_FILES = {"xgboost": "xgboost_v1.json", "random_forest": "random_forest_v1.pkl",
               "svm_rbf": "svm_rbf_v1.pkl"}


def _repeated_runs(model_type: str, best_params: dict, X: np.ndarray, y: np.ndarray,
                   n_runs: int) -> dict:
    """Retrain the tuned model on n_runs different stratified splits / seeds."""
    accs, f1s, fnrs = [], [], []
    params = dict(best_params)
    if model_type == "svm_rbf":
        params["probability"] = False        # only hard predictions needed here
    for seed in range(n_runs):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=S.TEST_SIZE, stratify=y,
                                              random_state=1000 + seed)
        clf = AttackClassifier(model_type)
        clf.train(Xtr, ytr, search="none", params=params, random_state=1000 + seed)
        pred = clf.model.predict(clf.transform(Xte))
        accs.append(accuracy_score(yte, pred))
        f1s.append(f1_score(yte, pred, average="macro"))
        fnrs.append(binary_metrics(yte != 0, pred != 0)["fnr"])
        print(f"    run {seed + 1:2d}/{n_runs}: acc={accs[-1]:.4f} f1={f1s[-1]:.4f}", flush=True)
    return {"n_runs": n_runs,
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "fnr_mean": float(np.mean(fnrs)), "fnr_std": float(np.std(fnrs)),
            "accuracies": accs}


def train_all_models(data_path: str = str(S.TRAINING_DATA_PATH),
                     output_dir: str = str(S.MODELS_DIR),
                     models=("xgboost", "random_forest", "svm_rbf"), search: str = "grid",
                     n_iter: int = 20, n_runs: int = S.N_EVAL_RUNS,
                     svm_tune_subsample: int | None = 6000, verbose: int = 0) -> dict:
    t_start = time.time()
    # 1. Load dataset
    df = pd.read_csv(data_path)
    X = df.iloc[:, :63].values
    y = df["label"].values
    print(f"[train] {len(y)} sessions, {X.shape[1]} features, classes={np.bincount(y).tolist()}")

    # 2. Stratified train/test split (80/20) - split indices so sequences align
    idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=S.TEST_SIZE, stratify=y,
                                      random_state=S.RANDOM_STATE)
    X_train, X_test, y_train, y_test = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]

    # 3. StandardScaler is fitted on the training split inside AttackClassifier.train
    #    (and saved as models/scaler_v1.pkl).

    metrics = {"dataset": {"path": data_path, "n_sessions": int(len(y)),
                           "n_train": int(len(idx_tr)), "n_test": int(len(idx_te))},
               "search": search, "models": {}}

    # 4-7. Train, evaluate, repeat, save
    for mt in models:
        print(f"\n[train] === {mt} ({search} search, {S.CV_FOLDS}-fold stratified CV) ===", flush=True)
        t0 = time.time()
        clf = AttackClassifier(mt)
        cv = clf.train(X_train, y_train, cv_folds=S.CV_FOLDS, search=search, n_iter=n_iter,
                       tune_subsample=svm_tune_subsample if mt == "svm_rbf" else None,
                       verbose=verbose)
        print(f"[train] best params {cv['best_params']}  cv acc={cv['cv_accuracy']:.4f} "
              f"cv f1={cv['cv_f1']:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        test = compute_metrics(y_test, clf.predict_proba(X_test))
        print(f"[train] test acc={test['accuracy']:.4f} macro-F1={test['macro_f1']:.4f} "
              f"macro-AUC={test['macro_roc_auc']:.4f} FNR={test['binary']['fnr']:.4f}", flush=True)
        clf.save(os.path.join(output_dir, MODEL_FILES[mt]),
                 scaler_path=os.path.join(output_dir, "scaler_v1.pkl"))
        entry = {"cv": cv, "test": test, "train_seconds": time.time() - t0}

        if os.path.exists(S.TEST_DATA_PATH):
            tdf = pd.read_csv(S.TEST_DATA_PATH)
            indep = compute_metrics(tdf["label"].values, clf.predict_proba(tdf.iloc[:, :63].values))
            entry["independent_test"] = {k: indep[k] for k in
                                         ("accuracy", "macro_f1", "macro_roc_auc", "binary",
                                          "per_class_f1", "confusion_matrix")}
            print(f"[train] independent test set acc={indep['accuracy']:.4f}", flush=True)

        if n_runs and n_runs > 1:
            print(f"[train] {n_runs} independent runs ...", flush=True)
            entry["runs"] = _repeated_runs(mt, cv["best_params"], X, y, n_runs)
            r = entry["runs"]
            print(f"[train] accuracy {r['accuracy_mean']:.4f} +/- {r['accuracy_std']:.4f}, "
                  f"macro-F1 {r['macro_f1_mean']:.4f} +/- {r['macro_f1_std']:.4f}", flush=True)
        metrics["models"][mt] = entry

    # 8. Baselines on the same test split
    base = threshold_baseline(X_test, y_test)
    comparison = {"Fixed 11% QBER": {"accuracy": base["accuracy"], "fnr": base["fnr"],
                                     "fpr": base["fpr"]}}
    metrics["threshold_baseline"] = base
    seq_path = data_path.replace("_data.csv", "_sequences.npz")
    seq = np.load(seq_path)["sequences"] if seq_path != data_path and os.path.exists(seq_path) else None
    if seq is not None and len(seq) == len(y):
        cparams = fit_cusum(seq[idx_tr], y_train)
        cus = cusum_baseline(seq[idx_te], y_test, cparams)
        metrics["cusum_baseline"] = cus
        comparison["CUSUM"] = {"accuracy": cus["accuracy"], "fnr": cus["fnr"], "fpr": cus["fpr"]}
    for mt in models:
        b = metrics["models"][mt]["test"]["binary"]
        comparison[f"ML ({mt})"] = {"accuracy": b["accuracy"], "fnr": b["fnr"], "fpr": b["fpr"]}
    metrics["comparison"] = comparison
    print("\n[train] attack-vs-normal comparison on the test split:")
    for k, v in comparison.items():
        print(f"    {k:22s} accuracy={v['accuracy']:.4f}  FNR={v['fnr']:.4f}  FPR={v['fpr']:.4f}")

    # Global SHAP importance for the deployed (XGBoost) model
    if "xgboost" in models:
        try:
            from ml.shap_explainer import SHAPExplainer
            clf = AttackClassifier("xgboost").load(os.path.join(output_dir, MODEL_FILES["xgboost"]))
            imp = SHAPExplainer(clf).global_importance(X_test)
            metrics["global_shap_importance"] = imp.to_dict(orient="records")
        except Exception as exc:                        # pragma: no cover
            print("[train] SHAP global importance failed:", exc)

    # circuit-breaker policy metrics (uses the default model location)
    if "xgboost" in models and os.path.abspath(output_dir) == os.path.abspath(S.MODELS_DIR):
        from training.evaluate import add_policy_metrics
        add_policy_metrics(metrics, data_path)

    metrics["total_seconds"] = time.time() - t_start
    os.makedirs(os.path.dirname(S.METRICS_PATH), exist_ok=True)
    with open(S.METRICS_PATH, "w") as fh:
        json.dump(metrics, fh, indent=1, default=float)
    figs = make_figures(metrics)
    print(f"\n[train] metrics -> {S.METRICS_PATH}; {len(figs)} figures -> {S.FIGURES_DIR}")
    print(f"[train] done in {metrics['total_seconds'] / 60:.1f} min")
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train ShieldQKD attack classifiers")
    ap.add_argument("--data", default=str(S.TRAINING_DATA_PATH))
    ap.add_argument("--out", default=str(S.MODELS_DIR))
    ap.add_argument("--models", nargs="+", default=["xgboost", "random_forest", "svm_rbf"],
                    choices=["xgboost", "random_forest", "svm_rbf"])
    ap.add_argument("--search", choices=["grid", "random"], default="grid")
    ap.add_argument("--n-iter", type=int, default=20, help="candidates for --search random")
    ap.add_argument("--runs", type=int, default=S.N_EVAL_RUNS)
    ap.add_argument("--svm-tune-subsample", type=int, default=6000,
                    help="tune SVM hyper-parameters on a stratified subsample (0 = all data)")
    ap.add_argument("--verbose", type=int, default=0)
    args = ap.parse_args(argv)
    train_all_models(args.data, args.out, tuple(args.models), args.search, args.n_iter,
                     args.runs, args.svm_tune_subsample or None, args.verbose)


if __name__ == "__main__":
    main()
