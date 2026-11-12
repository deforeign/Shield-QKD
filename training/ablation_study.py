"""
Feature-group ablation study.

For each of the 5 feature groups the XGBoost model (tuned hyper-parameters,
or sensible defaults) is retrained
    (a) WITHOUT that group  -> how much does accuracy drop?
    (b) with ONLY that group -> how much does it carry on its own?
Results: results/ablation_v1.json and results/figures/ablation.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import settings as S
from ml.feature_engine import FEATURE_GROUPS
from training.evaluate import binary_metrics


def _xgb(params: dict):
    from xgboost import XGBClassifier
    base = dict(objective="multi:softprob", num_class=len(S.CLASS_LABELS), tree_method="hist",
                n_jobs=-1, random_state=S.RANDOM_STATE, n_estimators=200, max_depth=6,
                learning_rate=0.1)
    base.update(params)
    return XGBClassifier(**base)


def _fit_eval(Xtr, ytr, Xte, yte, cols, params) -> dict:
    sc = StandardScaler().fit(Xtr[:, cols])
    m = _xgb(params).fit(sc.transform(Xtr[:, cols]), ytr)
    p = m.predict(sc.transform(Xte[:, cols]))
    return {"n_features": len(cols), "accuracy": float(accuracy_score(yte, p)),
            "macro_f1": float(f1_score(yte, p, average="macro")),
            "fnr": binary_metrics(yte != 0, p != 0)["fnr"],
            "per_class_f1": f1_score(yte, p, average=None).round(4).tolist()}


def run_ablation(data_path: str = str(S.TRAINING_DATA_PATH)) -> dict:
    df = pd.read_csv(data_path)
    X, y = df.iloc[:, :63].values, df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=S.TEST_SIZE, stratify=y,
                                          random_state=S.RANDOM_STATE)
    params = {}
    if os.path.exists(S.METRICS_PATH):
        with open(S.METRICS_PATH) as fh:
            params = json.load(fh).get("models", {}).get("xgboost", {}).get("cv", {}).get("best_params", {})
    all_cols = list(range(63))
    out = {"params": params, "full": _fit_eval(Xtr, ytr, Xte, yte, all_cols, params),
           "without": {}, "only": {}}
    print(f"[ablation] all 63 features      acc={out['full']['accuracy']:.4f}")
    for g, cols in FEATURE_GROUPS.items():
        rest = [c for c in all_cols if c not in cols]
        out["without"][g] = _fit_eval(Xtr, ytr, Xte, yte, rest, params)
        out["only"][g] = _fit_eval(Xtr, ytr, Xte, yte, cols, params)
        print(f"[ablation] {g:20s} without: acc={out['without'][g]['accuracy']:.4f} "
              f"(drop {out['full']['accuracy'] - out['without'][g]['accuracy']:+.4f})   "
              f"only: acc={out['only'][g]['accuracy']:.4f}")
    with open(S.ABLATION_PATH, "w") as fh:
        json.dump(out, fh, indent=1)
    _plot(out)
    return out


def _plot(out: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = list(out["without"])
    full = out["full"]["accuracy"] * 100
    wo = [out["without"][g]["accuracy"] * 100 for g in groups]
    only = [out["only"][g]["accuracy"] * 100 for g in groups]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - 0.2, wo, 0.4, label="without group", color="#2b7bba")
    b2 = ax.bar(x + 0.2, only, 0.4, label="only this group", color="#f28e2b")
    ax.bar_label(b1, fmt="%.1f", fontsize=8)
    ax.bar_label(b2, fmt="%.1f", fontsize=8)
    ax.axhline(full, color="k", ls="--", lw=1, label=f"all 63 features ({full:.1f}%)")
    ax.set_xticks(x, [g.replace("_", "\n") for g in groups])
    ax.set_ylabel("8-class accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Feature-group ablation (XGBoost)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    os.makedirs(S.FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(S.FIGURES_DIR, "ablation.png"), dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Feature group ablation study")
    ap.add_argument("--data", default=str(S.TRAINING_DATA_PATH))
    args = ap.parse_args(argv)
    run_ablation(args.data)


if __name__ == "__main__":
    main()
