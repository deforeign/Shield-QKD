"""
Generate the balanced BBM92 session dataset (spec Section 9.1).

    8 classes x 3000 sessions = 24 000 sessions (default)

Each session: run BBM92 (fast numpy backend) with the class' attack and a
randomly drawn noise channel -> temporal QBER (W=50) -> 63 features.

Outputs
    data/training_data.csv        feature_1 ... feature_63, label
    data/training_sequences.npz   per-session QBER sequences (for CUSUM baseline)
    data/training_sessions.csv    per-session generation parameters (analysis)
With --test an independent held-out set (different seed) is written to
data/test_data.csv (+ matching sequences / sessions files).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from config import settings as S
from ml.feature_engine import FEATURE_NAMES, extract_session_features
from quantum.bbm92_simulator import BBM92Session


def sample_session_config(label: int, rng: np.random.Generator) -> dict:
    spec = S.DATASET_CLASS_SPECS[label]
    params = {}
    for k, rngspec in spec["params"].items():
        if rngspec == "biased":
            v = rng.uniform(0.0, 0.25)
            params[k] = float(v if rng.random() < 0.5 else 1.0 - v)
        elif k == "burst_len":
            params[k] = int(rng.integers(rngspec[0], rngspec[1] + 1))
        else:
            params[k] = float(rng.uniform(*rngspec))
    return {
        "n_qubits": S.DEFAULT_N_QUBITS,
        "noise_model": str(rng.choice(S.DATASET_NOISE_MODELS)),
        "noise_prob": float(rng.uniform(*spec["noise"])),
        "attack_type": spec["attack"],
        "attack_params": params,
        "channel_loss": float(rng.uniform(*S.DATASET_CHANNEL_LOSS_RANGE)),
    }


def _simulate_one(label: int, seed: int):
    rng = np.random.default_rng(seed)
    cfg = sample_session_config(label, rng)
    result = BBM92Session(**cfg, backend="numpy", seed=int(rng.integers(0, 2**31 - 1))).run()
    x, td = extract_session_features(result)
    info = {"label": label, "seed": seed, "noise_model": cfg["noise_model"],
            "noise_prob": cfg["noise_prob"], "channel_loss": cfg["channel_loss"],
            "overall_qber": result["overall_qber"],
            "photon_loss_rate": result["photon_loss_rate"], "n_sifted": result["n_sifted"]}
    for k, v in cfg["attack_params"].items():
        info[k] = v
    return x, td["qber_sequence"], info


def _simulate_chunk(jobs):
    return [_simulate_one(lbl, sd) for lbl, sd in jobs]


def generate(sessions_per_class: int, seed: int, out_csv: str, seq_path: str,
             sessions_csv: str, n_jobs: int = -1) -> pd.DataFrame:
    master = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(S.CLASS_LABELS)), sessions_per_class)
    seeds = master.integers(0, 2**62, labels.size)
    jobs = list(zip(labels.tolist(), seeds.tolist()))
    chunk = 200
    chunks = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    t0 = time.time()
    print(f"[dataset] simulating {len(jobs)} BBM92 sessions on {n_jobs} workers ...", flush=True)
    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_simulate_chunk)(c) for c in chunks)
    flat = [r for part in results for r in part]
    X = np.vstack([r[0] for r in flat])
    seqs = [r[1] for r in flat]
    infos = pd.DataFrame([r[2] for r in flat])

    cols = [f"feature_{i + 1}" for i in range(S.N_FEATURES)]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = labels
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    max_t = max(len(s) for s in seqs)
    padded = np.full((len(seqs), max_t), np.nan)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    np.savez_compressed(seq_path, sequences=padded, labels=labels,
                        feature_names=np.array(FEATURE_NAMES))
    infos.to_csv(sessions_csv, index=False)
    print(f"[dataset] wrote {out_csv} ({len(df)} rows) in {time.time() - t0:.1f}s", flush=True)
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the ShieldQKD training dataset")
    ap.add_argument("--sessions-per-class", type=int, default=S.SESSIONS_PER_CLASS)
    ap.add_argument("--seed", type=int, default=S.DATASET_SEED)
    ap.add_argument("--test", action="store_true",
                    help="also generate the independent held-out test_data.csv")
    ap.add_argument("--test-sessions-per-class", type=int, default=S.TEST_SESSIONS_PER_CLASS)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    generate(args.sessions_per_class, args.seed, str(S.TRAINING_DATA_PATH),
             str(S.TRAINING_SEQUENCES_PATH), str(S.DATA_DIR / "training_sessions.csv"),
             args.n_jobs)
    if args.test:
        generate(args.test_sessions_per_class, S.TEST_DATASET_SEED, str(S.TEST_DATA_PATH),
                 str(S.TEST_SEQUENCES_PATH), str(S.DATA_DIR / "test_sessions.csv"), args.n_jobs)


if __name__ == "__main__":
    main()
