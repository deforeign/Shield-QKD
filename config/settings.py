"""
ShieldQKD - central configuration.

Every tunable parameter of the system lives here so that the simulator, the
feature engine, the training pipeline, the circuit breaker and the dashboard
all agree on the same numbers.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
TRAINING_SEQUENCES_PATH = DATA_DIR / "training_sequences.npz"
TEST_SEQUENCES_PATH = DATA_DIR / "test_sequences.npz"

XGBOOST_MODEL_PATH = MODELS_DIR / "xgboost_v1.json"
RF_MODEL_PATH = MODELS_DIR / "random_forest_v1.pkl"
SVM_MODEL_PATH = MODELS_DIR / "svm_rbf_v1.pkl"
SCALER_PATH = MODELS_DIR / "scaler_v1.pkl"
METRICS_PATH = RESULTS_DIR / "metrics_v1.json"
ABLATION_PATH = RESULTS_DIR / "ablation_v1.json"

for _d in (MODELS_DIR, DATA_DIR, REPORTS_DIR, RESULTS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Quantum session defaults
# --------------------------------------------------------------------------
DEFAULT_N_QUBITS = 5000
DEFAULT_CHANNEL_LOSS = 0.01          # baseline fibre/detector loss (fraction of photons)
NOISE_MODELS = ["bit_flip", "phase_flip", "amplitude_damping", "depolarizing"]
ATTACK_TYPES = [
    "intercept_resend", "pns", "beam_splitting", "time_shift",
    "trojan_horse", "quantum_cloning", "fake_state",
]
PROTOCOL = "BBM92"                   # the only protocol ShieldQKD runs
BACKENDS = ["numpy", "qiskit"]

# Default attack parameters (used when a parameter is not supplied)
DEFAULT_ATTACK_PARAMS = {
    "intercept_resend": {"interception_rate": 0.40},
    "pns": {"multi_photon_prob": 0.20},
    "beam_splitting": {"splitting_ratio": 0.30},
    "time_shift": {"shift_prob": 0.20, "burst_len": 10},
    "trojan_horse": {"probe_rate": 0.20},
    "quantum_cloning": {"clone_fidelity": 0.70},
    "fake_state": {"fake_rate": 0.30, "basis_bias": 0.90},
}

# --------------------------------------------------------------------------
# Attack physics constants (Section 10 of the spec)
# --------------------------------------------------------------------------
# PNS: single-photon pulses Eve cannot split are blocked with probability
#      PNS_BLOCK_FACTOR * multi_photon_prob  -> raises loss, adds no errors.
PNS_BLOCK_FACTOR = 1.0
# Beam splitting: a tapped qubit is lost with probability BS_TAP_LOSS; the
# surviving tapped qubits suffer a weak-measurement flip.  The flip rate is
# calibrated (see attack_models.loss_compensated_error) so that the QBER
# contribution in the SIFTED key is exactly splitting_ratio * BS_QBER_FACTOR.
BS_TAP_LOSS = 0.30
BS_QBER_FACTOR = 0.10
# Time shift: inside a shifted burst one of Bob's detectors (basis-dependent
# efficiency) loses TS_DETECTOR_LOSS of photons and errors cluster; the burst
# error rate is calibrated so the sifted-key QBER contribution is exactly
# shift_prob * TS_QBER_FACTOR.
TS_QBER_FACTOR = 0.20
TS_DETECTOR_LOSS = 0.50
TS_EPISODE_QUBITS = 150              # mean length (raw qubits) of a shifted episode
# Trojan horse: probes come in temporally correlated sequences.
TROJAN_ERROR = 0.15
TROJAN_SEQ_QUBITS = 600              # mean length (raw qubits) of a probe sequence
# Quantum cloning: per-bit error (1 - F) / 2 on ALL qubits (no constant needed).
# Fake state: Eve measures in a biased basis and re-sends a faked state.

# --------------------------------------------------------------------------
# Temporal QBER
# --------------------------------------------------------------------------
WINDOW_SIZE = 50
MULTISCALE_WINDOWS = (25, 100)
N_FEATURES = 63

# --------------------------------------------------------------------------
# Dataset generation (Section 9.1)
# --------------------------------------------------------------------------
SESSIONS_PER_CLASS = 3000
TEST_SESSIONS_PER_CLASS = 500        # independent held-out set (test_data.csv)
DATASET_SEED = 2026
TEST_DATASET_SEED = 7777
DATASET_CHANNEL_LOSS_RANGE = (0.0, 0.02)
# Noise channel drawn uniformly at random for every session.
DATASET_NOISE_MODELS = ["depolarizing", "bit_flip", "phase_flip", "amplitude_damping"]
DATASET_CLASS_SPECS = {
    0: {"attack": None, "noise": (0.0, 0.05), "params": {}},
    1: {"attack": "intercept_resend", "noise": (0.0, 0.03),
        "params": {"interception_rate": (0.05, 0.40)}},
    2: {"attack": "pns", "noise": (0.0, 0.03),
        "params": {"multi_photon_prob": (0.05, 0.30)}},
    3: {"attack": "beam_splitting", "noise": (0.0, 0.03),
        "params": {"splitting_ratio": (0.05, 0.40)}},
    4: {"attack": "time_shift", "noise": (0.0, 0.03),
        "params": {"shift_prob": (0.05, 0.30), "burst_len": (5, 20)}},
    5: {"attack": "trojan_horse", "noise": (0.0, 0.03),
        "params": {"probe_rate": (0.05, 0.30)}},
    6: {"attack": "quantum_cloning", "noise": (0.0, 0.03),
        "params": {"clone_fidelity": (0.50, 0.85)}},
    # basis_bias is drawn from the *biased* region [0, .25] U [.75, 1]
    # ("Eve prepares a qubit in a FIXED or BIASED basis", Section 10.7).
    7: {"attack": "fake_state", "noise": (0.0, 0.03),
        "params": {"fake_rate": (0.05, 0.40), "basis_bias": "biased"}},
}

# --------------------------------------------------------------------------
# ML
# --------------------------------------------------------------------------
CLASS_LABELS = {
    0: "Normal",
    1: "Intercept-Resend",
    2: "PNS",
    3: "Beam-Splitting",
    4: "Time-Shift",
    5: "Trojan-Horse",
    6: "Quantum-Cloning",
    7: "Fake-State",
}
ATTACK_TO_LABEL = {
    None: 0, "intercept_resend": 1, "pns": 2, "beam_splitting": 3,
    "time_shift": 4, "trojan_horse": 5, "quantum_cloning": 6, "fake_state": 7,
}
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_FOLDS = 5
N_EVAL_RUNS = 10

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
}
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.1],
}

# --------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------
ATTACK_CONFIDENCE_THRESHOLD = 0.70
NORMAL_CONFIDENCE_THRESHOLD = 0.85
FIXED_QBER_THRESHOLD = 0.11
MAX_RECLASSIFY_ROUNDS = 3            # extra sessions collected when uncertain

# --------------------------------------------------------------------------
# PQC / crypto
# --------------------------------------------------------------------------
LAMPORT_HASH_BITS = 256
LAMPORT_KEYPAIRS_PER_PARTY = 8       # one key per signed message (one-time!)
AES_KEY_BYTES = 32                   # AES-256
