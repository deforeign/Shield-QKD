# ShieldQKD

**Hybrid PQC-authenticated QKD with ML-powered temporal threat intelligence.**

ShieldQKD layers three defences on a quantum key distribution link:

1. **Post-quantum authentication of the classical channel.** Every basis announcement is signed with a
   Lamport one-time signature over SHA-256. A tampered, forged or replayed message triggers `ABORT_MITM`.
2. **Entanglement-based quantum key distribution (BBM92).** A source emits |Φ+⟩ pairs over a simulated
   photonic channel with 4 Kraus noise models and 7 eavesdropping attacks. There are two backends: real
   Qiskit-Aer circuits (including the Bell-pair circuit), and a fast numpy Born-rule sampler. The two are
   statistically identical.
3. **Temporal QBER threat intelligence.** The sifted key is cut into 50-bit windows. 63 physics-informed
   features are extracted from the windowed QBER series, and XGBoost classifies the session as Normal or
   one of 7 attacks. SHAP explains each verdict, and an attack-specific countermeasure is routed.

This replaces the conventional fixed 11 % QBER threshold, which misses most low-rate attacks.

Based on arXiv:2608.04047, *"Beyond the QBER Threshold: A Temporal QBER Based ML Framework for Multi
Attack Detection in BB84 QKD"*. The paper's framework is applied here to BBM92, the entanglement-based
counterpart of BB84. With matched Z/X bases, BBM92 produces the same sifted-key statistics.

---

## Results (this repository's trained models)

Training used 24 000 BBM92 sessions with a stratified 80/20 split and a full 5-fold grid search
(54 XGBoost candidates). Each model was then retrained over 10 independent splits and seeds. All numbers
come from `results/metrics_v1.json`.

**8-class classification**

| Model | CV accuracy | Test accuracy | Accuracy, 10 runs | Macro-F1, 10 runs | Macro ROC-AUC | Independent test set (4 000) |
|---|---|---|---|---|---|---|
| **XGBoost** (deployed) | 94.89 % | 94.96 % | 94.71 ± 0.29 % | 0.948 ± 0.003 | 0.997 | 95.08 % |
| Random Forest | 94.89 % | 94.94 % | 94.77 ± 0.32 % | 0.948 ± 0.003 | 0.997 | 95.00 % |
| SVM-RBF | 93.83 % | 94.60 % | 94.49 ± 0.28 % | 0.945 ± 0.003 | 0.995 | 94.95 % |

Per-class F1 (XGBoost):

| Normal | Intercept-Resend | PNS | Beam-Splitting | Time-Shift | Trojan-Horse | Quantum-Cloning | Fake-State |
|---|---|---|---|---|---|---|---|
| 0.948 | 0.832 | 0.993 | 0.986 | 0.998 | 0.997 | 0.906 | 0.942 |

**Attack vs normal, same test split**

| Detector | Detection accuracy | False negative rate |
|---|---|---|
| Fixed 11 % QBER threshold | 23.1 % | 87.9 % |
| CUSUM on the QBER series (alarm level tuned on training data) | 79.1 % | 23.6 % |
| **ShieldQKD XGBoost** | **98.7 %** | **0.83 %** |

The fixed threshold only catches strong quantum cloning (83 % of cloning sessions). It catches about 1 %
of intercept-resend and fake-state sessions and none of the PNS, beam-splitting, time-shift or Trojan
sessions. These baseline numbers reproduce the paper's 25.8 % accuracy and 84.8 % FNR closely.

**Circuit-breaker policy.** These are decisions, not just labels. The resampled numbers come from 150 fresh
channels per class, with up to 3 extra sessions collected when a verdict is uncertain:

| | Single session | With Bayesian resampling |
|---|---|---|
| Attacked channels that reach PROCEED (operational FNR) | 0.19 % | 0.95 % |
| Normal channels aborted as attacks | 1.2 % | 2.0 % |
| Normal channels that get a key | 85.3 % | 96.0 % (1.2 sessions on average) |

**Feature-group ablation** (XGBoost, 95.0 % with all features):

| Group | Accuracy without the group | Accuracy with only the group |
|---|---|---|
| Statistical | 94.9 % | 65.9 % |
| Burst & instability | 94.9 % | 73.9 % |
| Spectral & temporal | 94.8 % | 73.2 % |
| Basis-dependent | 83.3 % | 88.9 % |
| Channel interaction | 90.1 % | 83.5 % |

Basis-dependent and channel-interaction features carry the most information that no other group supplies.
The statistical, burst and spectral groups overlap heavily with each other.

**Known physical limit.** Intercept-resend at rate *r* and quantum cloning at fidelity *F* both produce
i.i.d., basis-symmetric bit errors. Where r/4 ≈ (1−F)/2, which happens for r ≈ 0.30–0.40 against
F ≈ 0.80–0.85 (QBER 7.5–10 %), the two attacks are statistically identical. This overlap is most of
the remaining confusion. In that region the breaker returns ABORT_ATTACK or WARN and never PROCEED.

---

## Quick start

```bash
pip install -r requirements.txt
python main.py generate-dataset --test      # 24 000 BBM92 sessions + 4 000 held-out sessions  (~1 min)
python main.py train                         # grid-search XGBoost / RF / SVM, 10 runs each
python main.py test                          # test-suite
python main.py demo                          # the 3-step evaluator demo (Qiskit backend)
python main.py dashboard                     # Streamlit UI
```

Other commands:

```bash
python main.py session --attack intercept_resend --rate 0.15 --noise depolarizing --noise-prob 0.01 -v
python main.py session --mitm tamper                      # classical MITM -> ABORT_MITM
python main.py session --message "hello" --backend qiskit # encrypt with the QKD key if PROCEED
python main.py session --encrypt-file report.pdf          # -> report.pdf.sqkd + Bob's report.decrypted.pdf
python main.py train --search random --n-iter 15          # faster randomised search
python main.py evaluate                                   # re-evaluate saved models, regenerate figures
python main.py ablation                                   # feature-group ablation study
```

> **Windows note:** if the project lives under a very long path, create the virtual environment
> somewhere short (e.g. `%USERPROFILE%\.venvs\shieldqkd`). Qiskit's deep package paths can exceed the
> 260-character limit otherwise.

---

## End-to-end flow (`security/pipeline.py`)

| Step | What happens | Module |
|---|---|---|
| 1 PQC handshake | 8 Lamport key pairs per party; public keys exchanged | `pqc/authenticated_sifting.py` |
| 2 Quantum transmission | Entangled pair emitted, Alice measures her photon; Bob's photon: channel loss → attack → noise, then Bob measures | `quantum/bbm92_simulator.py` |
| 3 Authenticated sifting | Bob's then Alice's bases, each signed with a fresh one-time key and verified | `pqc/` |
| 4 Temporal QBER | W = 50 windows → Q, Q_rect, Q_diag → 63 features | `ml/temporal_qber.py`, `ml/feature_engine.py` |
| 5 Circuit breaker | `PROCEED` / `CAUTION` / `WARN` / `ABORT_ATTACK` / `ABORT_MITM`, plus SHAP and forensics | `security/circuit_breaker.py` |
| 6 Application | Privacy amplification → HKDF → AES-256-GCM encryption of a message or file (CLI `--message` / `--encrypt-file`, dashboard upload); one-time keys, destroyed after use | `crypto/` |

Decision rules: normal with confidence > 0.85 → **PROCEED**; normal with ≤ 0.85 → **CAUTION**; attack with
> 0.70 → **ABORT_ATTACK**; attack with ≤ 0.70 → **WARN**. When a verdict is uncertain (CAUTION or WARN),
up to 3 extra independent sessions of the same size are collected. Their evidence is fused with Bayes' rule
(the normalised product of the per-session class probabilities) and the session is re-classified. This is
"increase sample size" without shifting the feature distribution the model was trained on. Plain averaging
was tried first. It cannot sharpen confidence, so it left most normal channels stuck in CAUTION.

---

## Physics model

**BBM92.** The source emits |Φ+⟩ = (|00⟩ + |11⟩)/√2. Alice and Bob each measure in a random rectilinear (Z)
or diagonal (X) basis and keep the rounds where the bases match. For |Φ+⟩ and real measurement bases,
Alice obtaining outcome *s* in basis *a* projects Bob's photon onto exactly |a, s⟩. The simulator lets Alice
measure first, which is exact. On the Qiskit backend this step is a real Bell-pair circuit (H, CNOT, basis
rotation, measurement). Bob's photon then crosses the noisy, attacked channel. Alice's outcomes are
uniformly random, yet in matching bases Bob's agree with them on a clean channel, as the tests check.

Qubit states live in the X-Z plane of the Bloch sphere. Measuring along angle *m* gives outcome 0 with
probability `(1 + n·r)/2`. The numpy backend evaluates this exactly. The Qiskit backend builds X/H/Ry
state preparation, the same Kraus channel, a Pauli-Y disturbance, the basis rotation and the measurement.
Identical circuits are grouped and run with `shots=count, memory=True`.

**Noise channels** (`quantum/noise_models.py`): bit-flip, phase-flip, depolarizing and amplitude damping,
each defined by explicit Kraus operators. A closed-form Bloch map is cross-checked against them in the tests.

**Attacks** (`quantum/attack_models.py`, spec Section 10). Eve attacks the photon on its way to Bob:

| Attack | Model | Signature |
|---|---|---|
| Intercept-resend | Eve measures a fraction *r* in a random basis and re-sends | uniform errors, QBER = r/4 |
| PNS | multi-photon emissions split without disturbance; single-photon ones blocked with probability *m* | ≈0 QBER, high loss |
| Beam splitting | tapped photons: 30 % lost, the rest weakly disturbed | QBER = 0.10·s plus proportional loss |
| Time-shift | clustered episodes (≈150 photons) of shifted `burst_len` blocks: errors and basis-dependent detector loss | QBER = 0.20·p, spikes, error–loss correlation |
| Trojan horse | long correlated probe sequences (≈600 photons) | QBER = 0.15·p, moderate bursts, high autocorrelation |
| Quantum cloning | every photon flips with probability (1−F)/2 | very uniform, QBER = (1−F)/2 |
| Fake state | measure-and-resend in a biased basis | QBER = 0.25·f, rectilinear vs diagonal asymmetry |

Every attack's sifted-key QBER matches the spec's Section 10 formula. `test_section10_expected_qber`
checks all seven. Beam splitting and time-shift also remove photons, and the lost photons are
disproportionately the disturbed ones. Their disturbance rate is therefore calibrated
(`loss_compensated_error`) so that the error that survives detection equals exactly `0.10·s` and `0.20·p`.

For the two temporally clustered attacks, the episodes are *placed* so that the realised duty cycle equals
`shift_prob` / `probe_rate` (`placed_episodes`). A free-running Markov chain was tried first, but at low
rates it often produced sessions where Eve never acted. Those were clean sessions carrying an attack label,
and that label noise cost about 4 points of accuracy.

---

## Dataset (spec Section 9.1)

There are 8 classes × 3 000 BBM92 sessions, each with 5 000 entangled pairs. The noise probability is
U(0, 0.05) for Normal and U(0, 0.03) for attacks. The noise channel is drawn at random from the four models,
and baseline loss is U(0, 0.02). The attack-parameter ranges are as specified. Fake-state `basis_bias` is
drawn from the *biased* region [0, 0.25] ∪ [0.75, 1]. Outputs:

* `data/training_data.csv`: `feature_1 … feature_63, label`
* `data/training_sequences.npz`: the raw QBER series, used by the CUSUM baseline
* `data/training_sessions.csv`: per-session generation parameters
* `data/test_data.csv`: an independent held-out set of 4 000 sessions with a different seed

---

## Project layout

```
config/settings.py          all tunable parameters
quantum/                    bbm92_simulator, noise_models, attack_models, backends (numpy / Qiskit-Aer)
pqc/                        lamport_ots, authenticated_sifting (+ simulate_mitm)
ml/                         temporal_qber, feature_engine (63 features), attack_classifier, shap_explainer
security/                   circuit_breaker, response_router, forensics, pipeline
crypto/                     key_manager (privacy amplification, one-time keys), file_vault (AES-256-GCM)
training/                   generate_dataset, train_models, evaluate, ablation_study
dashboard/                  Streamlit app (Live Demo, Training Results, Forensics Reports) + components
models/                     xgboost_v1.json, random_forest_v1.pkl, svm_rbf_v1.pkl, scaler_v1.pkl
results/                    metrics_v1.json, ablation_v1.json, training_log.txt, figures/*.png
reports/                    auto-generated forensics reports (.json / .txt / .pdf)
tests/                      pytest suite (Section 12 + extras)
```

## Implementation notes and deviations

* **Protocol scope.** The spec lists BB84, BBM92 and Ekert91 together with a protocol-comparison view. This
  build runs BBM92 only, and the comparison view has been removed. The response-routing table keeps the
  spec's wording ("switch to Ekert91", "Bell-CHSH test") as operator recommendations.
* **Scaler ownership.** `AttackClassifier` fits and stores its own `StandardScaler`, saved as
  `models/scaler_v1.pkl`. Callers always pass raw features. This avoids double scaling.
* **SVM probabilities.** scikit-learn 1.9 deprecated `SVC(probability=True)`. Platt scaling via
  `CalibratedClassifierCV(SVC, ensemble=False)` is used instead, which is the same method. SVM
  hyper-parameters are tuned on a stratified 6 000-sample subset (`--svm-tune-subsample`), and the best
  model is then refit on all training data.
* **SHAP.** `shap.TreeExplainer` is used. If the installed shap and xgboost versions are incompatible, the
  explainer falls back to XGBoost's exact built-in TreeSHAP (`pred_contribs=True`). SHAP values are in
  log-odds of the explained class.
* **Undefined features** (0/0, constant series) are 0.0, following the spec's `nan_to_num` rule.
  `effective_key_fraction` uses `estimated_leaked = n_sifted · (h(QBER) + photon_loss)`.
* **Key derivation.** Following the paper, the whole sifted key feeds the temporal analysis. On PROCEED the
  secure length is bounded by `n(1 − h(e) − loss) − 1.16·n·h(e)`, and at least 256 bits are required before
  HKDF-SHA256 derives the AES-256 key. Error correction is simulated (Bob reconciles to Alice's string).
