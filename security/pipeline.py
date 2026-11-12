"""
End-to-end ShieldQKD session (spec Section 1.3, steps 1-6).

    1  PQC handshake (Lamport public keys)
    2  quantum transmission (BBM92 entangled pairs, numpy or Qiskit)
    3  PQC-authenticated basis sifting (optional classical MITM injection)
    4  temporal QBER analysis (63 features, XGBoost)
    5  circuit-breaker decision (+ SHAP, forensics, countermeasure)
    6  application layer: AES-256 key on PROCEED, key destruction otherwise
"""
from __future__ import annotations

import time

import numpy as np

from config import settings as S
from crypto.key_manager import KeyManager
from pqc.authenticated_sifting import AuthenticatedSifting
from quantum import run_session
from security.circuit_breaker import CircuitBreaker

_BREAKER_CACHE: dict[str, CircuitBreaker] = {}


def get_breaker(path: str = str(S.XGBOOST_MODEL_PATH)) -> CircuitBreaker:
    if path not in _BREAKER_CACHE:
        _BREAKER_CACHE[path] = CircuitBreaker(path)
    return _BREAKER_CACHE[path]


def _destroy(session: dict) -> None:
    for k in ("sifted_alice", "sifted_bob", "alice_bits", "bob_bits"):
        arr = session.get(k)
        if isinstance(arr, np.ndarray):
            arr.fill(0)


def run_shieldqkd(n_qubits: int = S.DEFAULT_N_QUBITS,
                  noise_model: str | None = None, noise_prob: float = 0.0,
                  attack_type: str | None = None, attack_params: dict | None = None,
                  channel_loss: float = S.DEFAULT_CHANNEL_LOSS, mitm: str | None = None,
                  backend: str = "numpy", seed: int | None = None,
                  breaker: CircuitBreaker | None = None, key_manager: KeyManager | None = None,
                  resample: bool = True, pdf_report: bool = False) -> dict:
    timings = {}
    t0 = time.perf_counter()
    breaker = breaker or get_breaker()
    km = key_manager or KeyManager()

    # 1. PQC handshake
    sifter = AuthenticatedSifting()
    hs = sifter.handshake()
    timings["pqc_handshake_s"] = time.perf_counter() - t0

    # 2. quantum transmission
    def make_session(r: int) -> dict:
        return run_session(n_qubits=n_qubits, noise_model=noise_model,
                           noise_prob=noise_prob, attack_type=attack_type,
                           attack_params=attack_params, channel_loss=channel_loss,
                           backend=backend, seed=None if seed is None else seed + 1000 * r)

    t1 = time.perf_counter()
    session = make_session(0)
    timings["quantum_s"] = time.perf_counter() - t1

    # 3. PQC-authenticated sifting
    t2 = time.perf_counter()
    auth = sifter.authenticated_sift(session["alice_bases"], session["bob_bases"],
                                     session["lost_mask"], mitm=mitm)
    timings["pqc_sifting_s"] = time.perf_counter() - t2
    if auth["pqc_verified"]:
        expected = np.zeros(len(session["alice_bases"]), dtype=bool)
        expected[session["matching_indices"]] = True
        if not np.array_equal(auth["matching_mask"], expected):
            raise RuntimeError("authenticated sifting disagrees with the simulator")

    # 4-5. ML analysis + circuit breaker
    t3 = time.perf_counter()
    cache = {0: session}

    def factory(r: int) -> dict:
        if r not in cache:
            cache[r] = make_session(r)
        return cache[r]

    if resample and auth["pqc_verified"]:
        ev = breaker.evaluate_with_resampling(factory, True, auth["log"], pdf_report=pdf_report)
    else:
        ev = breaker.evaluate(session, auth["pqc_verified"], auth["log"], pdf_report=pdf_report)
    timings["ml_s"] = time.perf_counter() - t3

    # 6. application layer
    key_id, key_status = None, None
    if ev["decision"] == "PROCEED":
        try:
            key_id = km.derive_from_session(session["sifted_alice"], session["sifted_bob"],
                                            session["photon_loss_rate"])
            key_status = "AES-256 key derived (privacy-amplified, one-time use)"
        except ValueError as exc:
            key_status = f"key withheld: {exc}"
        _destroy(session)                 # raw sifted bits are never kept after derivation
    elif ev["decision"] == "CAUTION":
        key_status = "key withheld (uncertain verdict after resampling)"
        _destroy(session)
    else:
        _destroy(session)
        key_status = "KEY MATERIAL DESTROYED"
    timings["total_s"] = time.perf_counter() - t0
    return {"protocol": S.PROTOCOL, "session": session, "evaluation": ev, "pqc": auth,
            "handshake": {"keys_remaining": hs["keys_remaining"],
                          "alice_fingerprint": hs["alice_fingerprints"][0],
                          "bob_fingerprint": hs["bob_fingerprints"][0]},
            "key_id": key_id, "key_status": key_status, "key_manager": km, "timings": timings}
