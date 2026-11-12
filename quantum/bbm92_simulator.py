"""
BBM92 - entanglement-based QKD (the protocol ShieldQKD runs).

A source emits |Phi+> = (|00> + |11>)/sqrt2 and sends one photon to Alice and
one to Bob.  Each measures in a random rectilinear (Z) or diagonal (X) basis.

    Step 1  Alice measures her photon in a random basis (on the Qiskit path this
            is a real Bell-pair circuit: H, CNOT, basis rotation, measurement)
    Step 2  Channel to Bob: baseline loss, optional attack, noise channel
    Step 3  Bob measures in an independently random basis
    Step 4  Sifting: keep positions where the bases match (and the photon arrived)
    Step 5  Metadata: QBER, photon loss rate ...

For |Phi+> and real (X-Z plane) measurement bases, Alice obtaining outcome s
in basis a projects Bob's photon onto exactly the state |a, s>.  Letting Alice
measure first is therefore exact; Bob's photon then crosses the noisy /
attacked channel, and Eve attacks it on its way to Bob.

backend='numpy'  -> fast analytic Born-rule sampling (dataset generation)
backend='qiskit' -> real Qiskit-Aer circuits (live demo / single sessions)
"""
from __future__ import annotations

import numpy as np

from config import settings as S
from quantum.attack_models import apply_attack
from quantum.backends import BASIS_ANGLES, QubitStream, get_backend
from quantum.noise_models import VALID_NOISE_MODELS


def _validate(noise_model, noise_prob, attack_type):
    if noise_model not in VALID_NOISE_MODELS:
        raise ValueError(f"noise_model must be one of {VALID_NOISE_MODELS}")
    if not 0.0 <= noise_prob <= 0.5:
        raise ValueError("noise_prob must be in [0, 0.5]")
    if attack_type is not None and attack_type not in S.ATTACK_TYPES:
        raise ValueError(f"attack_type must be None or one of {S.ATTACK_TYPES}")


def bob_detect_and_measure(stream: QubitStream, bob_angles: np.ndarray, bob_basis_idx: np.ndarray,
                           noise_model, noise_prob, rng, backend) -> tuple[np.ndarray, np.ndarray]:
    """Apply detector loss, then measure every photon that arrived.
    Returns (bob_bits with -1 for lost photons, lost mask)."""
    lost = stream.lost.copy()
    if stream.detector_loss is not None:
        p_loss = stream.detector_loss[np.arange(stream.n), bob_basis_idx]
        lost |= rng.random(stream.n) < p_loss
    bob_bits = np.full(stream.n, -1, dtype=np.int8)
    rx = np.flatnonzero(~lost)
    bob_bits[rx] = backend.measure(stream.state_angle[rx], stream.state_bit[rx], bob_angles[rx],
                                   stream.extra_flip[rx], noise_model, noise_prob, rng)
    return bob_bits, lost


def summarise_flags(stream: QubitStream) -> dict:
    return {k: int(np.asarray(v).sum()) for k, v in stream.flags.items()}


def build_result(*, alice_bits, alice_bases, bob_bits, bob_bases, matching, lost, stream,
                 params: dict, backend_name: str) -> dict:
    n = len(alice_bits)
    matching_indices = np.flatnonzero(matching)
    sifted_alice = alice_bits[matching_indices].astype(np.int8)
    sifted_bob = bob_bits[matching_indices].astype(np.int8)
    n_sifted = int(matching_indices.size)
    qber = float(np.mean(sifted_alice != sifted_bob)) if n_sifted else 0.0
    loss_rate = float(lost.mean())
    metadata = {
        "protocol": "BBM92",
        "backend": backend_name,
        "n_qubits": n,
        "n_received": int((~lost).sum()),
        "n_sifted": n_sifted,
        "photon_loss_rate": loss_rate,
        "overall_qber": qber,
        "attack_flags": summarise_flags(stream),
        # raw-index bookkeeping used for per-window loss features
        "matching_indices": matching_indices,
        "lost_mask": lost,
        **params,
    }
    return {
        "alice_bits": alice_bits.astype(np.int8),
        "alice_bases": alice_bases.astype(np.int8),
        "bob_bits": bob_bits,
        "bob_bases": bob_bases.astype(np.int8),
        "sifted_alice": sifted_alice,
        "sifted_bob": sifted_bob,
        "matching_indices": matching_indices,
        "sifted_bases": alice_bases[matching_indices].astype(np.int8),
        "n_sifted": n_sifted,
        "overall_qber": qber,
        "photon_loss_rate": loss_rate,
        "lost_mask": lost,
        "metadata": metadata,
    }


class BBM92Session:
    def __init__(self, n_qubits: int = S.DEFAULT_N_QUBITS, noise_model: str = None,
                 noise_prob: float = 0.0, attack_type: str = None,
                 attack_params: dict = None, channel_loss: float = S.DEFAULT_CHANNEL_LOSS,
                 backend: str = "numpy", seed: int | None = None):
        """
        Args:
            n_qubits: number of entangled pairs emitted (default 5000).
            noise_model: None, 'bit_flip', 'phase_flip', 'amplitude_damping', 'depolarizing'.
            noise_prob: Probability parameter for the noise channel (0.0 to 0.5).
            attack_type: None or one of the 7 attacks (see quantum.attack_models).
            attack_params: attack-specific parameters (defaults in config.settings).
            channel_loss: baseline fraction of photons lost on Bob's arm.
            backend: 'numpy' (fast) or 'qiskit' (Qiskit-Aer circuits).
            seed: RNG seed for reproducibility.
        """
        _validate(noise_model, noise_prob, attack_type)
        self.n_qubits = int(n_qubits)
        self.noise_model = noise_model
        self.noise_prob = float(noise_prob)
        self.attack_type = attack_type
        self.attack_params = attack_params or {}
        self.channel_loss = float(channel_loss)
        self.backend_name = backend
        self.seed = seed

    def run(self) -> dict:
        """Execute a full BBM92 session.  Returns a dict with alice_bits,
        alice_bases (0 = rectilinear, 1 = diagonal), bob_bits (-1 = lost),
        bob_bases, sifted_alice, sifted_bob, matching_indices, sifted_bases,
        n_sifted, overall_qber, photon_loss_rate, lost_mask and metadata."""
        rng = np.random.default_rng(self.seed)
        backend = get_backend(self.backend_name)
        n = self.n_qubits

        # Step 1 - Alice measures her half of each |Phi+> pair
        alice_bases = rng.integers(0, 2, n).astype(np.int8)
        alice_angles = BASIS_ANGLES[alice_bases]
        alice_bits = backend.bell_alice_outcomes(alice_angles, rng)
        stream = QubitStream(alice_bits=alice_bits, alice_angles=alice_angles, n_bob_bases=2)

        # Step 2 - Bob's arm: baseline loss + attack (noise is applied at Bob's measurement)
        stream.lost |= rng.random(n) < self.channel_loss
        apply_attack(stream, self.attack_type, self.attack_params, rng, backend)

        # Step 3 - Bob measures in random bases
        bob_bases = rng.integers(0, 2, n).astype(np.int8)
        bob_bits, lost = bob_detect_and_measure(stream, BASIS_ANGLES[bob_bases], bob_bases,
                                                self.noise_model, self.noise_prob, rng, backend)

        # Step 4 - sifting;  Step 5 - metadata
        matching = (alice_bases == bob_bases) & ~lost
        params = {
            "noise_model": self.noise_model, "noise_prob": self.noise_prob,
            "attack_type": self.attack_type, "attack_params": dict(self.attack_params),
            "channel_loss": self.channel_loss, "seed": self.seed,
        }
        return build_result(alice_bits=alice_bits, alice_bases=alice_bases, bob_bits=bob_bits,
                            bob_bases=bob_bases, matching=matching, lost=lost, stream=stream,
                            params=params, backend_name=self.backend_name)
