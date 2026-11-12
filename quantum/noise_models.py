"""
Four single-qubit quantum noise channels.

Each channel is defined by its Kraus operators {K_i} with  sum K_i^dag K_i = I
and acts as  rho -> sum_i K_i rho K_i^dag.

Because every qubit in the protocols lives in the X-Z plane of the Bloch
sphere, the effect of a channel can also be written in closed form as an
affine map on the Bloch vector r = (x, z).  The numpy fast path uses those
closed forms; the Qiskit path attaches the *same* Kraus operators to the
circuit.  Both are therefore statistically identical.

    bit_flip(p)          K0 = sqrt(1-p) I,  K1 = sqrt(p) X
                         (x, z) -> (x, (1-2p) z)
    phase_flip(p)        K0 = sqrt(1-p) I,  K1 = sqrt(p) Z
                         (x, z) -> ((1-2p) x, z)
    depolarizing(p)      rho -> (1-p) rho + p I/2
                         K0 = sqrt(1-3p/4) I, K1..3 = sqrt(p/4) {X, Y, Z}
                         (x, z) -> ((1-p) x, (1-p) z)
    amplitude_damping(g) K0 = [[1,0],[0,sqrt(1-g)]], K1 = [[0,sqrt(g)],[0,0]]
                         (x, z) -> (sqrt(1-g) x, (1-g) z + g)
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

VALID_NOISE_MODELS = (None, "bit_flip", "phase_flip", "amplitude_damping", "depolarizing")


def kraus_operators(noise_model: str | None, p: float) -> list[np.ndarray]:
    """Return the Kraus operators of the requested channel."""
    p = float(np.clip(p, 0.0, 1.0))
    if noise_model is None or p == 0.0:
        return [I2.copy()]
    if noise_model == "bit_flip":
        return [np.sqrt(1 - p) * I2, np.sqrt(p) * X]
    if noise_model == "phase_flip":
        return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]
    if noise_model == "depolarizing":
        return [np.sqrt(1 - 3 * p / 4) * I2, np.sqrt(p / 4) * X,
                np.sqrt(p / 4) * Y, np.sqrt(p / 4) * Z]
    if noise_model == "amplitude_damping":
        return [np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex),
                np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)]
    raise ValueError(f"Unknown noise model: {noise_model!r}")


def check_completeness(kraus: list[np.ndarray]) -> bool:
    """sum K^dag K == I  (trace preservation)."""
    s = sum(k.conj().T @ k for k in kraus)
    return bool(np.allclose(s, I2))


def apply_channel_bloch(x: np.ndarray, z: np.ndarray, noise_model: str | None,
                        p: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form action of the channel on Bloch-vector components (x, z)."""
    if noise_model is None or p == 0.0:
        return x, z
    if noise_model == "bit_flip":
        return x, (1 - 2 * p) * z
    if noise_model == "phase_flip":
        return (1 - 2 * p) * x, z
    if noise_model == "depolarizing":
        return (1 - p) * x, (1 - p) * z
    if noise_model == "amplitude_damping":
        return np.sqrt(1 - p) * x, (1 - p) * z + p
    raise ValueError(f"Unknown noise model: {noise_model!r}")


def apply_channel_density(rho: np.ndarray, noise_model: str | None, p: float) -> np.ndarray:
    """Apply the channel to a 2x2 density matrix using the Kraus form."""
    return sum(k @ rho @ k.conj().T for k in kraus_operators(noise_model, p))


def expected_qber(noise_model: str | None, p: float) -> float:
    """Average sifted-key error rate the channel induces in BBM92
    (half Z-basis states, half X-basis states, bits uniform)."""
    errs = []
    for theta in (0.0, np.pi / 2):
        for bit in (0, 1):
            ang = theta + bit * np.pi
            x, z = apply_channel_bloch(np.array([np.sin(ang)]), np.array([np.cos(ang)]),
                                       noise_model, p)
            # probability that measuring along the preparation axis gives the wrong bit
            n_dot_r = np.sin(ang) * x[0] + np.cos(ang) * z[0]
            errs.append((1 - n_dot_r) / 2)
    return float(np.mean(errs))


def qiskit_noise_error(noise_model: str | None, p: float):
    """Build a qiskit-aer QuantumError from the same Kraus operators."""
    if noise_model is None or p <= 0.0:
        return None
    from qiskit_aer.noise import kraus_error
    return kraus_error(kraus_operators(noise_model, p))
