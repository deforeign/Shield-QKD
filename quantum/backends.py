"""
Qubit stream representation and the two measurement back-ends.

Every qubit in flight is described by
    * state_angle  - angle (radians) of its preparation basis in the X-Z plane
                     of the Bloch sphere (0 = Z / rectilinear, pi/2 = X / diagonal)
    * state_bit    - which of the two basis states (0 -> +axis, 1 -> -axis)
    * extra_flip   - probability of an additional symmetric Pauli-Y flip that an
                     attack introduced (weak measurement, cloning, probing ...)
and optionally lost / detector-loss information.

Measuring a qubit along angle m gives outcome 0 with probability
    P(0) = (1 + n . r) / 2,   n = (sin m, cos m)
where r is the Bloch vector after the channel noise and the extra flip.

* NumpyBackend evaluates that formula directly and samples - very fast,
  used for the 24 000-session dataset.
* QiskitBackend builds the real circuits (X/H/Ry state preparation, Kraus
  noise channel, Pauli-Y error, basis rotation, measurement) and executes them
  on qiskit-aer.  Qubits that share an identical circuit are grouped and run as
  `shots=count, memory=True`, which is statistically identical to running each
  qubit as its own circuit but thousands of times faster.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantum.noise_models import apply_channel_bloch, qiskit_noise_error

BASIS_ANGLES = np.array([0.0, np.pi / 2])       # basis 0 = rectilinear (Z), 1 = diagonal (X)
_ANGLE_DECIMALS = 9


@dataclass
class QubitStream:
    """The quantum channel between Alice's source and Bob's detectors."""
    alice_bits: np.ndarray
    alice_angles: np.ndarray
    n_bob_bases: int = 2
    state_angle: np.ndarray = None
    state_bit: np.ndarray = None
    extra_flip: np.ndarray = None
    lost: np.ndarray = None
    detector_loss: np.ndarray | None = None     # shape (n, n_bob_bases) or None
    flags: dict = field(default_factory=dict)    # attack bookkeeping masks
    eve: dict = field(default_factory=dict)      # what Eve learned (for forensics)

    def __post_init__(self):
        n = len(self.alice_bits)
        if self.state_angle is None:
            self.state_angle = self.alice_angles.astype(float).copy()
        if self.state_bit is None:
            self.state_bit = self.alice_bits.astype(np.int8).copy()
        if self.extra_flip is None:
            self.extra_flip = np.zeros(n, dtype=float)
        if self.lost is None:
            self.lost = np.zeros(n, dtype=bool)

    @property
    def n(self) -> int:
        return len(self.alice_bits)

    def add_flip(self, mask: np.ndarray, p) -> None:
        """Compose an extra symmetric flip of probability p on the masked qubits
        (two independent flips p1, p2 compose to p1 + p2 - 2 p1 p2)."""
        e = self.extra_flip[mask]
        self.extra_flip[mask] = e + p - 2.0 * e * p

    def add_detector_loss(self, mask: np.ndarray, basis_idx: np.ndarray, p: float) -> None:
        if self.detector_loss is None:
            self.detector_loss = np.zeros((self.n, self.n_bob_bases))
        idx = np.flatnonzero(mask)
        self.detector_loss[idx, basis_idx] = np.maximum(self.detector_loss[idx, basis_idx], p)


def outcome_zero_probability(state_angle, state_bit, meas_angle, extra_flip,
                             noise_model, noise_prob) -> np.ndarray:
    """Exact Born-rule probability of measuring 0."""
    ang = np.asarray(state_angle, float) + np.asarray(state_bit, float) * np.pi
    x, z = np.sin(ang), np.cos(ang)
    x, z = apply_channel_bloch(x, z, noise_model, noise_prob)
    shrink = 1.0 - 2.0 * np.asarray(extra_flip, float)
    x, z = x * shrink, z * shrink
    m = np.asarray(meas_angle, float)
    return np.clip((1.0 + np.sin(m) * x + np.cos(m) * z) / 2.0, 0.0, 1.0)


class NumpyBackend:
    name = "numpy"

    def measure(self, state_angle, state_bit, meas_angle, extra_flip, noise_model,
                noise_prob, rng: np.random.Generator) -> np.ndarray:
        p0 = outcome_zero_probability(state_angle, state_bit, meas_angle, extra_flip,
                                      noise_model, noise_prob)
        return (rng.random(len(p0)) >= p0).astype(np.int8)

    def bell_alice_outcomes(self, alice_angles, rng: np.random.Generator) -> np.ndarray:
        """Alice measuring her half of |Phi+> gives a uniformly random bit."""
        return rng.integers(0, 2, len(alice_angles)).astype(np.int8)


class QiskitBackend:
    name = "qiskit"

    def __init__(self):
        from qiskit_aer import AerSimulator
        self._sim = AerSimulator()

    # -- circuit builders -------------------------------------------------
    @staticmethod
    def _prep(qc, angle: float):
        """Prepare the state whose Bloch vector sits at `angle` in the X-Z plane.
        Z/X basis states use the textbook X / H gates (spec 3.2.2 step 1)."""
        a = angle % (2 * np.pi)
        if np.isclose(a, 0.0):
            return
        if np.isclose(a, np.pi):
            qc.x(0)
        elif np.isclose(a, np.pi / 2):
            qc.h(0)                      # |+>
        elif np.isclose(a, 3 * np.pi / 2):
            qc.x(0)
            qc.h(0)                      # |->
        else:
            qc.ry(a, 0)

    @staticmethod
    def _rotate_for_measurement(qc, qubit: int, m: float):
        if np.isclose(m, 0.0):
            return
        if np.isclose(m, np.pi / 2):
            qc.h(qubit)
        else:
            qc.ry(-m, qubit)

    def _run_grouped(self, keys: list[tuple], builder, rng) -> dict:
        """Run one circuit per unique key with shots = multiplicity."""
        from qiskit import transpile
        uniq, inverse, counts = np.unique(np.array([repr(k) for k in keys]),
                                          return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        out = np.zeros(len(keys), dtype=np.int8)
        first_idx = {}
        for i, u in enumerate(inverse.tolist()):
            first_idx.setdefault(u, i)
        for g in range(len(uniq)):
            members = np.flatnonzero(inverse == g)
            qc = builder(keys[first_idx[g]])
            tqc = transpile(qc, self._sim, optimization_level=0)
            seed = int(rng.integers(0, 2**31 - 1))
            res = self._sim.run(tqc, shots=int(counts[g]), memory=True,
                                seed_simulator=seed).result()
            mem = res.get_memory(0)
            out[members] = np.array([int(s[-1]) for s in mem], dtype=np.int8)
        return out

    def measure(self, state_angle, state_bit, meas_angle, extra_flip, noise_model,
                noise_prob, rng: np.random.Generator) -> np.ndarray:
        from qiskit import QuantumCircuit
        from qiskit_aer.noise import pauli_error

        n = len(state_angle)
        if n == 0:
            return np.zeros(0, dtype=np.int8)
        noise_err = qiskit_noise_error(noise_model, noise_prob)
        total = np.round((np.asarray(state_angle) + np.asarray(state_bit) * np.pi) % (2 * np.pi),
                         _ANGLE_DECIMALS)
        meas = np.round(np.asarray(meas_angle, float), _ANGLE_DECIMALS)
        flips = np.round(np.asarray(extra_flip, float), 6)
        keys = [(float(a), float(m), float(f)) for a, m, f in zip(total, meas, flips)]

        def build(key):
            a, m, f = key
            qc = QuantumCircuit(1, 1)
            self._prep(qc, a)
            if noise_err is not None:
                qc.append(noise_err.to_instruction(), [0])
            if f > 0:
                qc.append(pauli_error([("Y", f), ("I", 1 - f)]).to_instruction(), [0])
            self._rotate_for_measurement(qc, 0, m)
            qc.measure(0, 0)
            return qc

        return self._run_grouped(keys, build, rng)

    def bell_alice_outcomes(self, alice_angles, rng: np.random.Generator) -> np.ndarray:
        """Prepare |Phi+> = (|00> + |11>)/sqrt2 and let Alice measure qubit 0."""
        from qiskit import QuantumCircuit
        keys = [(float(np.round(a, _ANGLE_DECIMALS)),) for a in alice_angles]

        def build(key):
            qc = QuantumCircuit(2, 1)
            qc.h(0)
            qc.cx(0, 1)
            self._rotate_for_measurement(qc, 0, key[0])
            qc.measure(0, 0)
            return qc

        return self._run_grouped(keys, build, rng)


def get_backend(name: str = "numpy"):
    if name == "numpy":
        return NumpyBackend()
    if name == "qiskit":
        return QiskitBackend()
    raise ValueError(f"Unknown backend {name!r}; choose 'numpy' or 'qiskit'")
