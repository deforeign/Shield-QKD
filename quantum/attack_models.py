"""
The seven eavesdropping attacks (spec Section 10).

Every attack is a function  attack(stream, params, rng, backend) -> stream
that acts on the qubit stream BETWEEN Alice's preparation and Bob's
measurement.  Attacks that physically measure a qubit (intercept-resend,
fake-state) do so through the active backend, so on the Qiskit path Eve's
measurements are real circuit executions too.

Temporal signatures produced (what the ML layer learns to see):
    intercept_resend  uniform errors,       QBER += rate * 0.25
    pns               ~no errors, extra photon loss (key rate drops)
    beam_splitting    tiny uniform errors + proportional loss, QBER += ratio * 0.10
    time_shift        error BURSTS with basis-dependent detector loss, QBER += shift_prob * 0.20
    trojan_horse      QBER += probe_rate * 0.15
    trojan_horse      long, temporally correlated probe sequences
    quantum_cloning   errors on every qubit, (1 - F) / 2, extremely uniform
    fake_state        biased-basis re-sending -> rect/diag QBER asymmetry
"""
from __future__ import annotations

import numpy as np

from config import settings as S
from quantum.backends import BASIS_ANGLES, QubitStream


def markov_episodes(n_units: int, p: float, mean_on: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Two-state Markov (on/off) process whose stationary 'on' probability is p
    and whose mean 'on' run length is `mean_on` units.  Used for attacks whose
    activity is temporally clustered."""
    mask = np.zeros(n_units, dtype=bool)
    if p <= 0 or n_units == 0:
        return mask
    p = min(p, 0.95)
    mean_on = max(1.0, float(mean_on))
    mean_off = max(1.0, mean_on * (1.0 - p) / p)
    state = rng.random() < p
    pos = 0
    while pos < n_units:
        length = int(rng.geometric(1.0 / (mean_on if state else mean_off)))
        if state:
            mask[pos:pos + length] = True
        pos += length
        state = not state
    return mask


def placed_episodes(n_units: int, p: float, mean_on: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Temporally clustered activity whose realised duty cycle is ~p.

    Episode lengths are geometric with mean `mean_on`; episodes are drawn until
    they cover p * n_units and are then placed at random, non-overlapping
    positions.  Unlike a free-running Markov chain this guarantees that an
    attack with p > 0 is actually active in the session (no "silent" attacks
    that would be indistinguishable from a clean channel)."""
    mask = np.zeros(n_units, dtype=bool)
    target = int(round(min(max(p, 0.0), 0.95) * n_units))
    if target <= 0 or n_units == 0:
        return mask
    mean_on = max(1.0, float(mean_on))
    lengths = []
    total = 0
    while total < target:
        L = min(int(rng.geometric(1.0 / mean_on)), target - total)
        lengths.append(L)
        total += L
    k = len(lengths)
    rng.shuffle(lengths)
    off_total = n_units - total
    # random composition of the idle time into k + 1 gaps
    cuts = np.sort(rng.integers(0, off_total + 1, k))
    gaps = np.diff(np.concatenate([[0], cuts, [off_total]]))
    pos = 0
    for i, L in enumerate(lengths):
        pos += int(gaps[i])
        mask[pos:pos + L] = True
        pos += L
    return mask


def _eve_measure_and_resend(stream: QubitStream, mask: np.ndarray, p_rect: float,
                            rng, backend, flag: str) -> None:
    """Eve measures the masked qubits in Z (prob p_rect) or X and re-sends the
    state she observed."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        stream.flags[flag] = mask
        return
    eve_basis = (rng.random(idx.size) >= p_rect).astype(np.int8)   # 0 = rect, 1 = diag
    eve_angles = BASIS_ANGLES[eve_basis]
    eve_bits = backend.measure(stream.state_angle[idx], stream.state_bit[idx], eve_angles,
                               stream.extra_flip[idx], None, 0.0, rng)
    stream.state_angle[idx] = eve_angles
    stream.state_bit[idx] = eve_bits
    stream.extra_flip[idx] = 0.0          # measurement collapsed any earlier disturbance
    stream.flags[flag] = mask
    stream.eve.update({"n_measured": int(idx.size),
                       "eve_bases": eve_basis, "eve_bits": eve_bits, "indices": idx})


# --------------------------------------------------------------------------
# 10.1 Intercept-resend
# --------------------------------------------------------------------------
def intercept_resend(stream: QubitStream, params: dict, rng, backend) -> QubitStream:
    rate = float(params.get("interception_rate", 0.4))
    mask = rng.random(stream.n) < rate
    _eve_measure_and_resend(stream, mask, 0.5, rng, backend, "intercepted")
    return stream


# --------------------------------------------------------------------------
# 10.2 Photon number splitting
# --------------------------------------------------------------------------
def pns(stream: QubitStream, params: dict, rng, backend) -> QubitStream:
    m = float(params.get("multi_photon_prob", 0.2))
    multi = rng.random(stream.n) < m
    # Multi-photon pulses: Eve keeps one photon, the rest reaches Bob untouched.
    # Single-photon pulses she cannot split are blocked -> loss, no errors.
    blocked = (~multi) & (rng.random(stream.n) < S.PNS_BLOCK_FACTOR * m)
    stream.lost |= blocked
    stream.flags["multi_photon"] = multi
    stream.flags["blocked"] = blocked
    stream.eve["n_split_photons"] = int(multi.sum())
    return stream


# --------------------------------------------------------------------------
# 10.3 Beam splitting
# --------------------------------------------------------------------------
def loss_compensated_error(frac: float, loss: float, target_factor: float) -> float:
    """Flip rate for the affected qubits so that the QBER they add to the
    SIFTED key is exactly frac * target_factor, even though a fraction `loss`
    of the affected photons never reaches Bob.

    Affected share of detected qubits = frac (1 - loss) / (1 - frac * loss), so
        e = target_factor * (1 - frac * loss) / (1 - loss)."""
    if frac <= 0:
        return 0.0
    return float(min(0.5, target_factor * (1.0 - frac * loss) / (1.0 - loss)))


def beam_splitting(stream: QubitStream, params: dict, rng, backend) -> QubitStream:
    s = float(params.get("splitting_ratio", 0.3))
    tapped = rng.random(stream.n) < s
    lost = tapped & (rng.random(stream.n) < S.BS_TAP_LOSS)
    stream.lost |= lost
    err = loss_compensated_error(s, S.BS_TAP_LOSS, S.BS_QBER_FACTOR)
    stream.add_flip(tapped & ~lost, err)                   # weak measurement disturbance
    stream.flags["tapped"] = tapped
    stream.eve["n_tapped"] = int(tapped.sum())
    return stream


# --------------------------------------------------------------------------
# 10.4 Time shift
# --------------------------------------------------------------------------
def time_shift(stream: QubitStream, params: dict, rng, backend) -> QubitStream:
    p = float(params.get("shift_prob", 0.2))
    L = max(1, int(round(params.get("burst_len", 10))))
    n_blocks = int(np.ceil(stream.n / L))
    block_on = placed_episodes(n_blocks, p, S.TS_EPISODE_QUBITS / L, rng)
    shifted = np.repeat(block_on, L)[: stream.n]
    # In a shifted burst Bob's detector efficiency becomes basis dependent:
    # one basis' detector (chosen per burst) loses photons, and errors cluster.
    target_basis = np.repeat(rng.integers(0, stream.n_bob_bases, n_blocks), L)[: stream.n]
    # expected in-burst loss: Bob picks the targeted basis with prob 1/n_bob_bases
    burst_loss = S.TS_DETECTOR_LOSS / stream.n_bob_bases
    err = loss_compensated_error(float(shifted.mean()), burst_loss, S.TS_QBER_FACTOR)
    stream.add_flip(shifted, err)
    stream.add_detector_loss(shifted, target_basis[shifted], S.TS_DETECTOR_LOSS)
    stream.flags["shifted"] = shifted
    stream.eve["n_shifted"] = int(shifted.sum())
    return stream


# --------------------------------------------------------------------------
# 10.5 Trojan horse
# --------------------------------------------------------------------------
def trojan_horse(stream: QubitStream, params: dict, rng, backend) -> QubitStream:
    p = float(params.get("probe_rate", 0.2))
    probed = placed_episodes(stream.n, p, S.TROJAN_SEQ_QUBITS, rng)
    stream.add_flip(probed, S.TROJAN_ERROR)
    stream.flags["probed"] = probed
    stream.eve["n_probed"] = int(probed.sum())
    return stream


# --------------------------------------------------------------------------
# 10.6 Quantum cloning
# --------------------------------------------------------------------------
def quantum_cloning(stream: QubitStream, params: dict, rng, backend) -> QubitStream:
    F = float(np.clip(params.get("clone_fidelity", 0.7), 0.5, 1.0))
    all_q = np.ones(stream.n, dtype=bool)
    stream.add_flip(all_q, (1.0 - F) / 2.0)
    stream.flags["cloned"] = all_q
    stream.eve["clone_fidelity"] = F
    return stream


# --------------------------------------------------------------------------
# 10.7 Fake state
# --------------------------------------------------------------------------
def fake_state(stream: QubitStream, params: dict, rng, backend) -> QubitStream:
    f = float(params.get("fake_rate", 0.3))
    bias = float(np.clip(params.get("basis_bias", 0.9), 0.0, 1.0))   # P(Eve uses rectilinear)
    mask = rng.random(stream.n) < f
    _eve_measure_and_resend(stream, mask, bias, rng, backend, "faked")
    stream.eve["basis_bias"] = bias
    return stream


ATTACKS = {
    "intercept_resend": intercept_resend,
    "pns": pns,
    "beam_splitting": beam_splitting,
    "time_shift": time_shift,
    "trojan_horse": trojan_horse,
    "quantum_cloning": quantum_cloning,
    "fake_state": fake_state,
}


def apply_attack(stream: QubitStream, attack_type: str | None, params: dict | None,
                 rng, backend) -> QubitStream:
    if attack_type is None:
        return stream
    if attack_type not in ATTACKS:
        raise ValueError(f"Unknown attack {attack_type!r}; valid: {list(ATTACKS)}")
    merged = dict(S.DEFAULT_ATTACK_PARAMS.get(attack_type, {}))
    merged.update(params or {})
    return ATTACKS[attack_type](stream, merged, rng, backend)
