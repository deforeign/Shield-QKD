"""
63 physics-informed features extracted from the temporal QBER series
(spec Section 4.2 and Section 11).  The ORDER of features is part of the model
contract - never reorder without retraining.

    1-13   statistical
    14-26  burst & instability
    27-38  spectral & temporal
    39-51  basis-dependent
    52-63  channel interaction
Undefined values (0/0, constant series ...) are mapped to 0.0.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import stats

from config import settings as S
from ml.temporal_qber import TemporalQBER

FEATURE_NAMES: list[str] = [
    # Category 1 - statistical (13)
    "qber_mean", "qber_std", "qber_variance", "qber_skewness", "qber_kurtosis",
    "qber_median", "qber_iqr", "qber_cv", "qber_min", "qber_max", "qber_range",
    "qber_p10", "qber_p90",
    # Category 2 - burst & instability (13)
    "spike_density", "burst_duration_mean", "burst_duration_max", "jump_energy",
    "temporal_drift", "drift_r_squared", "error_gap_mean", "error_gap_std",
    "error_gap_cv", "max_consecutive_low", "max_consecutive_high", "run_ratio",
    "abs_diff_mean",
    # Category 3 - spectral & temporal (12)
    "autocorr_lag1", "autocorr_lag2", "autocorr_lag5", "dft_energy", "dft_peak_freq",
    "dft_peak_magnitude", "spectral_entropy", "shannon_entropy", "hurst_exponent",
    "turning_point_rate", "mean_crossing_rate", "energy_ratio_low_high",
    # Category 4 - basis dependent (13)
    "qber_rect_mean", "qber_diag_mean", "basis_asymmetry", "basis_ratio",
    "qber_rect_std", "qber_diag_std", "basis_std_diff", "rect_spike_density",
    "diag_spike_density", "basis_spike_ratio", "rect_jump_energy", "diag_jump_energy",
    "basis_energy_diff",
    # Category 5 - channel interaction (12)
    "key_gen_rate", "photon_loss_rate", "transmission_efficiency",
    "qber_loss_correlation", "qber_stability", "multiscale_qber_25",
    "multiscale_qber_100", "multiscale_std_25", "multiscale_std_100",
    "scale_ratio_mean", "scale_ratio_std", "effective_key_fraction",
]
assert len(FEATURE_NAMES) == S.N_FEATURES

FEATURE_GROUPS = {
    "statistical": list(range(0, 13)),
    "burst_instability": list(range(13, 26)),
    "spectral_temporal": list(range(26, 38)),
    "basis_dependent": list(range(38, 51)),
    "channel_interaction": list(range(51, 63)),
}
FEATURE_CATEGORY = {FEATURE_NAMES[i]: g for g, idx in FEATURE_GROUPS.items() for i in idx}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else 0.0


def _runs(mask: np.ndarray) -> np.ndarray:
    """Lengths of consecutive True runs."""
    if mask.size == 0 or not mask.any():
        return np.zeros(0, dtype=int)
    padded = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(padded)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def _impute(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=float)
    if seq.size == 0:
        return seq
    if np.all(np.isnan(seq)):
        return np.zeros_like(seq)
    return np.where(np.isnan(seq), np.nanmean(seq), seq)


def spike_mask(q: np.ndarray) -> np.ndarray:
    if q.size == 0:
        return np.zeros(0, dtype=bool)
    sd = q.std()
    if sd == 0:
        return np.zeros(q.size, dtype=bool)
    return q > q.mean() + 2.0 * sd


def spike_density(q: np.ndarray) -> float:
    return float(spike_mask(q).mean()) if q.size else 0.0


def jump_energy(q: np.ndarray) -> float:
    return float(np.sum(np.diff(q) ** 2)) if q.size > 1 else 0.0


def autocorr(q: np.ndarray, lag: int) -> float:
    if q.size <= lag:
        return 0.0
    c = q - q.mean()
    denom = np.sum(c * c)
    if denom == 0:
        return 0.0
    return float(np.sum(c[:-lag] * c[lag:]) / denom)


def hurst_rs(q: np.ndarray) -> float:
    """Hurst exponent via rescaled-range (R/S) analysis."""
    T = q.size
    if T < 8:
        return 0.5
    sizes = np.unique(np.floor(np.logspace(np.log10(4), np.log10(T // 2), 6)).astype(int))
    log_n, log_rs = [], []
    for n in sizes:
        k = T // n
        if k < 1 or n < 2:
            continue
        chunks = q[: k * n].reshape(k, n)
        dev = chunks - chunks.mean(axis=1, keepdims=True)
        z = np.cumsum(dev, axis=1)
        r = z.max(axis=1) - z.min(axis=1)
        s = chunks.std(axis=1)
        ok = s > 0
        if ok.any():
            log_n.append(np.log(n))
            log_rs.append(np.log(np.mean(r[ok] / s[ok])))
    if len(log_n) < 2:
        return 0.5
    return float(np.polyfit(log_n, log_rs, 1)[0])


def binary_entropy(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p)) if 0 < p < 1 else 0.0


# --------------------------------------------------------------------------
class FeatureEngine:
    def __init__(self, window_size: int = S.WINDOW_SIZE):
        self.W = window_size

    def get_feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def extract(self, temporal_data: dict, session_metadata: dict) -> np.ndarray:
        """Return the (63,) feature vector."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with np.errstate(all="ignore"):
                f = self._extract(temporal_data, session_metadata)
        x = np.array([f[name] for name in FEATURE_NAMES], dtype=float)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    def _extract(self, td: dict, meta: dict) -> dict:
        Q = np.asarray(td["qber_sequence"], dtype=float)
        T = Q.size
        f: dict[str, float] = {}

        # ---------------- Category 1: statistical ----------------
        if T == 0:
            Q = np.zeros(1)
            T = 1
        mean, std = float(Q.mean()), float(Q.std())
        f["qber_mean"] = mean
        f["qber_std"] = std
        f["qber_variance"] = float(Q.var())
        f["qber_skewness"] = float(stats.skew(Q)) if std > 0 else 0.0
        f["qber_kurtosis"] = float(stats.kurtosis(Q)) if std > 0 else 0.0
        f["qber_median"] = float(np.median(Q))
        q25, q75 = np.percentile(Q, [25, 75])
        f["qber_iqr"] = float(q75 - q25)
        f["qber_cv"] = _safe_div(std, mean) if mean > 0 else 0.0
        f["qber_min"] = float(Q.min())
        f["qber_max"] = float(Q.max())
        f["qber_range"] = float(Q.max() - Q.min())
        f["qber_p10"] = float(np.percentile(Q, 10))
        f["qber_p90"] = float(np.percentile(Q, 90))

        # ---------------- Category 2: burst & instability ----------------
        spikes = spike_mask(Q)
        f["spike_density"] = float(spikes.mean())
        bursts = _runs(spikes)
        f["burst_duration_mean"] = float(bursts.mean()) if bursts.size else 0.0
        f["burst_duration_max"] = float(bursts.max()) if bursts.size else 0.0
        f["jump_energy"] = jump_energy(Q)
        t = np.arange(T, dtype=float)
        if T >= 2 and std > 0:
            slope, intercept = np.polyfit(t, Q, 1)
            fit = slope * t + intercept
            ss_res = np.sum((Q - fit) ** 2)
            ss_tot = np.sum((Q - mean) ** 2)
            f["temporal_drift"] = float(slope)
            f["drift_r_squared"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            f["temporal_drift"] = 0.0
            f["drift_r_squared"] = 0.0
        spike_idx = np.flatnonzero(spikes)
        gaps = np.diff(spike_idx).astype(float)
        f["error_gap_mean"] = float(gaps.mean()) if gaps.size else 0.0
        f["error_gap_std"] = float(gaps.std()) if gaps.size else 0.0
        f["error_gap_cv"] = _safe_div(f["error_gap_std"], f["error_gap_mean"])
        low_runs, high_runs = _runs(Q < mean), _runs(Q > mean)
        f["max_consecutive_low"] = float(low_runs.max()) if low_runs.size else 0.0
        f["max_consecutive_high"] = float(high_runs.max()) if high_runs.size else 0.0
        f["run_ratio"] = _safe_div(f["max_consecutive_high"], f["max_consecutive_low"])
        f["abs_diff_mean"] = float(np.mean(np.abs(np.diff(Q)))) if T > 1 else 0.0

        # ---------------- Category 3: spectral & temporal ----------------
        f["autocorr_lag1"] = autocorr(Q, 1)
        f["autocorr_lag2"] = autocorr(Q, 2)
        f["autocorr_lag5"] = autocorr(Q, 5)
        mags = np.abs(np.fft.fft(Q))[1: T // 2 + 1] if T >= 2 else np.zeros(0)
        if mags.size and mags.sum() > 0:
            f["dft_energy"] = float(np.sum(mags ** 2))
            f["dft_peak_freq"] = float(np.argmax(mags) + 1)
            f["dft_peak_magnitude"] = float(mags.max())
            p = mags / mags.sum()
            p = p[p > 0]
            f["spectral_entropy"] = float(-np.sum(p * np.log2(p)))
            half = max(1, mags.size // 2)
            low_e, high_e = np.sum(mags[:half] ** 2), np.sum(mags[half:] ** 2)
            f["energy_ratio_low_high"] = _safe_div(low_e, high_e)
        else:
            for k in ("dft_energy", "dft_peak_freq", "dft_peak_magnitude",
                      "spectral_entropy", "energy_ratio_low_high"):
                f[k] = 0.0
        hist, _ = np.histogram(Q, bins=10)
        h = hist[hist > 0] / T
        f["shannon_entropy"] = float(-np.sum(h * np.log2(h)))
        f["hurst_exponent"] = hurst_rs(Q)
        d = np.diff(Q)
        f["turning_point_rate"] = (float(np.sum(d[:-1] * d[1:] < 0)) / (T - 2)) if T > 2 else 0.0
        c = Q - mean
        f["mean_crossing_rate"] = float(np.sum(c[:-1] * c[1:] < 0)) / T if T > 1 else 0.0

        # ---------------- Category 4: basis dependent ----------------
        Qr = _impute(td.get("qber_rect_sequence", np.zeros(0)))
        Qd = _impute(td.get("qber_diag_sequence", np.zeros(0)))
        if Qr.size == 0:
            Qr = np.zeros(1)
        if Qd.size == 0:
            Qd = np.zeros(1)
        rm, dm = float(Qr.mean()), float(Qd.mean())
        f["qber_rect_mean"] = rm
        f["qber_diag_mean"] = dm
        f["basis_asymmetry"] = abs(rm - dm)
        f["basis_ratio"] = _safe_div(max(rm, dm), min(rm, dm)) if min(rm, dm) > 0 else 0.0
        rs, ds = float(Qr.std()), float(Qd.std())
        f["qber_rect_std"] = rs
        f["qber_diag_std"] = ds
        f["basis_std_diff"] = abs(rs - ds)
        f["rect_spike_density"] = spike_density(Qr)
        f["diag_spike_density"] = spike_density(Qd)
        f["basis_spike_ratio"] = _safe_div(f["rect_spike_density"], f["diag_spike_density"])
        f["rect_jump_energy"] = jump_energy(Qr)
        f["diag_jump_energy"] = jump_energy(Qd)
        f["basis_energy_diff"] = abs(f["rect_jump_energy"] - f["diag_jump_energy"])

        # ---------------- Category 5: channel interaction ----------------
        n_qubits = float(meta.get("n_qubits", 0) or 0)
        n_sifted = float(meta.get("n_sifted", len(td.get("alice_sifted", []))) or 0)
        loss = float(meta.get("photon_loss_rate", 0.0) or 0.0)
        f["key_gen_rate"] = _safe_div(n_sifted, n_qubits)
        f["photon_loss_rate"] = loss
        f["transmission_efficiency"] = 1.0 - loss

        lpw = np.asarray(td.get("loss_per_window", np.zeros(0)), dtype=float)
        if (lpw.size != Q.size or not np.any(lpw)) and "matching_indices" in meta \
                and "lost_mask" in meta and td.get("n_windows", 0):
            lpw = TemporalQBER.loss_per_window(np.asarray(meta["matching_indices"]),
                                               np.asarray(meta["lost_mask"], dtype=bool),
                                               td.get("window_size", self.W), td["n_windows"])
        if lpw.size == Q.size and Q.size > 2 and lpw.std() > 0 and std > 0:
            f["qber_loss_correlation"] = float(stats.pearsonr(Q, lpw)[0])
        else:
            f["qber_loss_correlation"] = 0.0
        f["qber_stability"] = 1.0 - _safe_div(std, mean) if mean > 0 else 0.0

        a, b = td.get("alice_sifted"), td.get("bob_sifted")
        if a is not None and b is not None and len(a):
            err = (np.asarray(a) != np.asarray(b)).astype(float)
            q25s = TemporalQBER._window_qber(err, S.MULTISCALE_WINDOWS[0])
            q100s = TemporalQBER._window_qber(err, S.MULTISCALE_WINDOWS[1])
        else:
            q25s = q100s = np.zeros(0)
        m25 = float(q25s.mean()) if q25s.size else 0.0
        m100 = float(q100s.mean()) if q100s.size else 0.0
        s25 = float(q25s.std()) if q25s.size else 0.0
        s100 = float(q100s.std()) if q100s.size else 0.0
        f["multiscale_qber_25"] = m25
        f["multiscale_qber_100"] = m100
        f["multiscale_std_25"] = s25
        f["multiscale_std_100"] = s100
        f["scale_ratio_mean"] = _safe_div(m25, m100)
        f["scale_ratio_std"] = _safe_div(s25, s100)
        # Leakage estimate: Shannon bound h(QBER) for Eve's disturbance-based
        # information plus the lost-photon fraction (conservative PNS bound).
        if n_sifted > 0:
            leaked = min(n_sifted, n_sifted * (binary_entropy(mean) + loss))
            f["effective_key_fraction"] = (n_sifted - leaked) / n_sifted
        else:
            f["effective_key_fraction"] = 0.0
        return f


def extract_session_features(session_result: dict, window_size: int = S.WINDOW_SIZE
                             ) -> tuple[np.ndarray, dict]:
    """Full pipeline: session -> temporal QBER -> 63 features."""
    tq = TemporalQBER(window_size)
    td = tq.compute(session_result["sifted_alice"], session_result["sifted_bob"],
                    session_result["sifted_bases"],
                    matching_indices=session_result.get("matching_indices"),
                    lost_mask=session_result.get("lost_mask"))
    x = FeatureEngine(window_size).extract(td, session_result["metadata"])
    return x, td
