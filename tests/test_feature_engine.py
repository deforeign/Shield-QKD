import numpy as np
from scipy import stats

from ml.feature_engine import FEATURE_GROUPS, FEATURE_NAMES, FeatureEngine, extract_session_features
from ml.temporal_qber import TemporalQBER
from quantum import BBM92Session


def _features_from_q(Q, n_sifted=2500):
    td = {"qber_sequence": np.asarray(Q, float), "qber_rect_sequence": np.asarray(Q, float),
          "qber_diag_sequence": np.asarray(Q, float), "n_windows": len(Q), "window_size": 50,
          "loss_per_window": np.zeros(len(Q))}
    meta = {"n_qubits": 5000, "n_sifted": n_sifted, "photon_loss_rate": 0.02}
    return dict(zip(FEATURE_NAMES, FeatureEngine().extract(td, meta)))


def test_feature_count():
    r = BBM92Session(seed=1).run()
    x, _ = extract_session_features(r)
    assert x.shape == (63,)
    assert len(FeatureEngine().get_feature_names()) == 63
    assert sum(len(v) for v in FEATURE_GROUPS.values()) == 63


def test_no_nan():
    for nm in (None, "depolarizing"):
        r = BBM92Session(noise_model=nm, noise_prob=0.03, seed=2).run()
        x, _ = extract_session_features(r)
        assert np.all(np.isfinite(x))


def test_known_pattern():
    Q = np.array([0.02, 0.04] * 10 + [0.30] + [0.02, 0.04] * 9 + [0.03])
    f = _features_from_q(Q)
    assert np.isclose(f["qber_mean"], Q.mean())
    assert np.isclose(f["qber_std"], Q.std())
    assert np.isclose(f["qber_variance"], Q.var())
    assert np.isclose(f["qber_skewness"], stats.skew(Q))
    assert np.isclose(f["qber_kurtosis"], stats.kurtosis(Q))
    assert np.isclose(f["qber_median"], np.median(Q))
    assert np.isclose(f["qber_range"], 0.28)
    assert np.isclose(f["spike_density"], 1 / len(Q))       # the single 0.30 spike
    assert f["burst_duration_max"] == 1.0
    assert np.isclose(f["jump_energy"], np.sum(np.diff(Q) ** 2))
    assert np.isclose(f["abs_diff_mean"], np.mean(np.abs(np.diff(Q))))
    assert f["autocorr_lag1"] < 0                               # alternating pattern
    assert np.isclose(f["key_gen_rate"], 0.5)
    assert np.isclose(f["transmission_efficiency"], 0.98)
    assert f["basis_asymmetry"] == 0.0


def test_constant_series_is_safe():
    f = _features_from_q(np.zeros(40))
    assert f["qber_cv"] == 0.0 and f["spike_density"] == 0.0 and f["qber_skewness"] == 0.0


def test_linear_drift():
    Q = np.linspace(0.01, 0.09, 50)
    f = _features_from_q(Q)
    assert np.isclose(f["temporal_drift"], 0.08 / 49)
    assert np.isclose(f["drift_r_squared"], 1.0)


def test_windowing_and_basis_split():
    a = np.zeros(200, dtype=np.int8)
    b = a.copy()
    b[:50] = 1                         # window 1 fully wrong
    bases = np.tile([0, 1], 100)
    td = TemporalQBER(50).compute(a, b, bases)
    assert td["n_windows"] == 4
    assert np.allclose(td["qber_sequence"], [1, 0, 0, 0])
    assert np.allclose(td["qber_rect_sequence"], [1, 0, 0, 0])
    assert np.allclose(td["qber_diag_sequence"], [1, 0, 0, 0])


def test_time_shift_is_burstier_than_cloning():
    ts = extract_session_features(BBM92Session(attack_type="time_shift", seed=3).run())[0]
    cl = extract_session_features(BBM92Session(attack_type="quantum_cloning", seed=3).run())[0]
    i = FEATURE_NAMES.index("qber_cv")
    assert ts[i] > cl[i]
