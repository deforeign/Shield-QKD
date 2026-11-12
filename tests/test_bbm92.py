import numpy as np
import pytest

from quantum import BBM92Session, run_session
from quantum.attack_models import ATTACKS, markov_episodes, placed_episodes
from quantum.noise_models import (check_completeness, expected_qber, kraus_operators,
                                  apply_channel_density, apply_channel_bloch)


def test_no_attack_low_qber():
    r = BBM92Session(n_qubits=5000, seed=1).run()
    assert r["overall_qber"] < 0.02


def test_intercept_resend_qber():
    q = [BBM92Session(attack_type="intercept_resend", attack_params={"interception_rate": 0.40},
                      seed=s).run()["overall_qber"] for s in range(5)]
    assert abs(np.mean(q) - 0.10) < 0.015


def test_sifted_key_length():
    r = BBM92Session(n_qubits=5000, channel_loss=0.0, seed=2).run()
    assert abs(r["n_sifted"] - 2500) <= 0.05 * 2500


def test_basis_uniformity():
    r = BBM92Session(n_qubits=20000, seed=3).run()
    assert abs(r["alice_bases"].mean() - 0.5) < 0.02
    assert abs(r["bob_bases"].mean() - 0.5) < 0.02
    assert abs(r["sifted_bases"].mean() - 0.5) < 0.03


def test_output_contract():
    r = BBM92Session(n_qubits=1000, seed=4).run()
    for k in ("alice_bits", "alice_bases", "bob_bits", "bob_bases", "sifted_alice", "sifted_bob",
              "matching_indices", "sifted_bases", "n_sifted", "overall_qber",
              "photon_loss_rate", "metadata"):
        assert k in r
    assert r["metadata"]["protocol"] == "BBM92"
    assert len(r["sifted_alice"]) == len(r["sifted_bob"]) == len(r["sifted_bases"]) == r["n_sifted"]
    assert np.all(r["alice_bases"][r["matching_indices"]] == r["bob_bases"][r["matching_indices"]])
    assert run_session(n_qubits=500, seed=4)["n_sifted"] > 0


def test_entangled_pair_correlations():
    """|Phi+>: Alice's outcomes are uniformly random, yet in matching bases Bob's
    outcomes are identical to hers on a clean channel."""
    r = BBM92Session(n_qubits=20000, channel_loss=0.0, seed=9).run()
    assert abs(r["alice_bits"].mean() - 0.5) < 0.02
    assert np.array_equal(r["sifted_alice"], r["sifted_bob"])
    mism = r["alice_bases"] != r["bob_bases"]          # mismatched bases -> uncorrelated
    agree = np.mean(r["alice_bits"][mism] == r["bob_bits"][mism])
    assert abs(agree - 0.5) < 0.02


@pytest.mark.parametrize("attack", list(ATTACKS))
def test_every_attack_runs(attack):
    r = BBM92Session(attack_type=attack, seed=5).run()
    assert 0 <= r["overall_qber"] <= 0.5
    assert r["n_sifted"] > 1500


def test_attack_signatures():
    pns = BBM92Session(attack_type="pns", attack_params={"multi_photon_prob": 0.3}, seed=6).run()
    assert pns["overall_qber"] < 0.01 and pns["photon_loss_rate"] > 0.15
    clone = BBM92Session(attack_type="quantum_cloning", attack_params={"clone_fidelity": 0.6},
                         seed=7).run()
    assert abs(clone["overall_qber"] - 0.20) < 0.03
    fake = BBM92Session(n_qubits=20000, attack_type="fake_state",
                        attack_params={"fake_rate": 0.4, "basis_bias": 1.0}, seed=8).run()
    sa, sb, bs = fake["sifted_alice"], fake["sifted_bob"], fake["sifted_bases"]
    q_rect = np.mean(sa[bs == 0] != sb[bs == 0])
    q_diag = np.mean(sa[bs == 1] != sb[bs == 1])
    assert q_diag - q_rect > 0.1            # Eve always uses rectilinear -> diagonal errors


@pytest.mark.parametrize("attack,params,expected", [
    ("intercept_resend", {"interception_rate": 0.4}, 0.4 * 0.25),
    ("pns", {"multi_photon_prob": 0.2}, 0.0),
    ("beam_splitting", {"splitting_ratio": 0.3}, 0.3 * 0.10),
    ("time_shift", {"shift_prob": 0.2, "burst_len": 10}, 0.2 * 0.20),
    ("trojan_horse", {"probe_rate": 0.2}, 0.2 * 0.15),
    ("quantum_cloning", {"clone_fidelity": 0.7}, (1 - 0.7) / 2),
    ("fake_state", {"fake_rate": 0.3, "basis_bias": 0.9}, 0.3 * 0.25),
])
def test_section10_expected_qber(attack, params, expected):
    """Every attack's sifted-key QBER matches the spec's Section 10 formula."""
    q = [BBM92Session(attack_type=attack, attack_params=params, channel_loss=0.0, seed=s).run()["overall_qber"]
         for s in range(30)]
    assert abs(np.mean(q) - expected) < 0.003


def test_markov_episode_rate():
    rng = np.random.default_rng(0)
    m = markov_episodes(200000, 0.2, 100, rng)
    assert abs(m.mean() - 0.2) < 0.03


def test_placed_episodes_rate_and_clustering():
    rng = np.random.default_rng(1)
    for p in (0.05, 0.2, 0.3):
        m = placed_episodes(5000, p, 600, rng)
        assert abs(m.mean() - p) < 0.002          # realised duty cycle == p
        assert m.any()                            # the attack is always active
    runs = np.diff(np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]]))))[::2]
    assert runs.mean() > 50                       # clustered, not i.i.d.


@pytest.mark.parametrize("model", ["bit_flip", "phase_flip", "amplitude_damping", "depolarizing"])
def test_kraus_completeness_and_bloch_form(model):
    p = 0.2
    assert check_completeness(kraus_operators(model, p))
    # closed-form Bloch map == Kraus map on |+>
    plus = 0.5 * np.array([[1, 1], [1, 1]], dtype=complex)
    rho = apply_channel_density(plus, model, p)
    x, z = 2 * rho[0, 1].real, (rho[0, 0] - rho[1, 1]).real
    xb, zb = apply_channel_bloch(np.array([1.0]), np.array([0.0]), model, p)
    assert np.isclose(x, xb[0]) and np.isclose(z, zb[0])


@pytest.mark.parametrize("model", ["bit_flip", "phase_flip", "amplitude_damping", "depolarizing"])
def test_noise_qber_matches_theory(model):
    q = np.mean([BBM92Session(noise_model=model, noise_prob=0.1, seed=s).run()["overall_qber"]
                 for s in range(6)])
    assert abs(q - expected_qber(model, 0.1)) < 0.01


@pytest.mark.slow
def test_qiskit_matches_numpy():
    pytest.importorskip("qiskit_aer")
    clean = BBM92Session(n_qubits=2000, channel_loss=0.0, backend="qiskit", seed=1).run()
    assert np.array_equal(clean["sifted_alice"], clean["sifted_bob"])   # real Bell-pair circuits
    assert abs(clean["alice_bits"].mean() - 0.5) < 0.05
    kw = dict(n_qubits=3000, noise_model="depolarizing", noise_prob=0.06,
              attack_type="intercept_resend", attack_params={"interception_rate": 0.3})
    q_qk = np.mean([BBM92Session(backend="qiskit", seed=s, **kw).run()["overall_qber"] for s in range(3)])
    q_np = np.mean([BBM92Session(backend="numpy", seed=s, **kw).run()["overall_qber"] for s in range(10)])
    expected = 0.075 + 0.03 - 2 * 0.075 * 0.03
    assert abs(q_qk - expected) < 0.015 and abs(q_np - expected) < 0.01
