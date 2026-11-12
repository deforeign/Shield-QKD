import numpy as np
import pytest

from crypto import FileVault, KeyManager
from quantum import BBM92Session
from security.response_router import ResponseRouter


def test_proceed_on_normal(breaker):
    r = BBM92Session(seed=11).run()
    ev = breaker.evaluate(r, pqc_verified=True)
    assert ev["decision"] == "PROCEED"
    assert ev["attack_class"] is None


def test_abort_on_mitm(breaker):
    r = BBM92Session(seed=12).run()
    ev = breaker.evaluate(r, pqc_verified=False)
    assert ev["decision"] == "ABORT_MITM"
    assert ev["attack_class"] == "Classical MITM" and ev["confidence"] == 1.0


def test_abort_on_attack(breaker):
    r = BBM92Session(attack_type="intercept_resend", attack_params={"interception_rate": 0.25},
                    seed=13).run()
    ev = breaker.evaluate(r, pqc_verified=True)
    assert ev["decision"] == "ABORT_ATTACK"
    assert ev["attack_class"] == "Intercept-Resend"
    assert ev["recommended_action"] == ResponseRouter.get_action("Intercept-Resend")


def test_strong_intercept_resend_never_proceeds(breaker):
    """IR at 40% (QBER ~10%) produces i.i.d. symmetric errors exactly like a
    cloner with F ~ 0.8, so the class may be ambiguous - but the key must
    never be released."""
    for s in range(5):
        r = BBM92Session(attack_type="intercept_resend", attack_params={"interception_rate": 0.4},
                        seed=300 + s).run()
        ev = breaker.evaluate(r, pqc_verified=True)
        assert ev["decision"] in ("ABORT_ATTACK", "WARN")
        assert ev["attack_class"] in ("Intercept-Resend", "Quantum-Cloning", "Fake-State")


def test_resampling_resolves_uncertain_normal(breaker):
    from quantum.bbm92_simulator import BBM92Session as B
    ev = breaker.evaluate_with_resampling(
        lambda r: B(noise_model="depolarizing", noise_prob=0.03, seed=400 + r).run(), True)
    assert ev["decision"] == "PROCEED"
    assert 1 <= ev["n_sessions_used"] <= 4


def test_forensics_report(breaker):
    r = BBM92Session(attack_type="trojan_horse", attack_params={"probe_rate": 0.3}, seed=14).run()
    ev = breaker.evaluate(r, pqc_verified=True)
    rep = ev["forensics_report"]
    assert ev["decision"] in ("ABORT_ATTACK", "WARN")
    assert "Attack class" in rep and "Confidence" in rep and "SHAP" in rep
    assert ev["shap_values"] and len(ev["shap_values"]["top_features"]) == 5
    assert ev["attack_class"] in rep


def test_stealthy_attack_below_threshold(breaker):
    """Spec demo step 3: IR at 15% keeps QBER < 11% but ML still flags it."""
    flagged = 0
    for s in range(5):
        r = BBM92Session(attack_type="intercept_resend", attack_params={"interception_rate": 0.15},
                        noise_model="depolarizing", noise_prob=0.01, seed=200 + s).run()
        ev = breaker.evaluate(r, pqc_verified=True)
        assert ev["threshold_verdict"]["flags_attack"] is False
        flagged += ev["decision"] in ("ABORT_ATTACK", "WARN")
    assert flagged >= 4


def test_pipeline_end_to_end(breaker):
    from security.pipeline import run_shieldqkd
    out = run_shieldqkd(seed=21, breaker=breaker)
    assert out["evaluation"]["decision"] == "PROCEED" and out["key_id"]
    km = out["key_manager"]
    vault = FileVault(km)
    blob = vault.encrypt_bytes(b"top secret", out["key_id"])
    assert vault.decrypt_bytes(blob) == b"top secret"
    assert km.is_destroyed(out["key_id"])                     # one-time key consumed
    mitm = run_shieldqkd(seed=22, mitm="tamper", breaker=breaker)
    assert mitm["evaluation"]["decision"] == "ABORT_MITM"
    assert mitm["key_status"] == "KEY MATERIAL DESTROYED"
    assert not mitm["session"]["sifted_alice"].any()


def test_key_manager_one_time_use():
    km = KeyManager()
    bits = np.random.default_rng(0).integers(0, 2, 2400)
    kid = km.derive_from_session(bits, bits)
    k1 = km.consume(kid, "alice")
    assert len(k1) == 32
    with pytest.raises(PermissionError):
        km.consume(kid, "alice")
    km.destroy(kid)
    assert km.status()[0]["all_zero"]


def test_key_manager_refuses_insecure_key():
    km = KeyManager()
    a = np.random.default_rng(1).integers(0, 2, 2400)
    b = a.copy()
    b[: 600] ^= 1                                            # 25 % QBER
    with pytest.raises(ValueError):
        km.derive_from_session(a, b)
