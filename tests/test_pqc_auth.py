import numpy as np

from pqc.authenticated_sifting import AuthenticatedSifting, simulate_mitm
from pqc.lamport_ots import LamportOTS


def test_valid_signature():
    ots = LamportOTS()
    sk, pk = ots.keygen()
    msg = b"alice bases: 0110"
    assert ots.verify(msg, ots.sign(msg, sk), pk)


def test_tampered_message():
    ots = LamportOTS()
    sk, pk = ots.keygen()
    sig = ots.sign(b"alice bases: 0110", sk)
    assert not ots.verify(b"alice bases: 0111", sig, pk)


def test_wrong_key():
    ots = LamportOTS()
    sk_a, _ = ots.keygen()
    _, pk_b = ots.keygen()
    msg = b"hello"
    assert not ots.verify(msg, ots.sign(msg, sk_a), pk_b)


def test_signature_structure():
    ots = LamportOTS()
    sk, pk = ots.keygen()
    sig = ots.sign(b"x", sk)
    assert len(sk) == len(pk) == len(sig) == 256
    assert all(len(s) == 32 for s in sig)
    assert not ots.verify(b"x", sig[:-1], pk)


def test_authenticated_basis_exchange_roundtrip():
    auth = AuthenticatedSifting()
    hs = auth.handshake()
    assert hs["keys_remaining"] == auth.n_keypairs
    bases = np.random.default_rng(0).integers(0, 2, 5000).astype(np.int8)
    pkt = auth.send_bases(bases, "alice")
    ok, rx = auth.verify_bases(pkt, auth.alice_pk)
    assert ok and np.array_equal(rx, bases)
    # one-time property: private key was destroyed and replay is rejected
    assert auth.alice_keys[pkt["pk_index"]][0] is None
    ok2, _ = auth.verify_bases(pkt, auth.alice_pk)
    assert not ok2


def test_mitm_modes_are_detected():
    bases = np.random.default_rng(1).integers(0, 2, 2000).astype(np.int8)
    for mode in ("tamper", "forge", "replay"):
        auth = AuthenticatedSifting()
        auth.handshake()
        pkt = simulate_mitm(auth.send_bases(bases, "bob"), mode=mode, seed=3)
        ok, rx = auth.verify_bases(pkt, auth.bob_pk)
        assert ok is False and rx is None


def test_full_sifting_with_and_without_mitm():
    rng = np.random.default_rng(2)
    a, b = rng.integers(0, 2, 1000), rng.integers(0, 2, 1000)
    lost = rng.random(1000) < 0.05
    res = AuthenticatedSifting().authenticated_sift(a, b, lost)
    assert res["pqc_verified"]
    assert np.array_equal(res["matching_mask"], (a == b) & ~lost)
    res2 = AuthenticatedSifting().authenticated_sift(a, b, lost, mitm="tamper")
    assert not res2["pqc_verified"] and res2["matching_mask"] is None
