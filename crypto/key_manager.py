"""
Key storage, rotation and destruction.

From sifted key to AES-256 key:
  1. Error correction (simulated Cascade): Bob's errors are corrected; the
     parity information disclosed is  leak_EC = f * n * h(QBER), f = 1.16.
  2. Privacy amplification: the secure length is bounded by
         n_secure = n * (1 - h(QBER) - photon_loss) - leak_EC
     (Shannon bound + conservative lost-photon term, the same estimate as
     the `effective_key_fraction` feature).  At least 256 secure bits are required.
  3. HKDF-SHA256(key bits, salt, info) -> 32-byte AES-256 key, one copy per party.

Keys are one-time: `consume` hands the key out once and zeroises the stored
bytes; `destroy` wipes both copies (forward secrecy).
"""
from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from config import settings as S

EC_EFFICIENCY = 1.16


def h2(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def secure_key_length(n_sifted: int, qber: float, photon_loss: float = 0.0) -> dict:
    leak_ec = EC_EFFICIENCY * n_sifted * h2(qber)
    n_secure = n_sifted * (1 - h2(qber) - photon_loss) - leak_ec
    return {"n_sifted": n_sifted, "qber": qber, "leak_ec_bits": leak_ec,
            "n_secure_bits": max(0, int(np.floor(n_secure)))}


@dataclass
class KeyRecord:
    key_id: str
    alice: bytearray
    bob: bytearray
    created: str
    n_secure_bits: int
    source: dict = field(default_factory=dict)
    used: dict = field(default_factory=lambda: {"alice": False, "bob": False})
    destroyed: bool = False


def _zeroize(buf: bytearray) -> None:
    for i in range(len(buf)):
        buf[i] = 0


class KeyManager:
    def __init__(self):
        self._keys: dict[str, KeyRecord] = {}
        self.audit: list[str] = []

    def _log(self, msg: str):
        self.audit.append(f"{datetime.now(timezone.utc).isoformat()} {msg}")

    # ------------------------------------------------------------------
    def derive_from_session(self, sifted_alice: np.ndarray, sifted_bob: np.ndarray,
                            photon_loss: float = 0.0, info: bytes = b"ShieldQKD AES-256-GCM v1"
                            ) -> str:
        a = np.asarray(sifted_alice, dtype=np.uint8)
        b = np.asarray(sifted_bob, dtype=np.uint8)
        n = int(a.size)
        if n == 0:
            raise ValueError("empty sifted key")
        qber = float(np.mean(a != b))
        budget = secure_key_length(n, qber, photon_loss)
        if budget["n_secure_bits"] < 8 * S.AES_KEY_BYTES:
            raise ValueError(f"only {budget['n_secure_bits']} secure bits after privacy "
                             f"amplification - not enough for AES-256")
        # error correction: Bob reconciles to Alice's string (verified by hash in practice)
        bob_corrected = a.copy()
        salt = secrets.token_bytes(16)          # public, sent over the authenticated channel

        def kdf(bits: np.ndarray) -> bytearray:
            material = np.packbits(bits).tobytes()
            return bytearray(HKDF(algorithm=hashes.SHA256(), length=S.AES_KEY_BYTES, salt=salt,
                                  info=info).derive(material))

        key_id = uuid.uuid4().hex[:12]
        rec = KeyRecord(key_id=key_id, alice=kdf(a), bob=kdf(bob_corrected),
                        created=datetime.now(timezone.utc).isoformat(),
                        n_secure_bits=budget["n_secure_bits"],
                        source={**budget, "photon_loss": photon_loss, "salt": salt.hex()})
        assert rec.alice == rec.bob
        self._keys[key_id] = rec
        self._log(f"derived key {key_id} ({budget['n_secure_bits']} secure bits, QBER {qber:.4f})")
        return key_id

    def fingerprint(self, key_id: str) -> str:
        import hashlib
        rec = self._keys[key_id]
        src = rec.alice if not rec.used["alice"] else rec.bob
        return hashlib.sha256(bytes(src)).hexdigest()[:16]

    def consume(self, key_id: str, party: str) -> bytes:
        """Return the party's copy once, then wipe it."""
        rec = self._keys.get(key_id)
        if rec is None or rec.destroyed:
            raise KeyError(f"key {key_id} unknown or destroyed")
        if rec.used[party]:
            raise PermissionError(f"key {key_id} already consumed by {party} (one-time use)")
        buf = rec.alice if party == "alice" else rec.bob
        key = bytes(buf)
        _zeroize(buf)
        rec.used[party] = True
        self._log(f"key {key_id} consumed by {party}")
        if all(rec.used.values()):
            rec.destroyed = True
            self._log(f"key {key_id} fully consumed -> destroyed")
        return key

    def destroy(self, key_id: str) -> None:
        rec = self._keys.get(key_id)
        if rec is None:
            return
        _zeroize(rec.alice)
        _zeroize(rec.bob)
        rec.destroyed = True
        self._log(f"key {key_id} destroyed")

    def destroy_all(self) -> int:
        n = 0
        for kid in list(self._keys):
            if not self._keys[kid].destroyed:
                self.destroy(kid)
                n += 1
        return n

    def rotate(self, old_key_id: str, sifted_alice, sifted_bob, photon_loss: float = 0.0) -> str:
        self.destroy(old_key_id)
        new_id = self.derive_from_session(sifted_alice, sifted_bob, photon_loss)
        self._log(f"rotated {old_key_id} -> {new_id}")
        return new_id

    def status(self) -> list[dict]:
        return [{"key_id": r.key_id, "created": r.created, "n_secure_bits": r.n_secure_bits,
                 "used": dict(r.used), "destroyed": r.destroyed,
                 "all_zero": not any(r.alice) and not any(r.bob)} for r in self._keys.values()]

    def is_destroyed(self, key_id: str) -> bool:
        rec = self._keys.get(key_id)
        return rec is None or rec.destroyed
