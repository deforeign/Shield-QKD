"""
PQC-authenticated basis sifting (spec Section 6.2).

BBM92 is only secure if the classical channel is authenticated: otherwise Eve
can sit in the middle and run two separate QKD sessions.  Here every classical
basis announcement is signed with a fresh Lamport one-time key whose public
half was distributed beforehand (handshake).

Wire format of a basis announcement:
    b"SQKD1" | sender (1 byte) | round (uint32) | n (uint32) | bases (int8 * n)
Bases use 0/1 and -1 for "no detection".
"""
from __future__ import annotations

import struct

import numpy as np

from config import settings as S
from pqc.lamport_ots import LamportOTS

MAGIC = b"SQKD1"
SENDER_CODES = {"alice": 1, "bob": 2}
CODE_SENDERS = {v: k for k, v in SENDER_CODES.items()}


def serialize_bases(bases: np.ndarray, sender: str, round_no: int) -> bytes:
    arr = np.asarray(bases, dtype=np.int8)
    return MAGIC + struct.pack(">BII", SENDER_CODES[sender], round_no, arr.size) + arr.tobytes()


def deserialize_bases(message: bytes) -> tuple[str, int, np.ndarray]:
    if not message.startswith(MAGIC) or len(message) < len(MAGIC) + 9:
        raise ValueError("malformed basis announcement")
    code, round_no, n = struct.unpack(">BII", message[len(MAGIC):len(MAGIC) + 9])
    body = message[len(MAGIC) + 9:]
    if len(body) != n:
        raise ValueError("length mismatch in basis announcement")
    return CODE_SENDERS[code], round_no, np.frombuffer(body, dtype=np.int8).copy()


class AuthenticatedSifting:
    def __init__(self, n_keypairs: int = S.LAMPORT_KEYPAIRS_PER_PARTY):
        self.lamport = LamportOTS()
        self.n_keypairs = n_keypairs
        self.alice_keys: list = []          # pool of (sk, pk); sk wiped after use
        self.bob_keys: list = []
        self.alice_pk: list = []
        self.bob_pk: list = []
        self._next = {"alice": 0, "bob": 0}
        self._used = {"alice": set(), "bob": set()}
        self._round = 0
        self.log: list[str] = []

    # ------------------------------------------------------------------
    def handshake(self) -> dict:
        """Pre-distribute one-time public keys (out-of-band in practice)."""
        self.alice_keys = [self.lamport.keygen() for _ in range(self.n_keypairs)]
        self.bob_keys = [self.lamport.keygen() for _ in range(self.n_keypairs)]
        self.alice_pk = [pk for _, pk in self.alice_keys]
        self.bob_pk = [pk for _, pk in self.bob_keys]
        self._next = {"alice": 0, "bob": 0}
        self._used = {"alice": set(), "bob": set()}
        self.log.append(f"handshake: {self.n_keypairs} Lamport keypairs per party exchanged")
        return {"alice_pk": self.alice_pk, "bob_pk": self.bob_pk,
                "keys_remaining": self.keys_remaining(),
                "alice_fingerprints": [LamportOTS.public_key_fingerprint(p)[:16] for p in self.alice_pk],
                "bob_fingerprints": [LamportOTS.public_key_fingerprint(p)[:16] for p in self.bob_pk]}

    def keys_remaining(self) -> int:
        return min(self.n_keypairs - self._next["alice"], self.n_keypairs - self._next["bob"])

    def public_keys(self, sender: str) -> list:
        return self.alice_pk if sender == "alice" else self.bob_pk

    # ------------------------------------------------------------------
    def send_bases(self, bases: np.ndarray, sender: str) -> dict:
        """Create a signed basis announcement with the sender's next unused key."""
        if sender not in SENDER_CODES:
            raise ValueError("sender must be 'alice' or 'bob'")
        pool = self.alice_keys if sender == "alice" else self.bob_keys
        idx = self._next[sender]
        if idx >= len(pool):
            raise RuntimeError(f"{sender} has exhausted her one-time keys; re-run handshake()")
        sk, _ = pool[idx]
        self._round += 1
        message = serialize_bases(bases, sender, self._round)
        signature = self.lamport.sign(message, sk)
        pool[idx] = (None, pool[idx][1])                 # one-time: destroy the private key
        self._next[sender] = idx + 1
        self.log.append(f"{sender} signed basis announcement #{self._round} with key {idx}")
        return {"message": message, "signature": signature, "pk_index": idx, "sender": sender}

    def verify_bases(self, packet: dict, sender_public_keys: list) -> tuple:
        """Return (is_valid, bases or None).  False -> MITM detected, abort."""
        try:
            idx = int(packet["pk_index"])
            if not 0 <= idx < len(sender_public_keys):
                self.log.append("verify: unknown key index -> REJECT")
                return False, None
            sender, round_no, bases = deserialize_bases(packet["message"])
            if idx in self._used[sender]:
                self.log.append(f"verify: key {idx} of {sender} re-used (replay) -> REJECT")
                return False, None
            if not self.lamport.verify(packet["message"], packet["signature"],
                                       sender_public_keys[idx]):
                self.log.append(f"verify: Lamport signature from {sender} INVALID -> MITM")
                return False, None
            self._used[sender].add(idx)
            self.log.append(f"verify: announcement #{round_no} from {sender} VERIFIED")
            return True, bases
        except Exception as exc:                          # malformed packet == attack
            self.log.append(f"verify: malformed packet ({exc}) -> REJECT")
            return False, None

    # ------------------------------------------------------------------
    def authenticated_sift(self, alice_bases: np.ndarray, bob_bases: np.ndarray,
                           lost_mask: np.ndarray | None = None, mitm: str | None = None) -> dict:
        """Full two-way authenticated sifting exchange.

        1. Bob announces his bases (-1 where he detected nothing), signed.
        2. Alice verifies, then announces her bases, signed.
        3. Bob verifies.  Both keep positions with equal bases.
        `mitm` in {None, 'tamper', 'forge', 'replay'} injects a classical attacker.
        """
        if not self.alice_pk:
            self.handshake()
        bob_announce = np.asarray(bob_bases, dtype=np.int8).copy()
        if lost_mask is not None:
            bob_announce[np.asarray(lost_mask, bool)] = -1

        pkt_b = self.send_bases(bob_announce, "bob")
        if mitm:
            pkt_b = simulate_mitm(pkt_b, mode=mitm, lamport=self.lamport)
        ok_b, recv_bob = self.verify_bases(pkt_b, self.bob_pk)
        if not ok_b:
            return {"pqc_verified": False, "failed_at": "bob->alice", "matching_mask": None,
                    "log": list(self.log)}

        pkt_a = self.send_bases(np.asarray(alice_bases, dtype=np.int8), "alice")
        ok_a, recv_alice = self.verify_bases(pkt_a, self.alice_pk)
        if not ok_a:
            return {"pqc_verified": False, "failed_at": "alice->bob", "matching_mask": None,
                    "log": list(self.log)}
        matching = (recv_alice == recv_bob) & (recv_bob >= 0)
        return {"pqc_verified": True, "failed_at": None, "matching_mask": matching,
                "signature_bytes": LamportOTS.signature_size_bytes(),
                "keys_remaining": self.keys_remaining(), "log": list(self.log)}


def simulate_mitm(packet: dict, mode: str = "tamper", lamport: LamportOTS | None = None,
                  flip_fraction: float = 0.05, seed: int | None = None) -> dict:
    """Classical man-in-the-middle on the basis announcement.

    tamper : modify bases in the message, keep the original signature
    forge  : modify bases and re-sign with the attacker's own Lamport key
    replay : re-send the message with a signature over a different message
    None of these can produce a valid signature under the honest public key.
    """
    lamport = lamport or LamportOTS()
    rng = np.random.default_rng(seed)
    sender, round_no, bases = deserialize_bases(packet["message"])
    forged = bases.copy()
    n_flip = max(1, int(flip_fraction * forged.size))
    idx = rng.choice(forged.size, n_flip, replace=False)
    valid = forged[idx] >= 0
    forged[idx[valid]] = 1 - forged[idx[valid]]
    new_msg = serialize_bases(forged, sender, round_no)
    out = dict(packet)
    out["message"] = new_msg
    if mode == "tamper":
        pass
    elif mode == "forge":
        sk, _ = lamport.keygen()
        out["signature"] = lamport.sign(new_msg, sk)
    elif mode == "replay":
        sk, _ = lamport.keygen()
        out["signature"] = lamport.sign(b"some earlier message", sk)
    else:
        raise ValueError("mode must be 'tamper', 'forge' or 'replay'")
    out["mitm"] = {"mode": mode, "n_modified": int(valid.sum())}
    return out
