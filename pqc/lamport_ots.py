"""
Lamport One-Time Signature (OTS) over SHA-256 (spec Section 6.1).

Security rests only on the preimage resistance of SHA-256, which Shor's
algorithm does not break (Grover gives at most a square-root speed-up, leaving
~128-bit post-quantum security).  Each private key MUST sign only one message:
signing reveals half of the secret values.

    keygen:  sk[i] = (sk0_i, sk1_i)  256 pairs of 256-bit random values
             pk[i] = (H(sk0_i), H(sk1_i))
    sign:    h = H(m); sig[i] = sk_{h_i}_i
    verify:  H(sig[i]) == pk_{h_i}_i  for all i
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

N_BITS = 256
VALUE_BYTES = 32


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _bits(digest: bytes):
    """Most-significant-bit-first bits of a 32-byte digest."""
    for byte in digest:
        for k in range(7, -1, -1):
            yield (byte >> k) & 1


class LamportOTS:
    def keygen(self) -> tuple[list, list]:
        """Return (private_key, public_key), each a list of 256 (v0, v1) pairs."""
        sk = [(secrets.token_bytes(VALUE_BYTES), secrets.token_bytes(VALUE_BYTES))
              for _ in range(N_BITS)]
        pk = [(_h(a), _h(b)) for a, b in sk]
        return sk, pk

    def sign(self, message: bytes, private_key: list) -> list:
        if len(private_key) != N_BITS:
            raise ValueError("invalid Lamport private key")
        digest = _h(message)
        return [private_key[i][bit] for i, bit in enumerate(_bits(digest))]

    def verify(self, message: bytes, signature: list, public_key: list) -> bool:
        if not isinstance(signature, (list, tuple)) or len(signature) != N_BITS \
                or len(public_key) != N_BITS:
            return False
        digest = _h(message)
        ok = True
        for i, bit in enumerate(_bits(digest)):
            s = signature[i]
            if not isinstance(s, (bytes, bytearray)) or len(s) != VALUE_BYTES:
                return False
            # constant-time comparison; evaluate all 256 checks
            ok &= hmac.compare_digest(_h(bytes(s)), public_key[i][bit])
        return bool(ok)

    # -- helpers --------------------------------------------------------
    @staticmethod
    def public_key_fingerprint(public_key: list) -> str:
        return hashlib.sha256(b"".join(a + b for a, b in public_key)).hexdigest()

    @staticmethod
    def signature_size_bytes() -> int:
        return N_BITS * VALUE_BYTES          # 8 KiB

    @staticmethod
    def public_key_size_bytes() -> int:
        return 2 * N_BITS * VALUE_BYTES      # 16 KiB
