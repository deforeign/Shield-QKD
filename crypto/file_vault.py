"""
AES-256-GCM encryption of messages and files with QKD-derived one-time keys.

Container format:  b"SQKDV1" | key_id (12 ascii) | nonce (12) | ciphertext||tag
The key_id is not secret; it tells the receiver which QKD key to consume.
Alice's copy is consumed on encryption and Bob's on decryption, so every key
encrypts exactly one payload (forward secrecy).
"""
from __future__ import annotations

import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from crypto.key_manager import KeyManager

MAGIC = b"SQKDV1"
KEY_ID_LEN = 12
NONCE_LEN = 12


class FileVault:
    def __init__(self, key_manager: KeyManager):
        self.km = key_manager

    def encrypt_bytes(self, plaintext: bytes, key_id: str, party: str = "alice",
                      aad: bytes = b"ShieldQKD") -> bytes:
        key = self.km.consume(key_id, party)
        nonce = secrets.token_bytes(NONCE_LEN)
        ct = AESGCM(key).encrypt(nonce, plaintext, aad)
        return MAGIC + key_id.encode("ascii").ljust(KEY_ID_LEN, b"_") + nonce + ct

    @staticmethod
    def parse(blob: bytes) -> tuple[str, bytes, bytes]:
        if not blob.startswith(MAGIC):
            raise ValueError("not a ShieldQKD vault container")
        off = len(MAGIC)
        key_id = blob[off:off + KEY_ID_LEN].decode("ascii").rstrip("_")
        nonce = blob[off + KEY_ID_LEN: off + KEY_ID_LEN + NONCE_LEN]
        return key_id, nonce, blob[off + KEY_ID_LEN + NONCE_LEN:]

    def decrypt_bytes(self, blob: bytes, party: str = "bob", aad: bytes = b"ShieldQKD") -> bytes:
        key_id, nonce, ct = self.parse(blob)
        key = self.km.consume(key_id, party)
        return AESGCM(key).decrypt(nonce, ct, aad)

    def encrypt_file(self, in_path: str, key_id: str, out_path: str | None = None) -> str:
        out_path = out_path or in_path + ".sqkd"
        with open(in_path, "rb") as fh:
            data = fh.read()
        blob = self.encrypt_bytes(data, key_id, "alice", aad=os.path.basename(in_path).encode())
        with open(out_path, "wb") as fh:
            fh.write(blob)
        return out_path

    def decrypt_file(self, in_path: str, original_name: str, out_path: str) -> str:
        with open(in_path, "rb") as fh:
            blob = fh.read()
        data = self.decrypt_bytes(blob, "bob", aad=original_name.encode())
        with open(out_path, "wb") as fh:
            fh.write(data)
        return out_path
