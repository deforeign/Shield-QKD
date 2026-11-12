"""Post-quantum authentication of the classical channel (Lamport OTS / SHA-256)."""
from pqc.authenticated_sifting import AuthenticatedSifting, simulate_mitm
from pqc.lamport_ots import LamportOTS

__all__ = ["LamportOTS", "AuthenticatedSifting", "simulate_mitm"]
