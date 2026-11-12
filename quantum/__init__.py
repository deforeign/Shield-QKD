"""Quantum layer: BBM92 simulator, noise channels and attacks."""
from quantum.bbm92_simulator import BBM92Session


def run_session(**kwargs) -> dict:
    """Convenience wrapper: run_session(attack_type='pns', seed=1)."""
    return BBM92Session(**kwargs).run()


__all__ = ["BBM92Session", "run_session"]
