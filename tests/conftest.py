import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config import settings as S  # noqa: E402


def _have_model():
    return os.path.exists(S.XGBOOST_MODEL_PATH) and os.path.exists(S.SCALER_PATH)


requires_model = pytest.mark.skipif(not _have_model(),
                                    reason="trained model missing - run `python main.py train`")


@pytest.fixture(scope="session")
def breaker(tmp_path_factory):
    if not _have_model():
        pytest.skip("trained model missing")
    from security.circuit_breaker import CircuitBreaker
    b = CircuitBreaker(str(S.XGBOOST_MODEL_PATH))
    b.save_reports = False
    return b
