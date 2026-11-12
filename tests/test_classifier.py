import numpy as np

from config import settings as S
from ml.attack_classifier import AttackClassifier
from ml.feature_engine import extract_session_features
from quantum import BBM92Session
from tests.conftest import requires_model


@requires_model
def test_model_loads():
    clf = AttackClassifier("xgboost").load(str(S.XGBOOST_MODEL_PATH))
    x, _ = extract_session_features(BBM92Session(seed=1).run())
    cls, proba = clf.predict(x)
    assert proba.shape == (8,)
    assert np.isclose(proba.sum(), 1.0, atol=1e-5)
    assert 0 <= cls < 8
    cls_b, proba_b = clf.predict(np.vstack([x, x]))
    assert proba_b.shape == (2, 8)


@requires_model
def test_normal_session():
    clf = AttackClassifier("xgboost").load(str(S.XGBOOST_MODEL_PATH))
    x, _ = extract_session_features(BBM92Session(seed=2).run())
    cls, proba = clf.predict(x)
    assert cls == 0 and proba.max() > 0.8


@requires_model
def test_cloning_detection():
    clf = AttackClassifier("xgboost").load(str(S.XGBOOST_MODEL_PATH))
    hits = 0
    for s in range(5):
        x, _ = extract_session_features(BBM92Session(attack_type="quantum_cloning",
                                                    attack_params={"clone_fidelity": 0.65},
                                                    seed=100 + s).run())
        hits += clf.predict(x)[0] == 6
    assert hits >= 4


@requires_model
def test_other_models_load():
    import os
    for mt, p in (("random_forest", S.RF_MODEL_PATH), ("svm_rbf", S.SVM_MODEL_PATH)):
        if os.path.exists(p):
            clf = AttackClassifier(mt).load(str(p))
            x, _ = extract_session_features(BBM92Session(seed=3).run())
            assert clf.predict(x)[1].shape == (8,)


def test_train_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 63))
    y = np.repeat(np.arange(8), 50)
    X[:, 0] += y                                    # make it learnable
    clf = AttackClassifier("xgboost")
    clf.train(X, y, search="none", params={"n_estimators": 30, "max_depth": 3})
    path = tmp_path / "m.json"
    clf.save(str(path), scaler_path=str(tmp_path / "scaler.pkl"))
    clf2 = AttackClassifier().load(str(path), scaler_path=str(tmp_path / "scaler.pkl"))
    assert np.allclose(clf.predict(X[:5])[1], clf2.predict(X[:5])[1], atol=1e-6)
