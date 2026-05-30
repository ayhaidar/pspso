from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from pspso import pspso


def _diabetes_split():
    X, y = load_diabetes(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X[:80], y[:80], test_size=0.25, random_state=42
    )
    scaler = MinMaxScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), y_train, y_val


def test_legacy_import_and_grid_return_shape():
    X_train, X_val, y_train, y_val = _diabetes_split()
    params = {"kernel": ["linear"], "C": [0.1, 0.2, 1], "gamma": [0.1, 0.1, 1]}
    legacy = pspso(estimator="svm", params=params, task="regression", score="rmse")

    pos, cost, duration, model, combinations, results = legacy.fitpsgrid(
        X_train, y_train, X_val, y_val
    )

    assert pos is not None
    assert cost is not None
    assert duration is not None
    assert model is not None
    assert len(combinations) == 2
    assert len(results) == 2


def test_legacy_instances_keep_their_own_search_space():
    a = pspso(
        estimator="svm",
        params={"kernel": ["linear"], "C": [0.1, 0.1, 1]},
        task="regression",
        score="rmse",
    )
    b = pspso(
        estimator="svm",
        params={"kernel": ["rbf"], "C": [1, 1, 0], "gamma": [1, 1, 0]},
        task="regression",
        score="rmse",
    )

    assert b.decode_position([0, 1, 1]) == {"kernel": "rbf", "C": 1, "gamma": 1}
    assert a.decode_position([0, 0.1]) == {"kernel": "linear", "C": 0.1}
