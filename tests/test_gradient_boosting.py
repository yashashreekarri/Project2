import numpy as np
from models.gradient_boosting import GradientBoostingClassifier
from data.synthetic_data import generate_synthetic_data

def test_gradient_boosting_binary():
    X, y = generate_synthetic_data(n_samples=200, n_classes=2)
    clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.8

def test_gradient_boosting_multiclass():
    X, y = generate_synthetic_data(n_samples=200, n_classes=3)
    clf = GradientBoostingClassifier(n_estimators=15, learning_rate=0.1, max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = np.mean(preds == y)
    assert acc > 0.7
