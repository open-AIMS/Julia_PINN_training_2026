"""
Random forest on the iris dataset, 5-fold cross-validated accuracy.

Run via the project venv:
    .venv/bin/python units/unit_02/scripts/iris_rf.py

Or via the build script:
    ./build.sh execute 2
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

clf = RandomForestClassifier(n_estimators=100, random_state=123)
scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
