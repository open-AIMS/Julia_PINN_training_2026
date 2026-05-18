"""
Random-forest baseline on MNIST in scikit-learn.

Cross-language pair with mnist_rf_julia.jl. Loads the standard
60k/10k split via fetch_openml (cached to ~/scikit_learn_data
after first run), fits a 100-tree forest, and prints test accuracy.

Run via ./build.sh execute 2 — writes output to ../output/mnist_rf_sklearn.md.
"""

from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X = X.astype("float32") / 255.0
y = y.astype("int64")

X_train, X_test = X[:60_000], X[60_000:]
y_train, y_test = y[:60_000], y[60_000:]

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=0)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"MNIST test accuracy (sklearn RF, 100 trees): {acc:.4f}")
