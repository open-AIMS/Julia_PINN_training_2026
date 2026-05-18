"""
MLP on MNIST in scikit-learn.

Two hidden layers (256, 128), ReLU, Adam. Closes the gap left by the
softmax baseline in mnist_linear_sklearn.py — expect ~98% test
accuracy.
"""

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X = X.astype("float32") / 255.0
y = y.astype("int64")
X_train, X_test = X[:60_000], X[60_000:]
y_train, y_test = y[:60_000], y[60_000:]

clf = MLPClassifier(
    hidden_layer_sizes=(256, 128), activation="relu",
    solver="adam", batch_size=128, max_iter=20, random_state=0,
)
clf.fit(X_train, y_train)
print(f"MNIST test accuracy (sklearn MLP): "
      f"{accuracy_score(y_test, clf.predict(X_test)):.4f}")
