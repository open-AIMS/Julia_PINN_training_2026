"""
Softmax-regression baseline on MNIST in scikit-learn.

The deep-learning-book Chapter 5 "linear baseline": a single softmax
layer trained on 28x28 pixel features. Sits at ~92% test accuracy,
which is what motivates depth and nonlinearity in §2.4.
"""

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X = X.astype("float32") / 255.0
y = y.astype("int64")

X_train, X_test = X[:60_000], X[60_000:]
y_train, y_test = y[:60_000], y[60_000:]

clf = LogisticRegression(
    solver="lbfgs", max_iter=200, n_jobs=-1, random_state=0,
)
clf.fit(X_train, y_train)
acc = accuracy_score(y_test, clf.predict(X_test))
print(f"MNIST test accuracy (softmax regression): {acc:.4f}")
