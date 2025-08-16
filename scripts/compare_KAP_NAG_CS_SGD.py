# -*- coding: utf-8 -*-
"""
Comparison of Nesterov method of momentum,
Polyak method of momentum (heavy ball)
Kaplinskiy method
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------
# Data: Two regimes
# -------------------------
X1, y1 = make_moons(n_samples=500, noise=0.2, random_state=0)   # Task A
X2, y2 = make_moons(n_samples=500, noise=0.2, random_state=1)   # Task B (shifted)
X2 = np.dot(X2, [[0, -1], [1, 0]])   # rotate 90 degrees

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
X2 = scaler.transform(X2)

# -------------------------
# Model: simple logistic regression
# -------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def predict(X, w):
    return (sigmoid(X @ w) > 0.5).astype(int)

# Loss gradient
def grad(X, y, w):
    preds = sigmoid(X @ w)
    return X.T @ (preds - y) / len(y)

# -------------------------
# Optimizers
# -------------------------
def train_polyak(X, y, w, lr=0.1, beta=0.9, epochs=50):
    v = np.zeros_like(w)
    for _ in range(epochs):
        g = grad(X, y, w)
        v = beta * v - lr * g
        w = w + v
    return w

def train_nesterov(X, y, w, lr=0.1, beta=0.9, epochs=50):
    v = np.zeros_like(w)
    for _ in range(epochs):
        g = grad(X, y, w + beta * v)  # lookahead
        v = beta * v - lr * g
        w = w + v
    return w

def train_kaplinsky(X, y, w, lr=0.1, lam=0.1, w_ref=None, epochs=50):
    """
    Kaplinsky-inspired: minimize loss + Lyapunov regularizer
    Keeps w close to previous reference (conservation law).
    """
    if w_ref is None:
        w_ref = np.copy(w)
    for _ in range(epochs):
        g = grad(X, y, w)
        # conservation correction term
        g_total = g + lam * (w - w_ref)
        w = w - lr * g_total
    return w

# -------------------------
# Experiment
# -------------------------
def run_experiment(train_fn, name):
    w = np.zeros(X1.shape[1])  # init
    
    # Train on Task A
    wA = train_fn(X1, y1, w)
    acc_A_before = accuracy_score(y1, predict(X1, wA))
    
    # Train on Task B sequentially
    if "kaplinsky" in train_fn.__name__:
        wB = train_fn(X2, y2, wA, w_ref=wA)  # keep memory
    else:
        wB = train_fn(X2, y2, wA)
    acc_A_after = accuracy_score(y1, predict(X1, wB))
    acc_B = accuracy_score(y2, predict(X2, wB))
    
    return acc_A_before, acc_A_after, acc_B

results = {}
results["Polyak"] = run_experiment(train_polyak, "Polyak")
results["Nesterov"] = run_experiment(train_nesterov, "Nesterov")
results["Kaplinsky"] = run_experiment(train_kaplinsky, "Kaplinsky")

# -------------------------
# Plot Results
# -------------------------
labels = list(results.keys())
acc_A_before = [results[m][0] for m in labels]
acc_A_after  = [results[m][1] for m in labels]
acc_B        = [results[m][2] for m in labels]

x = np.arange(len(labels))
width = 0.25

plt.bar(x - width, acc_A_before, width, label='Task A before Task B')
plt.bar(x, acc_A_after, width, label='Task A after Task B')
plt.bar(x + width, acc_B, width, label='Task B')

plt.xticks(x, labels)
plt.ylabel("Accuracy")
plt.title("Catastrophic Forgetting Comparison")
plt.legend()
plt.ylim(0,1)
plt.show()
