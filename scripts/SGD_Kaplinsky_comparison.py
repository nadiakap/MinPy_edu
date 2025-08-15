# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:17:39 2025

@author: nadia
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# ====== Utility: gradient orthogonalization (Kaplinsky-like idea) ======
def project_gradients(g_new, g_old):
    """
    Project new gradients g_new to be orthogonal to g_old.
    Both are flattened numpy arrays.
    """
    if np.linalg.norm(g_old) == 0:
        return g_new
    projection = (np.dot(g_new, g_old) / np.dot(g_old, g_old)) * g_old
    return g_new - projection

# ====== Create two sequential tasks ======
np.random.seed(42)
XA, yA = make_moons(n_samples=500, noise=0.2)

theta = np.pi / 2
rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
XB = XA @ rot_matrix.T
yB = yA

# ====== Train with standard SGD ======
sgd = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1, warm_start=True,
                    solver='sgd', learning_rate_init=0.05, random_state=42)

epochs_A = 20
epochs_B = 20

acc_A_sgd, acc_B_sgd = [], []

# Train on Task A
for _ in range(epochs_A):
    sgd.fit(XA, yA)
    acc_A_sgd.append((accuracy_score(yA, sgd.predict(XA)),
                      accuracy_score(yB, sgd.predict(XB))))

# Train on Task B
for _ in range(epochs_B):
    sgd.fit(XB, yB)
    acc_A_sgd.append((accuracy_score(yA, sgd.predict(XA)),
                      accuracy_score(yB, sgd.predict(XB))))

# ====== Train with simplified Kaplinsky-like algorithm ======
# We'll manually implement gradient orthogonalization on top of sklearn MLP
# Note: scikit-learn doesn't expose per-sample gradients, so we'll simulate it
# by using partial_fit and tracking the old-task gradient direction

kapl = MLPClassifier(hidden_layer_sizes=(20,), max_iter=1, warm_start=True,
                     solver='sgd', learning_rate_init=0.05, random_state=42)

acc_A_kapl, acc_B_kapl = [], []

# Train on Task A (store its gradient direction)
grad_A_direction = None
for _ in range(epochs_A):
    kapl.fit(XA, yA)
    # Fake "gradient direction" by difference in weights
    if grad_A_direction is None:
        grad_A_direction = [np.zeros_like(coef) for coef in kapl.coefs_]
    else:
        for i, coef in enumerate(kapl.coefs_):
            grad_A_direction[i] = coef - grad_A_prev[i]
    grad_A_prev = [coef.copy() for coef in kapl.coefs_]
    acc_A_kapl.append((accuracy_score(yA, kapl.predict(XA)),
                       accuracy_score(yB, kapl.predict(XB))))

# Train on Task B with orthogonalization
for _ in range(epochs_B):
    prev_weights = [coef.copy() for coef in kapl.coefs_]
    kapl.fit(XB, yB)
    # Get "new gradient" = change in weights
    grad_new = [coef - prev for coef, prev in zip(kapl.coefs_, prev_weights)]
    # Project each layer's gradient orthogonal to old-task gradient
    for i in range(len(kapl.coefs_)):
        g_new_flat = grad_new[i].flatten()
        g_old_flat = grad_A_direction[i].flatten()
        g_proj_flat = project_gradients(g_new_flat, g_old_flat)
        g_proj = g_proj_flat.reshape(kapl.coefs_[i].shape)
        kapl.coefs_[i] = prev_weights[i] + g_proj
    acc_A_kapl.append((accuracy_score(yA, kapl.predict(XA)),
                       accuracy_score(yB, kapl.predict(XB))))

# ====== Plot comparison ======
plt.figure(figsize=(10,4))

# SGD plot
plt.subplot(1,2,1)
acc_A_sgd_plot = [a[0] for a in acc_A_sgd]
acc_B_sgd_plot = [a[1] for a in acc_A_sgd]
plt.plot(range(epochs_A + epochs_B), acc_A_sgd_plot, label='Task A')
plt.plot(range(epochs_A + epochs_B), acc_B_sgd_plot, label='Task B')
plt.axvline(epochs_A, color='gray', linestyle='--', label="Switch to Task B")
plt.title("Standard SGD")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Kaplinsky-like plot
plt.subplot(1,2,2)
acc_A_kapl_plot = [a[0] for a in acc_A_kapl]
acc_B_kapl_plot = [a[1] for a in acc_A_kapl]
plt.plot(range(epochs_A + epochs_B), acc_A_kapl_plot, label='Task A')
plt.plot(range(epochs_A + epochs_B), acc_B_kapl_plot, label='Task B')
plt.axvline(epochs_A, color='gray', linestyle='--', label="Switch to Task B")
plt.title("Kaplinsky-like Projection")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()
