import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------
# Generate multi-regime dataset (4 sequential tasks)
# -------------------------
def generate_regime(seed, shift_angle=0):
    X, y = make_moons(n_samples=400, noise=0.25, random_state=seed)
    if shift_angle != 0:
        theta = np.radians(shift_angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
        X = X @ R.T
    return X, y

regimes = [
    generate_regime(0, shift_angle=0),     # Task 1
    generate_regime(1, shift_angle=45),    # Task 2
    generate_regime(2, shift_angle=90),    # Task 3
    generate_regime(3, shift_angle=135),   # Task 4
]

# Standardize all regimes
scaler = StandardScaler()
X_all = [scaler.fit_transform(X) for X, y in regimes]
y_all = [y for X, y in regimes]

# Add bias term for logistic regression
X_all_b = [np.hstack([X, np.ones((X.shape[0],1))]) for X in X_all]

# -------------------------
# Logistic model and gradients
# -------------------------
def sigmoid(z): 
    return 1.0 / (1.0 + np.exp(-z))

def grad_logistic(X, y, w):
    z = X @ w
    p = sigmoid(z)
    return X.T @ (p - y) / y.size

def predict_labels(X, w):
    return (sigmoid(X @ w) >= 0.5).astype(int)

# -------------------------
# Optimizers
# -------------------------
def train_polyak(X, y, w, lr=0.1, beta=0.9, epochs=80):
    v = np.zeros_like(w)
    for _ in range(epochs):
        g = grad_logistic(X, y, w)
        v = beta * v - lr * g
        w = w + v
    return w

def train_nesterov(X, y, w, lr=0.1, beta=0.9, epochs=80):
    v = np.zeros_like(w)
    for _ in range(epochs):
        g = grad_logistic(X, y, w + beta * v)
        v = beta * v - lr * g
        w = w + v
    return w

def train_shor_diminishing(X, y, w, a=1.0, epochs=80):
    for t in range(epochs):
        g = grad_logistic(X, y, w)
        step = a / np.sqrt(t + 1.0)
        w = w - step * g
    return w

def train_kaplinsky_like(X, y, w, lr=0.1, lam=0.2, w_ref=None, epochs=80):
    if w_ref is None:
        w_ref = w.copy()
    for _ in range(epochs):
        g = grad_logistic(X, y, w)
        g_total = g + lam * (w - w_ref)
        w = w - lr * g_total
    return w

# -------------------------
# Sequential training across multiple regimes
# -------------------------
def run_multi_regimes(train_fn, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    w = np.zeros(X_all_b[0].shape[1])
    accs = []
    for i, (Xb, y) in enumerate(zip(X_all_b, y_all)):
        if train_fn is train_kaplinsky_like and i > 0:
            w = train_fn(Xb, y, w, w_ref=w_ref.copy(), **extra_kwargs)
        else:
            w = train_fn(Xb, y, w.copy(), **extra_kwargs)
        acc_task = [accuracy_score(y_all[j], predict_labels(X_all_b[j], w)) for j in range(len(regimes))]
        accs.append(acc_task)
        # For Kaplinsky, keep reference to Task 1 weights
        if i == 0 and train_fn is train_kaplinsky_like:
            w_ref = w.copy()
    return np.array(accs)

# -------------------------
# Run experiments
# -------------------------
results_polyak   = run_multi_regimes(train_polyak, dict(lr=0.15, beta=0.9, epochs=80))
results_nesterov = run_multi_regimes(train_nesterov, dict(lr=0.12, beta=0.9, epochs=80))
results_shor     = run_multi_regimes(train_shor_diminishing, dict(a=0.8, epochs=100))
results_kaplinsky= run_multi_regimes(train_kaplinsky_like, dict(lr=0.12, lam=0.25, epochs=80))

# -------------------------
# Plot forgetting curves
# -------------------------
tasks = [f"Task {i+1}" for i in range(len(regimes))]
plt.figure(figsize=(10,6))

for accs, name in zip(
        [results_polyak, results_nesterov, results_shor, results_kaplinsky],
        ["Polyak", "Nesterov", "Shor", "Kaplinsky"]):
    for t in range(len(regimes)):
        plt.plot(accs[:,t], label=f"{name} - {tasks[t]}", marker='o')

plt.xlabel("Regime sequence (training)")
plt.ylabel("Accuracy on each task")
plt.title("Sequential Learning Across Multiple Regimes")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()

# -------------------------
# Print summary table
# -------------------------
print("\n=== Accuracy Summary ===")
for name, accs in zip(["Polyak","Nesterov","Shor","Kaplinsky"],
                      [results_polyak, results_nesterov, results_shor, results_kaplinsky]):
    print(f"{name}:")
    for i, row in enumerate(accs):
        print(f"  After Task {i+1}: " + ", ".join([f"{v:.3f}" for v in row]))

