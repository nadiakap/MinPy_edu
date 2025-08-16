import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# -------------------------
# Data: two regimes (Task A -> Task B)
# -------------------------
X1, y1 = make_moons(n_samples=600, noise=0.22, random_state=0)   # Task A
X2, y2 = make_moons(n_samples=600, noise=0.22, random_state=1)   # Task B
# Rotate Task B by 90 degrees to change the regime
R = np.array([[0, -1],[1, 0]])
X2 = X2 @ R.T

scaler = StandardScaler()
X1 = scaler.fit_transform(X1)
X2 = scaler.transform(X2)

# Add bias term (so model is w in R^3: [x1,x2,1])
def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0],1))])

X1b = add_bias(X1)
X2b = add_bias(X2)

# -------------------------
# Logistic model + loss/grad
# -------------------------
def sigmoid(z): return 1.0/(1.0 + np.exp(-z))

def loss_logistic(X, y, w):
    z = X @ w
    # stable logistic loss
    return np.mean(np.log1p(np.exp(- (2*y-1)*z)))

def grad_logistic(X, y, w):
    # gradient of logistic loss (labels 0/1)
    z = X @ w
    p = sigmoid(z)
    return (X.T @ (p - y)) / y.size

def predict_labels(X, w):
    return (sigmoid(X @ w) >= 0.5).astype(int)

# -------------------------
# Optimizers
# -------------------------
def train_polyak(X, y, w, lr=0.1, beta=0.9, epochs=80):
    """Polyak heavy-ball momentum."""
    v = np.zeros_like(w)
    for _ in range(epochs):
        g = grad_logistic(X, y, w)
        v = beta * v - lr * g
        w = w + v
    return w

def train_nesterov(X, y, w, lr=0.1, beta=0.9, epochs=80):
    """Nesterov accelerated gradient."""
    v = np.zeros_like(w)
    for _ in range(epochs):
        g = grad_logistic(X, y, w + beta * v)  # look-ahead gradient
        v = beta * v - lr * g
        w = w + v
    return w

def train_shor_diminishing(X, y, w, a=1.0, epochs=80):
    """
    Naum Shor–style subgradient method with diminishing step sizes:
      alpha_t = a / sqrt(t+1).
    For smooth losses (like logistic), subgradient = gradient.
    """
    for t in range(epochs):
        g = grad_logistic(X, y, w)
        step = a / np.sqrt(t + 1.0)
        w = w - step * g
    return w

def train_kaplinsky_like(X, y, w, lr=0.1, lam=0.2, w_ref=None, epochs=80):
    """
    Kaplinsky-inspired Lyapunov/proximal stabilization:
      minimize L(X,y;w) + (lam/2)||w - w_ref||^2
    This resists drift from Task-A solution (prevents forgetting).
    """
    if w_ref is None:
        w_ref = w.copy()
    for _ in range(epochs):
        g = grad_logistic(X, y, w)
        g_total = g + lam * (w - w_ref)  # Lyapunov/proximal term
        w = w - lr * g_total
    return w

# -------------------------
# Experiment harness
# -------------------------
def run_seq(train_fn, name, extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    w0 = np.zeros(X1b.shape[1])

    # Train on Task A
    wA = train_fn(X1b, y1, w0.copy(), **extra_kwargs)
    accA_before = accuracy_score(y1, predict_labels(X1b, wA))
    lossA_before = loss_logistic(X1b, y1, wA)

    # Train on Task B (sequential)
    if train_fn is train_kaplinsky_like:
        # keep reference to Task-A weights
        wB = train_fn(X2b, y2, wA.copy(), w_ref=wA.copy(), **extra_kwargs)
    else:
        wB = train_fn(X2b, y2, wA.copy(), **extra_kwargs)

    accA_after = accuracy_score(y1, predict_labels(X1b, wB))
    accB       = accuracy_score(y2, predict_labels(X2b, wB))
    lossA_after = loss_logistic(X1b, y1, wB)
    lossB       = loss_logistic(X2b, y2, wB)

    return {
        "name": name,
        "A_before_acc": accA_before,
        "A_after_acc": accA_after,
        "B_acc": accB,
        "A_before_loss": lossA_before,
        "A_after_loss": lossA_after,
        "B_loss": lossB
    }

results = []
results.append(run_seq(train_polyak, "Polyak (Heavy Ball)", dict(lr=0.15, beta=0.9, epochs=120)))
results.append(run_seq(train_nesterov, "Nesterov", dict(lr=0.12, beta=0.9, epochs=120)))
results.append(run_seq(train_shor_diminishing, "Shor (diminishing)", dict(a=0.8, epochs=160)))
results.append(run_seq(train_kaplinsky_like, "Kaplinsky-like", dict(lr=0.12, lam=0.25, epochs=120)))

# -------------------------
# Display results
# -------------------------
labels = [r["name"] for r in results]
acc_A_before = [r["A_before_acc"] for r in results]
acc_A_after  = [r["A_after_acc"]  for r in results]
acc_B        = [r["B_acc"]        for r in results]

x = np.arange(len(labels)); width = 0.27
plt.figure(figsize=(9,4))
plt.bar(x - width, acc_A_before, width, label="Task A (before B)")
plt.bar(x,         acc_A_after,  width, label="Task A (after B)")
plt.bar(x + width, acc_B,        width, label="Task B (final)")
plt.xticks(x, labels, rotation=10)
plt.ylim(0,1); plt.ylabel("Accuracy")
plt.title("Changing-Regime Sequential Training: Forgetting Comparison")
plt.legend()
plt.tight_layout()
plt.show()

# Print a compact table
print("\n=== Summary ===")
for r in results:
    print(f"{r['name']:<20}  A_before={r['A_before_acc']:.3f}  A_after={r['A_after_acc']:.3f}  "
          f"B={r['B_acc']:.3f}  ΔA={(r['A_after_acc']-r['A_before_acc']):+.3f}")
