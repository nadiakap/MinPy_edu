# -*- coding: utf-8 -*-

"""
comparing Stochastic Gradient Descent and Kaplinsliy hybrid
how these algorithms overcome catastrophic forgetting in data with changing regimes
"""


# -*- coding: utf-8 -*-
# Demonstration: Catastrophic forgetting with SGD vs. Kaplinsky-style potential descent
# Requires: torch, matplotlib, scikit-learn
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1) Data: Two sequential tasks with different geometry
#    Task A: Two moons; Task B: Rotated moons (different decision boundary)
# -------------------------
XA, yA = make_moons(n_samples=1200, noise=0.2)
XA_tr, XA_te, yA_tr, yA_te = train_test_split(XA, yA, test_size=0.3, random_state=42)

theta = np.pi/2  # 90 degrees rotation
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
XB = XA @ R.T
XB_tr, XB_te, yB_tr, yB_te = train_test_split(XB, yA, test_size=0.3, random_state=42)

def to_tensor(X, y):
    X = torch.tensor(X, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return X, y

XA_tr_t, yA_tr_t = to_tensor(XA_tr, yA_tr)
XA_te_t, yA_te_t = to_tensor(XA_te, yA_te)
XB_tr_t, yB_tr_t = to_tensor(XB_tr, yB_tr)
XB_te_t, yB_te_t = to_tensor(XB_te, yB_te)

# -------------------------
# 2) Simple MLP classifier
# -------------------------
class MLP(nn.Module):
    def __init__(self, d_in=2, d_h=64, d_out=2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_h)
        self.fc2 = nn.Linear(d_h, d_h)
        self.fc3 = nn.Linear(d_h, d_out)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def accuracy(model, X, y):
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        return (preds == y).float().mean().item()

# -------------------------
# 3) Utilities to “flatten/unflatten” parameters and gradients
# -------------------------
def flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

def flat_grads(model):
    g = []
    for p in model.parameters():
        if p.grad is None:
            g.append(torch.zeros_like(p).flatten())
        else:
            g.append(p.grad.detach().flatten())
    return torch.cat(g)

def unflat_like(vec, model):
    out = []
    i = 0
    for p in model.parameters():
        n = p.numel()
        out.append(vec[i:i+n].view_as(p))
        i += n
    return out

def set_params_from_flat(model, flat_vec):
    chunks = unflat_like(flat_vec, model)
    with torch.no_grad():
        for p, c in zip(model.parameters(), chunks):
            p.copy_(c)

# -------------------------
# 4) Estimate diagonal Fisher (per-parameter importance) on Task A
#    (Used in Φ as stability term around w_A*)
# -------------------------
def estimate_fisher_diag(model, X, y, n_samples=2048):
    model.eval()
    idx = torch.randint(low=0, high=X.size(0), size=(min(n_samples, X.size(0)),), device=device)
    Xs, ys = X[idx], y[idx]
    logits = model(Xs)
    log_probs = F.log_softmax(logits, dim=1)
    # Sample labels from model's predicted distribution (common in Fisher approx)
    with torch.no_grad():
        probs = log_probs.exp()
        y_sample = torch.multinomial(probs, num_samples=1).squeeze(1)
    # Negative log-likelihood of sampled labels
    nll = F.nll_loss(log_probs, y_sample, reduction='mean')
    model.zero_grad()
    nll.backward()
    g = flat_grads(model)
    return g.pow(2)  # diagonal Fisher approximation

# -------------------------
# 5) Kaplinsky-style potential Φ and gradient direction
#    Φ(w) = L_B(w) + (λ/2)*(w - w_A*)^T F (w - w_A*)
#    plus gradient orthogonalization against ∇L_A to avoid first-order forgetting
# -------------------------
def task_loss(model, X, y):
    logits = model(X)
    return F.cross_entropy(logits, y, reduction='mean')

def phi_potential(model, X_B, y_B, wA_star, Fisher_diag, lam):
    Lb = task_loss(model, X_B, y_B)
    w = flat_params(model)
    diff = w - wA_star
    stab = 0.5 * lam * (Fisher_diag * diff.pow(2)).sum()
    return Lb + stab

def grad_of(loss, model):
    model.zero_grad()
    loss.backward()
    return flat_grads(model)

def project_orthogonal(g_new, g_old, eps=1e-12):
    denom = torch.dot(g_old, g_old).clamp_min(eps)
    proj = torch.dot(g_new, g_old) / denom * g_old
    return g_new - proj

# -------------------------
# 6) Training: (A) Plain sequential SGD and (B) Potential-descent
# -------------------------
def train_plain_sgd():
    model = MLP().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
    E_A, E_B = 50, 50
    accA_hist, accB_hist = [], []
    # Phase 1: Task A
    for _ in range(E_A):
        model.train()
        lossA = task_loss(model, XA_tr_t, yA_tr_t)
        opt.zero_grad(); lossA.backward(); opt.step()
        accA_hist.append(accuracy(model, XA_te_t, yA_te_t))
        accB_hist.append(accuracy(model, XB_te_t, yB_te_t))
    # Phase 2: Task B (no access to A)
    for _ in range(E_B):
        model.train()
        lossB = task_loss(model, XB_tr_t, yB_tr_t)
        opt.zero_grad(); lossB.backward(); opt.step()
        accA_hist.append(accuracy(model, XA_te_t, yA_te_t))
        accB_hist.append(accuracy(model, XB_te_t, yB_te_t))
    return accA_hist, accB_hist

def train_kaplinsky_style(lam=50.0, lr_init=0.2, backtrack=0.5, max_bt=15):
    """
    Kaplinsky-style training on B:
    - Potential Φ with Fisher-weighted stability about w_A*
    - Directional search: orthogonalize ∇Φ to old-task gradient ∇L_A
    - Adaptive step: backtracking line search ensuring Φ decreases
    """
    model = MLP().to(device)
    # --- Phase 1: Train on A to get w_A*
    optA = torch.optim.SGD(model.parameters(), lr=0.05)
    E_A, E_B = 50, 50
    accA_hist, accB_hist = [], []
    for _ in range(E_A):
        lossA = task_loss(model, XA_tr_t, yA_tr_t)
        optA.zero_grad(); lossA.backward(); optA.step()
        accA_hist.append(accuracy(model, XA_te_t, yA_te_t))
        accB_hist.append(accuracy(model, XB_te_t, yB_te_t))
    # Snapshot w_A* and Fisher
    wA_star = flat_params(model).detach()
    Fdiag = estimate_fisher_diag(model, XA_tr_t, yA_tr_t) + 1e-8  # avoid zeros

    # Pre-compute an "old-task gradient direction" at w_A*
    LA_at_wA = task_loss(model, XA_tr_t, yA_tr_t)
    gA = grad_of(LA_at_wA, model).detach()

    # --- Phase 2: Train on B with Φ, projection, and backtracking
    for _ in range(E_B):
        # Compute ∇Φ(w) at current w
        cur_phi = phi_potential(model, XB_tr_t, yB_tr_t, wA_star, Fdiag, lam)
        gPhi = grad_of(cur_phi, model)

        # Orthogonalize ∇Φ relative to gA to remove first-order interference
        g_dir = project_orthogonal(gPhi, gA)

        # Backtracking line search to ensure Φ decrease (Lyapunov-like)
        w0 = flat_params(model).detach()
        phi0 = cur_phi.item()
        step = lr_init
        accepted = False
        for _bt in range(max_bt):
            w_new = w0 - step * g_dir
            set_params_from_flat(model, w_new)
            with torch.no_grad():
                phi_new = phi_potential(model, XB_tr_t, yB_tr_t, wA_star, Fdiag, lam).item()
            if phi_new < phi0:  # monotone decrease
                accepted = True
                break
            step *= backtrack  # shrink step
        if not accepted:
            # fallback: tiny step to keep stability
            set_params_from_flat(model, w0 - (lr_init * (backtrack**max_bt)) * g_dir)

        # Track test accuracies
        accA_hist.append(accuracy(model, XA_te_t, yA_te_t))
        accB_hist.append(accuracy(model, XB_te_t, yB_te_t))

    return accA_hist, accB_hist

# -------------------------
# 7) Run both trainings and plot
# -------------------------
accA_sgd, accB_sgd = train_plain_sgd()
accA_kap, accB_kap = train_kaplinsky_style(lam=50.0, lr_init=0.2, backtrack=0.5, max_bt=15)

E_A = 50
T = len(accA_sgd)
x = np.arange(T)
plt.figure(figsize=(11,4))

plt.subplot(1,2,1)
plt.plot(x, accA_sgd, label="Task A (test)")
plt.plot(x, accB_sgd, label="Task B (test)")
plt.axvline(E_A-1, linestyle="--")
plt.title("Plain Sequential SGD")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim(0,1)
plt.legend()

plt.subplot(1,2,2)
plt.plot(x, accA_kap, label="Task A (test)")
plt.plot(x, accB_kap, label="Task B (test)")
plt.axvline(E_A-1, linestyle="--")
plt.title("Kaplinsky-style Potential Descent")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.ylim(0,1)
plt.legend()

plt.tight_layout()
plt.show()

print(f"Final accuracies (SGD):    A={accA_sgd[-1]:.3f}, B={accB_sgd[-1]:.3f}")
print(f"Final accuracies (Kap.):   A={accA_kap[-1]:.3f}, B={accB_kap[-1]:.3f}")
