# -*- coding: utf-8 -*-
"""
comparing Stochastic Gradient Descent and its hybrid with Kapl style algorithm
how these algorithms overcome catastrophic forgetting in data with changing regimes
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# ----- 1. Create synthetic datasets -----
def create_task(seed):
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                               n_classes=2, random_state=seed)
    X = StandardScaler().fit_transform(X)
    return X, y

X_a, y_a = create_task(seed=0)
X_b, y_b = create_task(seed=42)

Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_a, y_a, test_size=0.2, random_state=0)
Xb_train, Xb_test, yb_train, yb_test = train_test_split(X_b, y_b, test_size=0.2, random_state=0)

Xa_train, ya_train = torch.tensor(Xa_train, dtype=torch.float32), torch.tensor(ya_train, dtype=torch.long)
Xa_test, ya_test = torch.tensor(Xa_test, dtype=torch.float32), torch.tensor(ya_test, dtype=torch.long)
Xb_train, yb_train = torch.tensor(Xb_train, dtype=torch.float32), torch.tensor(yb_train, dtype=torch.long)
Xb_test, yb_test = torch.tensor(Xb_test, dtype=torch.float32), torch.tensor(yb_test, dtype=torch.long)

# ----- 2. Simple feed-forward network -----
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# ----- 3. Training function -----
def train_model(model, optimizer, data, labels, old_params=None, lam=0.0):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        # Kaplinsky-style stability term: penalize deviation from old params
        if old_params is not None:
            stability_loss = 0
            for p, old_p in zip(model.parameters(), old_params):
                stability_loss += torch.sum((p - old_p) ** 2)
            loss += lam * stability_loss
        loss.backward()
        optimizer.step()

def test_model(model, data, labels):
    with torch.no_grad():
        outputs = model(data)
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
    return acc

# ----- 4. SGD Experiment -----
sgd_model = SimpleNet()
opt_sgd = optim.SGD(sgd_model.parameters(), lr=0.01)

# Train on Task A
train_model(sgd_model, opt_sgd, Xa_train, ya_train)
acc_a_before = test_model(sgd_model, Xa_test, ya_test)
# Train on Task B (no regularization)
train_model(sgd_model, opt_sgd, Xb_train, yb_train)
acc_a_after = test_model(sgd_model, Xa_test, ya_test)
acc_b_final = test_model(sgd_model, Xb_test, yb_test)

# ----- 5. Kaplinsky-Style Experiment -----
kap_model = SimpleNet()
opt_kap = optim.SGD(kap_model.parameters(), lr=0.01)

# Train on Task A
train_model(kap_model, opt_kap, Xa_train, ya_train)
acc_a_before_kap = test_model(kap_model, Xa_test, ya_test)
# Save parameters
old_params = [p.clone().detach() for p in kap_model.parameters()]
# Train on Task B with stability term
train_model(kap_model, opt_kap, Xb_train, yb_train, old_params, lam=0.01)
acc_a_after_kap = test_model(kap_model, Xa_test, ya_test)
acc_b_final_kap = test_model(kap_model, Xb_test, yb_test)

# ----- 6. Plot results -----
labels = ["Task A Before B", "Task A After B", "Task B Final"]
sgd_scores = [acc_a_before, acc_a_after, acc_b_final]
kap_scores = [acc_a_before_kap, acc_a_after_kap, acc_b_final_kap]

x = np.arange(len(labels))
width = 0.35
plt.bar(x - width/2, sgd_scores, width, label="SGD")
plt.bar(x + width/2, kap_scores, width, label="Kaplinsky")
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Catastrophic Forgetting: SGD vs Kaplinsky")
plt.xticks(x, labels)
plt.legend()
plt.savefig("forgetting_comparison.png")
plt.show()
