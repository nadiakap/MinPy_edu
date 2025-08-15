# -*- coding: utf-8 -*-
"""
forgetting effect in MLP (trained with default algorithm - Adam) 
#when trained on financial data with changing regimes
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1. Generate synthetic bull market regime ---
np.random.seed(42)
n_days = 1000
bull_returns = np.random.normal(loc=0.001, scale=0.01, size=n_days)  # mild upward drift
bull_prices = 100 + np.cumsum(bull_returns)

# Features: rolling returns
bull_X = np.column_stack([
    np.roll(bull_returns, 1),
    np.roll(bull_returns, 2),
    np.roll(bull_returns, 3)
])[3:]  # drop NaNs
bull_y = (bull_returns[3:] > 0).astype(int)  # 1 = up, 0 = down

# --- 2. Generate synthetic bear market regime ---
bear_returns = np.random.normal(loc=-0.001, scale=0.02, size=n_days)  # downward drift
bear_prices = bull_prices[-1] + np.cumsum(bear_returns)

bear_X = np.column_stack([
    np.roll(bear_returns, 1),
    np.roll(bear_returns, 2),
    np.roll(bear_returns, 3)
])[3:]
bear_y = (bear_returns[3:] > 0).astype(int)

# --- 3. Initialize MLP model ---
clf = MLPClassifier(hidden_layer_sizes=(200,), max_iter=600, random_state=42)

# --- 4. Train on bull market ---
clf.fit(bull_X, bull_y)
acc_bull_before = accuracy_score(bull_y, clf.predict(bull_X))
acc_bear_before = accuracy_score(bear_y, clf.predict(bear_X))

print("After Bull Market training:")
print(f"  Bull accuracy: {acc_bull_before:.3f}")
print(f"  Bear accuracy: {acc_bear_before:.3f}")

# --- 5. Train on bear market (forgetting bull data) ---
clf.fit(bear_X, bear_y)
acc_bull_after = accuracy_score(bull_y, clf.predict(bull_X))
acc_bear_after = accuracy_score(bear_y, clf.predict(bear_X))

print("\nAfter Bear Market training:")
print(f"  Bull accuracy: {acc_bull_after:.3f}  <-- drop shows forgetting")
print(f"  Bear accuracy: {acc_bear_after:.3f}")

# --- 6. Plot forgetting effect ---
plt.figure(figsize=(6,4))
plt.bar(["Bull Before", "Bull After"], [acc_bull_before, acc_bull_after], color=['blue', 'red'])
plt.ylabel("Accuracy")
plt.title("Catastrophic Forgetting: Bull Market Task")
plt.ylim(0, 1)
plt.show()

