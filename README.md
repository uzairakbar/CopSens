# CopSens Python

A Python implementation of `CopSens`: Copula-based Sensitivity Analysis for Multi-Treatment Causal Inference with Unobserved Confounding.

This package provides a `scikit-learn` compatible estimator to perform sensitivity analysis on observational data where unobserved confounders may bias causal estimates. It supports both Gaussian and Binary outcomes and allows for latent confounder modeling using PCA or PyTorch Neural Networks.

```bash
copsens_python/
├── copsens/
│   ├── __init__.py
│   ├── estimator.py
│   ├── utils.py
│   └── nn_utils.py
├── examples/
│   ├── gaussian_example.py
│   ├── binary_example.py
│   └── torch_example.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### Gaussian Outcome with Worst-Case Calibration

```python
import numpy as np
import pandas as pd
from copsens import CopSens

# Generate synthetic data
N = 1000
T = np.random.normal(0, 1, (N, 10)) # 10 Treatments
U = np.random.normal(0, 1, (N, 3))  # 3 Unobserved Confounders
# Y depends on T and U
y = T[:, 0] * 0.5 + U[:, 0] * 1.0 + np.random.normal(0, 0.5, N)

# Initialize Estimator
# Uses PCA by default to infer latent confounders
cs = CopSens(model_type='gaussian', n_components=3)

# Fit models
cs.fit(T, y)

# Define treatments for contrast (Treatment 0: 1 vs 0)
t1 = T.copy()
t1[:, 0] = 1
t2 = T.copy()
t2[:, 0] = 0

# Perform Calibration
results = cs.calibrate(t1, t2, calitype='worstcase', R2=[0.1, 0.5, 1.0])

print(results['est_df'].head())
print("Robustness Values:", results['rv'][:5])
```

### Using PyTorch for Deep Latent Variable Modeling

You can pass a custom `torch.nn.Module` or use the default built-in VAE logic.

```python
from copsens import CopSens
import torch.nn as nn

# Initialize with PyTorch backend
# 'latent_model' can be an instantiated nn.Module or just 'vae' (uses internal SimpleVAE)
cs_torch = CopSens(model_type='gaussian', latent_model='vae', n_components=3)

cs_torch.fit(T, y)

# Multivariate Calibration (minimizing L2 norm of effects)
results = cs_torch.calibrate(t1, t2, calitype='multicali', R2_constr=0.5)

print("Optimized Gamma:", results['gamma'])
print(results['est_df'].head())
```

### Binary Outcome

```python
# Binary outcome
y_bin = (y > 0).astype(int)

cs_bin = CopSens(model_type='binary', n_components=3)
cs_bin.fit(T, y_bin)

# Naive estimates (Probabilities)
naive_probs = cs_bin.predict(T)

# Calibrate with specific gamma direction
gamma_guess = np.array([1.0, -0.5, 0.0])
res_bin = cs_bin.calibrate(t1, t2, calitype='null', gamma=gamma_guess, R2=[0.2])

print(res_bin['est_df'].head())
```

### Causal Bounds
Bounds on the ATE $\mathbf{E}[ Y | \operatorname{do}(X = \mathbf{x}) ]$
```python
# ... training code ...
cs_torch.fit(T, y)

# 1. Specify query point(s) x* (can be a batch)
x_star = T[0:5] # First 5 images

# 2. Get bounds for specific sensitivity parameters R2
# Returns a dictionary keyed by R2
bounds = cs_torch.predict_bounds(x_star, R2=[0.1, 0.5, 1.0])

# 3. Access results
print("--- Bounds for R^2 = 0.5 ---")
print(bounds[0.5])
```

## Reference

[J. Zheng, A. D’Amour, & A. Franks, "Copula-based Sensitivity Analysis for Multi-Treatment Causal Inference with Unobserved Confounding," arXiv:2102.09412, 2021](https://arxiv.org/abs/2102.09412).
