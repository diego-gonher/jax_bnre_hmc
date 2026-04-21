# Noisy Sinusoidal Time Series Benchmark

This benchmark provides a simple, fully controlled **1D simulation-based inference (SBI)** task based on noisy sinusoidal time series. It is designed to test inference pipelines under:

- nonlinear forward models,
- heteroscedastic noise,
- and (optionally) missing data via contiguous masks.

It serves as a minimal benchmark before moving to more complex scientific datasets.

---

## Generative Model

We model a scalar time series as:

$$
y(t) = A \sin(2\pi f t + \phi) + b
$$

where the parameters are:

$$
\theta = (A, f, \phi, b)
$$

- $A$: amplitude  
- $f$: frequency  
- $\phi$: phase  
- $b$: offset  

The time grid is uniform:

$$
t_i \in [0, 1], \quad i = 1, \dots, N_{\text{time}}
$$

---

## Observation Model (Noise)

Observations are corrupted by **heteroscedastic Gaussian noise**:

$$
y_i^{\text{obs}} = y(t_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma_i^2)
$$

with per-timepoint noise:

$$
\sigma_i \sim \text{LogNormal}(\mu=-2.0,\ \tau=0.3)
$$

This results in moderate variability in noise levels across the sequence.

---

## Priors

The parameters are sampled independently from uniform priors:

- $A \sim \mathcal{U}(0.5, 2.0)$
- $f \sim \mathcal{U}(0.25, 10.0)$
- $\phi \sim \mathcal{U}(0, 2\pi)$
- $b \sim \mathcal{U}(-0.5, 0.5)$

---

## Missing Data (Optional)

When masking is enabled, missing data is introduced via **random contiguous blocks**:

- Number of blocks: uniformly sampled from $\{1, 2, 3\}$
- Total masked fraction: uniformly sampled from $[0.1, 0.25]$
- Blocks are placed randomly in time
- Overlaps are allowed

A binary mask is defined as:

$$
m_i =
\begin{cases}
1 & \text{valid observation} \\
0 & \text{missing}
\end{cases}
$$

---

## Data Representation

Each observation is represented as:

$$
x_i = [y_i^{\text{obs}},\ m_i]
$$

so that the full input has shape:

$$
x \in \mathbb{R}^{N_{\text{time}} \times 2}
$$

Datasets follow the standard pipeline contract:

- `theta_*`: shape `(N, 4)`
- `x_*`: shape `(N, N_time, 2)`
- `mask_*`: shape `(N, N_time)`

---

## Dataset Generation

The dataset is generated using:

```bash
generate_dataset.py
```

### Create dataset **with masks**

```bash
python generate_dataset.py \
  --out datasets/sinusoid/sinusoid_noisy_with_masks.h5 \
  --use_mask
```

### Create dataset **without masks**

```bash
python generate_dataset.py \
  --out datasets/sinusoid/sinusoid_noisy_no_masks.h5
```

#### Available Arguments
- `--out` (str, required): Output path to HDF5 file
- `--n_time` (int, default=50) Number of time steps per sample
- `--n_train` (int, default=10000) Number of training samples
- `--n_val` (int, default=2000) Number of validation samples
- `--n_test` (int, default=2000) Number of test samples
- `--seed` (int, default=117) Random seed for reproducibility
- `--use_mask` (flag) Enables contiguous masking

---

### Purpose

This benchmark is designed to:

- validate inference pipelines on a controlled nonlinear problem,
- test robustness to heteroscedastic noise,
- evaluate handling of missing contiguous data,
- and provide a lightweight baseline for SBI methods such as BNRE + HMC.

---

### Notes
- The dataset is generated on the fly and is not version-controlled.
- Regenerating the dataset with the same seed yields identical results.
- The masked and unmasked versions are identical except for the presence of missing data.