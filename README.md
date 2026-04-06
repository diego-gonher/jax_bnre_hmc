# BNRE + HMC
## Balanced Neural Ratio Estimation with Hamiltonian Monte Carlo implemented with JAX

This repository implements **Balanced Neural Ratio Estimation (BNRE)** in **JAX/Flax**, and uses the learned ratio estimator inside **Hamiltonian Monte Carlo (HMC)** (via NumPyro) for fast and statistically robust Bayesian parameter inference.

At a high level:

- **NRE/BNRE**: Train a classifier  $d_\phi(\theta, x)$ that distinguishes joint samples $(\theta, x)$ from marginally paired $(\theta, x')$. From its logits you obtain an estimate of the likelihood(-ratio).
- **BNRE**: Adds a **balance penalty** encouraging the classifier’s average output on joint vs marginal samples to be equal, which stabilizes training and improves downstream inference.
- **HMC**: Uses the learned ratio estimator as a surrogate likelihood to run MCMC over parameters $\theta$, giving calibrated posterior samples.

The code is written to be:

- **Fully JAX-compatible** (JIT, pure functions, PyTrees)
- **Config-driven** via **Hydra**
- **Checkpointed** via **Orbax**, with support for best-model restoration

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [HDF5 training dataset contract](#hdf5-training-dataset-contract)
- [Example: Sinusoid (end-to-end workflow)](#example-sinusoid-end-to-end-workflow)
- [Using the Trained Ratio Estimator in HMC](#using-the-trained-ratio-estimator-in-hmc)
- [Run summary JSON files](#run-summary-json-files)
- [Transformer ratio estimator for missing 1D data (masked observations)](#transformer-ratio-estimator-for-missing-1d-data-masked-observations)
- [Using Codex CLI for the agentic workflow](#using-codex-cli-for-the-agentic-workflow)

---

## Repository Structure

### Core library (`src/jax_bnre_hmc`)

- **`train.py`**  
  - `TrainConfig`: Configuration for training (learning rate, epochs, batch size, BNRE weight, early stopping, checkpointing, etc.).  
  - `create_train_state`: Builds the Flax model and Optax optimizer.  
  - `train_step` / `validation_step`: JIT-ed functions that compute NRE + optional BNRE penalty, update parameters (train), and report metrics.  
  - `train`: High-level training loop with:
    - mini-batching and shuffling (static shapes, remainder dropped)
    - BNRE loss integration (NRE + λ·penalty)
    - checkpointing (latest and best)
    - **early stopping** based on validation loss.

- **`loss.py`**  
  - `nre_loss_from_logits`: Label-free NRE objective.  
  - `nre_loss_bce_style_from_logits`: BCE-with-logits form of NRE for SBI-style comparison.  
  - `bnre_balance_from_logits`: Computes BNRE balance term $B$ and penalty $(B-1)^2$.

- **`data.py`**  
  - `Batch`: Simple container for batched $(\theta, x)$.  
  - `_derangement`: Samples a permutation with no fixed points (to avoid mislabeled negatives).  
  - `make_joint_and_marginal`: Builds joint and marginal batches from paired samples.  
  - `make_batches`: Deterministic shuffling + mini-batch iterator (drops remainder).

- **`model.py`**  
  - `RatioEstimatorMLP`: MLP mapping $(\theta, x)$ to a scalar logit.  
  - `RatioEstimatorResNet`: ResNet-style MLP over concatenated $(\theta, x)$.  
  - `ResidualBlock`, `get_activation`: Supporting components.

- **`checkpointing.py`**  
  - Uses **`orbax.checkpoint.PyTreeCheckpointer`**.  
  - `get_run_dir`: Hydra run directory helper.  
  - `resolve_run_train_config_path`: Resolves `run_dir/train.yaml`, or legacy `run_dir/config.yaml`.  
  - `ensure_dirs`: Creates `checkpoints/latest/` and `checkpoints/best/`.  
  - `save_latest`: Saves the full `TrainState` (latest).  
  - `save_best`: Saves only the best model parameters (params PyTree).  
  - `load_best_params`: Restores best parameters for inference.  
  - `write_meta`: Writes small JSON metadata (`epoch`, `val_loss`) for latest/best.

- **`hmc.py`, `diagnostics.py`**  
  - Utilities for running HMC (NumPyro-based) using the learned ratio estimator, and for inspecting / diagnosing chains.  
  - **`diagnostics.py`** also provides **TARP** coverage curves (`run_tarp_jax`), **simulation-based calibration (SBC)** from posterior samples (`run_sbc_from_samples`, `check_sbc` with marginal rank **KS p-values** per parameter), and optional **SBC rank histogram** plotting (`plot_sbc_rank_histograms`). Canonical HMC scripts run TARP and SBC and record summary metrics in `hmc_summary.json` (see below).  
  - TARP and SBC capture complementary aspects of calibration:  
    - TARP evaluates global coverage behavior of the posterior.  
    - SBC evaluates marginal calibration for each parameter.  
  - It is possible for TARP to appear well-calibrated while SBC reveals biases in individual parameters.

### Configs (`configs/`)

Hydra configs for model architecture, training, and (for HMC) inference settings. The **canonical templates** are:

- **`configs/sinusoid/train.yaml`** / **`configs/sinusoid/hmc.yaml`** – MLP ratio estimator on pre-split HDF5 data.  
- **`configs/sinusoid_transformer/train.yaml`** / **`configs/sinusoid_transformer/hmc.yaml`** – transformer ratio estimator with optional observation masks (see below).

Additional experiments (e.g. amber501) have their own folders under `configs/`.

Typical training keys in the canonical configs:

- **`data`**: `dataset_file` path to an HDF5 file following the contract below.  
- **`model`**: Architecture (MLP hidden dims / activation / norm, or transformer hyperparameters).  
- **`train`**: Learning rate, epochs, BNRE weight (`bnre_gamma`), batch size, gradient clipping, checkpointing, and early stopping (`stop_after_epochs`).

### Experiments (`experiments/`)

**Canonical templates** (recommended starting points):

- **`experiments/sinusoid/train.py`, `experiments/sinusoid/hmc.py`**  
  - Load pre-split HDF5 data, train an MLP ratio estimator, run NUTS/HMC on held-out observations.

- **`experiments/sinusoid_transformer/train.py`, `experiments/sinusoid_transformer/hmc.py`**  
  - Same workflow with masked 1D observations and a transformer ratio estimator.

Other directories under `experiments/` (e.g. amber501) follow the same Hydra + `train` / `hmc` script pattern for domain-specific data.

### Outputs (`outputs/`)

Hydra writes each run under:

```text
outputs/<exp_name>/<YYYY-MM-DD>_<HH-MM-SS>/
```

Inside you’ll find:

- **`train.yaml`** – full resolved Hydra training config for the run (older runs may use `config.yaml`; HMC loads `train.yaml` when present).  
- **`metrics.txt`** – final losses and summary statistics.  
- **`losses.png`, `bce_style_losses.png`, `sigmoid.png`** – training curves and diagnostic plots.  
- **`checkpoints/`** –  
  - `latest/` – full `TrainState` (latest epoch).  
  - `best/` – best params only.  
  - `latest_meta.json`, `best_meta.json` – small JSON metadata with `epoch` and `val_loss`.
- **`train_summary.json`** – machine-readable training outcome (see [Run summary JSON files](#run-summary-json-files)).

---

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e .
```

Make sure you have a compatible JAX / JAXLIB / NumPyro stack; see `environment.yml` for one working environment.

---

## HDF5 training dataset contract

File-based experiments load pre-split data with `jax_bnre_hmc.datasets.load_hdf5_dataset` (and optional validation via `validate_hdf5_dataset`). The expected **HDF5 layout** is:

**Required datasets (2D numeric arrays, finite values):**

| Dataset       | Meaning |
|---------------|---------|
| `theta_train` | Parameters for training rows |
| `x_train`     | Observations for training rows |
| `theta_val`   | Parameters for validation rows |
| `x_val`       | Observations for validation rows |
| `theta_test`  | Parameters for test rows |
| `x_test`      | Observations for test rows |

**Shape rules:**

- Each split: `theta_*` and `x_*` must have the **same number of rows** (paired $(\theta, x)$ examples).
- Across splits: all `theta_*` must share the same **second dimension** (parameter dim); all `x_*` must share the same **second dimension** (feature dim).

**Optional masks (for $x$ only, e.g. missing grid points):**

| Dataset       | Meaning |
|---------------|---------|
| `x_train_mask` | Mask for `x_train` |
| `x_val_mask`   | Mask for `x_val` |
| `x_test_mask`  | Mask for `x_test` |

If **any** mask dataset is present, **all three** must be present. Each mask must be **2D**, numeric, finite, and **exactly the same shape** as the corresponding `x_*` array. Typical convention: `1` = valid / observed, `0` = missing (downstream scripts may treat values $> 0.5$ as valid).

**Optional metadata (file-level HDF5 attributes):**

- `theta_names`, `x_names` (if present, used for plotting labels where supported)
- `description` and any other attributes are collected into `LoadedDataset.metadata`

No scaling, shuffling, or reshaping is applied in the loader (see `load_hdf5_dataset` docstring); fit scaling on train only and apply downstream, together with any experiment-specific preprocessing in `train.py` / `hmc.py`.

---

## Example: Sinusoid (end-to-end workflow)

The simplest way to get started is the **`sinusoid`** experiment (MLP on HDF5 data), following the full pipeline from training to HMC and saved outputs. For **masked observations** and a transformer, use **`sinusoid_transformer`** the same way with `configs/sinusoid_transformer/train.yaml` and `python experiments/sinusoid_transformer/train.py`.

### 1. Inspect / tweak the config

Open `configs/sinusoid/train.yaml` (set `data.dataset_file` to your HDF5 path if needed):

```yaml
exp_name: sinusoid

hydra:
  job:
    chdir: false
  run:
    dir: outputs/${exp_name}/${now:%Y-%m-%d}_${now:%H-%M-%S}

seed: 0

data:
  dataset_file: datasets/sinusoid/sinusoid_noisy_no_masks.h5

model:
  hidden_dims: [50, 50, 50]
  activation: tanh
  norm: layernorm

train:
  lr: 0.0005
  epochs: 5000
  bnre_gamma: 100.0
  batch_size: 128
  print_every: 10
  clip_max_norm: 5.00
  save_every: 500
  checkpoint_dirname: "checkpoints"
  stop_after_epochs: 250
```

Key knobs:

- **`train.bnre_gamma`**: Set to `0.0` for standard NRE, or positive for BNRE.  
- **`stop_after_epochs`**: Early stopping patience based on validation loss.  
- **`save_every`**, **`checkpoint_dirname`**: Checkpointing frequency and location.  
- **`data.dataset_file`**: HDF5 path (see the dataset contract below).

### 2. Run training

From the project root:

```bash
python experiments/sinusoid/train.py
```

This will:

- Load pre-split train/val/test data from the HDF5 file.  
- Train an NRE/BNRE classifier with mini-batching, BNRE penalty, gradient clipping, checkpointing, and early stopping.  
- Save `train.yaml`, `metrics.txt`, `train_summary.json`, plots, and checkpoints under `outputs/sinusoid/...`.  
- Load the **best** checkpoint at the end and verify recomputed validation loss against metadata (up to shuffling stochasticity).

You can override any config value from the command line via Hydra, e.g.:

```bash
python experiments/sinusoid/train.py train.bnre_gamma=0.0 train.stop_after_epochs=null
```

---

## Using the Trained Ratio Estimator in HMC

Inference is driven by **Hydra configs** in `configs/*/hmc.yaml`. Each `experiments/*/hmc.py` script loads its corresponding `hmc.yaml`, then:

- Reads `run_dir/train.yaml` (the saved training config; falls back to `config.yaml` for legacy runs) to **rebuild the exact ratio-estimator architecture**.
- Loads best parameters from `run_dir/<checkpoint_dirname>/best/` via Orbax.
- Uses HMC/NUTS (NumPyro) to sample \(\theta\) conditioned on selected observations.

### Quickstart: running inference with `hmc.yaml`

1. Pick a completed training run directory (e.g. `outputs/sinusoid/2026-03-15_17-48-50`). It must contain:
   - `train.yaml` (or legacy `config.yaml`)
   - `checkpoints/best/` (and/or `checkpoints/latest/`)

2. Edit (or override) the experiment’s `configs/<exp>/hmc.yaml`. For example, `configs/sinusoid/hmc.yaml` contains:

```yaml
run_dir: outputs/sinusoid/2026-03-15_17-48-50
output_dir: null

data:
  dataset_file: null  # set via CLI or override

num_chains: 4
num_warmup: 4000
num_samples: 4000

n_observations: 500
n_plots: 25
seed: 2401

prior:
  type: box
  low: [-1.0, -1.0, -1.0, -1.0]
  high: [1.0, 1.0, 1.0, 1.0]
```

3. Run the HMC script, overriding any values you want. Examples:

```bash
# Sinusoid (MLP): point to run_dir and dataset_file
python experiments/sinusoid/hmc.py \
  run_dir=outputs/sinusoid/2026-03-15_17-48-50 \
  data.dataset_file=/absolute/path/to/sinusoid_noisy_no_masks.h5

# Sinusoid transformer: same pattern with the masked HDF5 and transformer run_dir
python experiments/sinusoid_transformer/hmc.py \
  run_dir=outputs/sinusoid_transformer/<run_timestamp> \
  data.dataset_file=/absolute/path/to/sinusoid_noisy_with_masks.h5
```

By default, if `output_dir: null`, results are written to `run_dir/hmc_results/`. That folder includes **`hmc_summary.json`**, **`hmc_metrics.txt`**, TARP/SBC plots, and **`posterior_samples.h5`** (see [Run summary JSON files](#run-summary-json-files)).

Posterior samples in **`posterior_samples.h5`** are stored with shape `(S, N, D)`:

- **`S`**: number of samples (chains × draws).  
- **`N`**: number of observations.  
- **`D`**: parameter dimension.

### What `hmc.yaml` controls

- **`run_dir`**: Which trained estimator to load (architecture from `run_dir/train.yaml`, or `run_dir/config.yaml` on older runs; params from `run_dir/<checkpoint_dirname>/best/`).  
- **`data.dataset_file`**: Where to load inference observations (HDF5) for file-based experiments.  
- **`num_chains`, `num_warmup`, `num_samples`**: NUTS/HMC settings.  
- **`n_observations`, `seed`**: How many observations to run and how they are selected.  
- **`n_plots`, `corner_labels`**: Plotting configuration.  
- **`prior.*`**: Prior bounds/shape used by HMC (box prior in most experiments; convex-hull prior for amber501 examples).

### Typical pattern (under the hood)

The typical inference pattern is:

- Load the **best** parameters with `load_best_params`.  
- Reconstruct the same model architecture as during training.  
- Define a surrogate log-likelihood (or likelihood-ratio) in terms of the model’s logits.  
- Run HMC / NUTS in NumPyro to obtain posterior samples over $\theta$.

For concrete code, see **`experiments/sinusoid/hmc.py`** and **`experiments/sinusoid_transformer/hmc.py`** (canonical templates for moving from BNRE training to HMC with the learned ratio estimator).

---

## Run summary JSON files

Small JSON files record outcomes for training runs and for HMC inference folders. They are written by `write_train_summary` (`src/jax_bnre_hmc/train.py`) and `write_hmc_summary` (`src/jax_bnre_hmc/hmc.py`).

### `train_summary.json` (training run directory)

Written at the **Hydra run root** (e.g. `outputs/<exp_name>/<timestamp>/`) when training finishes successfully, or with `status: "error"` if the script catches a failure.

**Success (`status: "ok"`):**

| Field | Meaning |
|-------|---------|
| `status` | Always `"ok"` for a completed run that wrote this schema. |
| `best_val_loss` | Validation loss at the best checkpoint (matches `checkpoints/best/best_meta.json`). |
| `best_epoch` | 1-based epoch index when that best checkpoint was saved. |
| `dims.theta_dim` | Parameter dimension $D$ (columns of `theta_train`). |
| `dims.x_dim` | Observation “width”: for flat $x$ of shape `(N, D)`, this is `D`; for token inputs `(N, T, F)` (e.g. transformer), this is the sequence length `T`. |
| `model_type` | Coarse architecture label from config (e.g. `mlp`, `transformer`). |

**Error (`status: "error"`):**

| Field | Meaning |
|-------|---------|
| `status` | `"error"`. |
| `message` | Optional error description (omitted if no message was passed). |

### `hmc_summary.json` (HMC output directory)

Written under **`output_dir`** (default `run_dir/hmc_results/`) by the canonical HMC scripts after posterior samples, TARP, and SBC. On failure, only an error payload is written.

**Success (`status: "ok"`):**

| Field | Meaning |
|-------|---------|
| `status` | `"ok"`. |
| `divergences_total` | Total count of diverging NUTS transitions (warmup + sampling, all chains, all observations). |
| `divergences_per_observation` | `divergences_total / n_observations`. |
| `divergences_per_observation_per_chain` | `divergences_total / (n_observations × num_chains)`. |
| `sbc_ks_pval_min` | Minimum Kolmogorov–Smirnov p-value across **marginal SBC rank** columns (uniformity of ranks under calibration; see `check_sbc` in `diagnostics.py`). |
| `sbc_ks_pval_mean` | Mean of those marginal KS p-values. |
| `tarp_mae` | Mean absolute error between the TARP empirical coverage curve and the diagonal: `mean(|ecp − α|)` over the TARP grid. |
| `tarp_iae` | Integrated absolute error: `trapz(|ecp − α|, α)` over the same grid. |
| `posterior_samples_path` | Relative path to the HDF5 file with posterior samples (typically `posterior_samples.h5`). |

**Interpretation:**

- Lower `tarp_mae` / `tarp_iae` → better global calibration (ideal = 0).  
- Higher `sbc_ks_pval_min` / `sbc_ks_pval_mean` → better marginal calibration (values near 1 indicate consistency with uniform ranks).  
- Low `sbc_ks_pval_min` (e.g. < 0.05) indicates at least one parameter is miscalibrated.  
- Nonzero `divergences_*` indicate potential HMC geometry issues; lower is better.

SBC is performed in the same parameter space used by HMC (i.e., after any scaling or transformations applied in the experiment scripts).

Per-dimension **SBC KS p-values** and raw **TARP** curves are also printed and saved in **`hmc_metrics.txt`** and plots (`sbc_rank_histograms.png`, `tarp_ecp_curve.png`) in the same folder.

**Error (`status: "error"`):**

| Field | Meaning |
|-------|---------|
| `status` | `"error"`. |
| `message` | Error description. |

---

## Transformer ratio estimator for missing 1D data (masked observations)

The second canonical template, **`sinusoid_transformer`**, uses a ratio estimator that handles **missing values** in 1D observations under the assumption that every observation lives on the **same grid** (e.g. time bins), but some entries can be invalid/missing.

Representation:

- Observation values: $y = (y_1, \dots, y_T)$
- Mask: $m = (m_1, \dots, m_T)$, where $m_i = 1$ if $y_i$ is valid and $0$ otherwise

The model consumes a token sequence of shape `(T, 2)` with tokens $[y_i, m_i]$. In the training script, masked entries are:

- filled only for scaling (using per-timepoint mean over valid entries),
- then **zeroed** in the actual model input, while the mask channel indicates validity.

See:

- `experiments/sinusoid_transformer/train.py` for preprocessing and tokenization into `x_tokens`.
- `configs/sinusoid_transformer/hmc.yaml` and `experiments/sinusoid_transformer/hmc.py` for inference using the same representation.

---

## Using Codex CLI for the agentic workflow

You can use Codex as a lightweight lab-technician layer for this repository. The expected workflow and constraints are documented in `AGENTS.md`, so start Codex from a shell where your intended Python environment is already activated, then give it a single natural-language prompt with:

- dataset path
- experiment name
- whether masks are present
- preprocessing instructions

Codex should then follow the repository workflow: inspect the dataset, choose the correct canonical template, create/adapt experiment files, run training, run HMC, and generate the report.

Reusable prompt template:

```text
Follow AGENTS.md strictly. Run experiment using the dataset located in 'datasets/<dataset_folder>/<dataset_file>.h5', experiment name '<experiment_name>', there is no missing data, use MinMax scaling to [-1, 1] for both theta and x (fit on train only).
```

Concrete example:

```text
Follow AGENTS.md strictly. Run experiment using the dataset located in 'datasets/lotka_volterra/lotka_volterra_no_masks.h5', experiment name 'lotka_volterra', there is no missing data, use MinMax scaling to [-1, 1] for both theta and x (fit on train only).
```

Expected outputs include a Hydra run directory under `outputs/<experiment_name>/...`, `train_summary.json`, `hmc_results/hmc_summary.json`, and a generated markdown report at `hmc_results/report.md`.

You can compare reports across runs to inspect how configuration changes (for example changing `train.bnre_gamma`) affect recorded metrics and diagnostics.
