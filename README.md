# BNRE + HMC
## Balanced Neural Ratio Estimation with Hamiltonian Monte Carlo implemented with JAX

This repository implements **Balanced Neural Ratio Estimation (BNRE)** in **JAX/Flax**, and uses the learned ratio estimator inside **Hamiltonian Monte Carlo (HMC)** (via NumPyro) for fast and statistically robust Bayesian parameter inference.

At a high level:

- **NRE/BNRE**: Train a classifier \( d_\phi(\theta, x) \) that distinguishes joint samples \((\theta, x)\) from marginally paired \((\theta, x')\). From its logits you obtain an estimate of the likelihood(-ratio).
- **BNRE**: Adds a **balance penalty** encouraging the classifier’s average output on joint vs marginal samples to be equal, which stabilizes training and improves downstream inference.
- **HMC**: Uses the learned ratio estimator as a surrogate likelihood to run MCMC over parameters \(\theta\), giving calibrated posterior samples.

The code is written to be:

- **Fully JAX-compatible** (JIT, pure functions, PyTrees)
- **Config-driven** via **Hydra**
- **Checkpointed** via **Orbax**, with support for best-model restoration

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
  - `bnre_balance_from_logits`: Computes BNRE balance term \(B\) and penalty \((B-1)^2\).

- **`data.py`**  
  - `Batch`: Simple container for batched \((\theta, x)\).  
  - `_derangement`: Samples a permutation with no fixed points (to avoid mislabeled negatives).  
  - `make_joint_and_marginal`: Builds joint and marginal batches from paired samples.  
  - `make_batches`: Deterministic shuffling + mini-batch iterator (drops remainder).

- **`model.py`**  
  - `RatioEstimatorMLP`: MLP mapping \((\theta, x)\) to a scalar logit.  
  - `RatioEstimatorResNet`: ResNet-style MLP over concatenated \((\theta, x)\).  
  - `ResidualBlock`, `get_activation`: Supporting components.

- **`checkpointing.py`**  
  - Uses **`orbax.checkpoint.PyTreeCheckpointer`**.  
  - `get_run_dir`: Hydra run directory helper.  
  - `ensure_dirs`: Creates `checkpoints/latest/` and `checkpoints/best/`.  
  - `save_latest`: Saves the full `TrainState` (latest).  
  - `save_best`: Saves only the best model parameters (params PyTree).  
  - `load_best_params`: Restores best parameters for inference.  
  - `write_meta`: Writes small JSON metadata (`epoch`, `val_loss`) for latest/best.

- **`hmc.py`, `diagnostics.py`**  
  - Utilities for running HMC (NumPyro-based) using the learned ratio estimator, and for inspecting / diagnosing chains. (See the experiment scripts below for concrete usage.)

### Configs (`configs/`)

Hydra configs describing simulators, priors, model architecture, and training settings:

- **`configs/linear_toy/train.yaml`** – linear regression toy example.  
- **`configs/sinusoid/train.yaml`** – sinusoid regression example.  
- **`configs/2param/train.yaml`** – two-parameter experiment.

Each config defines:

- **`prior`**: Parameter ranges.  
- **`data`**: Number of simulations, noise level, train/validation split.  
- **`model`**: Hidden dimensions, activation, normalization.  
- **`train`**: Learning rate, epochs, BNRE λ, batch size, gradient clipping, checkpointing settings, and early stopping (`stop_after_epochs`).

### Experiments (`experiments/`)

- **`experiments/linear_toy/train.py`**  
  - Uses the linear simulator to generate \((\theta, x)\) pairs.  
  - Splits into train/validation.  
  - Builds `TrainConfig` from `configs/linear_toy/train.yaml`.  
  - Calls `jax_bnre_hmc.train.train` to train NRE/BNRE.  
  - Saves config, metrics, and plots to the Hydra run directory.  
  - Loads the **best** saved parameters and verifies the validation loss.

- **`experiments/linear_toy/hmc.py`**  
  - Example of running HMC using the trained ratio estimator for the linear toy.

- **`experiments/sinusoid/train.py`, `experiments/sinusoid/hmc.py`**  
  - Similar structure for a sinusoidal simulator and its HMC-based posterior inference.

- **`experiments/2param/train.py`, `experiments/2param/hmc.py`**  
  - Example for a two-parameter simulator.

### Outputs (`outputs/`)

Hydra writes each run under:

```text
outputs/<exp_name>/<YYYY-MM-DD>_<HH-MM-SS>/
```

Inside you’ll find:

- **`config.yaml`** – full resolved Hydra config for the run.  
- **`metrics.txt`** – final losses and summary statistics.  
- **`losses.png`, `bce_style_losses.png`, `sigmoid.png`** – training curves and diagnostic plots.  
- **`checkpoints/`** –  
  - `latest/` – full `TrainState` (latest epoch).  
  - `best/` – best params only.  
  - `latest_meta.json`, `best_meta.json` – small JSON metadata with `epoch` and `val_loss`.

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

## Example: Linear Toy with BNRE + Early Stopping

The simplest way to get started is to run the **linear toy** experiment.

### 1. Inspect / tweak the config

Open `configs/linear_toy/train.yaml`:

```yaml
exp_name: linear_toy

hydra:
  job:
    chdir: false
  run:
    dir: outputs/${exp_name}/${now:%Y-%m-%d}_${now:%H-%M-%S}

seed: 0

prior:
  m_low: 0.0
  m_high: 1.0
  b_low: 0.0
  b_high: 1.0

data:
  n_simulations: 1024
  n_points: 10
  sigma: 0.1
  validation_fraction: 0.2

model:
  hidden_dims: [50, 50, 50]
  activation: tanh
  norm: layernorm

train:
  lr: 0.0005
  epochs: 5000
  bnre_lambda: 10000.0
  batch_size: 128
  print_every: 100
  clip_max_norm: 5.00 # null for no clipping
  save_every: 200      # save latest checkpoint every N epochs (0 or null to disable)
  checkpoint_dirname: "checkpoints"
  stop_after_epochs: 100  # early stopping patience (null to disable)
```

Key knobs:

- **`bnre_lambda`**: Set to `0.0` for standard NRE, or to a positive value to enable BNRE.  
- **`stop_after_epochs`**: Early stopping patience based on validation loss.  
- **`save_every`**, **`checkpoint_dirname`**: Control checkpointing frequency and location.

### 2. Run training

From the project root:

```bash
python experiments/linear_toy/train.py
```

This will:

- Simulate data from the linear model.  
- Train an NRE/BNRE classifier with mini-batching, BNRE penalty, gradient clipping, checkpointing, and early stopping.  
- Save metrics and plots under `outputs/linear_toy/...`.  
- Save **latest** and **best** checkpoints under `outputs/linear_toy/.../checkpoints/`.  
- Load the **best** checkpoint at the end and verify that recomputed validation loss matches the saved best value (up to stochasticity from shuffling).

You can override any config value from the command line via Hydra, e.g.:

```bash
python experiments/linear_toy/train.py train.bnre_lambda=0.0 train.stop_after_epochs=null
```

---

## Using the Trained Ratio Estimator in HMC

The `experiments/*/hmc.py` scripts show how to plug the trained ratio estimator into a NumPyro-based HMC sampler. The typical pattern is:

- Load the **best** parameters with `load_best_params`.  
- Reconstruct the same model architecture as during training.  
- Define a surrogate log-likelihood (or likelihood-ratio) in terms of the model’s logits.  
- Run HMC / NUTS in NumPyro to obtain posterior samples over \(\theta\).

For concrete code, see:

- `experiments/linear_toy/hmc.py`  
- `experiments/sinusoid/hmc.py`  
- `experiments/2param/hmc.py`

These examples illustrate how to move from simulation-based training (BNRE) to full Bayesian inference (HMC) with the learned ratio estimator.
