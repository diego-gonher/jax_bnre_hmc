# Lotka–Volterra Benchmark

This benchmark provides a classic **nonlinear dynamical system** for simulation-based inference (SBI), based on the Lotka–Volterra predator–prey equations. It is designed to test inference pipelines on:

- nonlinear coupled ODE systems,
- structured temporal data,
- and multiplicative observation noise.

This benchmark is more realistic than simple toy models (e.g. sinusoids) while remaining computationally lightweight.

---

## Generative Model

The system evolves according to the Lotka–Volterra equations:

$$
\frac{dx}{dt} = \alpha x - \beta x y
$$

$$
\frac{dy}{dt} = -\gamma y + \delta x y
$$

where:

- $x(t)$: prey population  
- $y(t)$: predator population  

The parameters are:

$$
\theta = (\alpha, \beta, \gamma, \delta)
$$

- $\alpha$: prey growth rate  
- $\beta$: predation rate  
- $\gamma$: predator death rate  
- $\delta$: predator reproduction rate  

---

## Simulation Details

- Initial condition:
  $$
  (x_0, y_0) = (30.0, 1.0)
  $$

- Time span:
  $$
  t \in [0, \text{days}]
  $$

- Time discretization:
  $$
  \Delta t = \text{saveat}
  $$

The system is numerically integrated using a standard ODE solver (`RK45`).

---

## Observation Model

Rather than using the full trajectory, we construct a **summary representation**:

1. Subsample the trajectory every `subsample_stride` time steps  
2. Flatten both species into a single vector  
3. Add **multiplicative log-normal noise**

Formally:

$$
x_{\text{sub}} = \text{subsample}(x(t), y(t))
$$

$$
x_i^{\text{obs}} \sim \text{LogNormal}(\log(x_i^{\text{sub}}),\ \sigma)
$$

where:
- $\sigma =$  obs\_noise\_log\_scale

This produces a positive-valued noisy summary vector.

---

## Priors

Parameters are sampled independently from uniform distributions:

- $\alpha \sim \mathcal{U}(0.5, 1.5)$
- $\beta \sim \mathcal{U}(0.01, 0.10)$
- $\gamma \sim \mathcal{U}(0.5, 1.5)$
- $\delta \sim \mathcal{U}(0.01, 0.10)$

---

## Data Representation

Each observation is a flattened vector:

$$
x \in \mathbb{R}^{2 \times N_{\text{sub}}}
$$

where $N_{\text{sub}}$ depends on:

- total integration time (`days`)
- time resolution (`saveat`)
- subsampling stride (`subsample_stride`)

Datasets follow the standard pipeline contract:

- `theta_*`: shape `(N, 4)`
- `x_*`: shape `(N, D)` where $D$ is the summary dimension

---

## Dataset Generation

The dataset is generated using:

```bash
generate_dataset.py
```

### Create dataset

```bash
python generate_dataset.py \
  --out datasets/lotka_volterra/lotka_volterra_no_masks.h5
```

#### Available Arguments
- `--out` (str, required) Output path to HDF5 file
- `--n_train` (int, default=10000) Number of training samples
- `--n_val` (int, default=2000) Number of validation samples
- `--n_test` (int, default=2000) Number of test samples
- `--seed` (int, default=57038) Random seed for reproducibility
- `--days` (float, default=20.0) Total simulation time
- `--saveat` (float, default=0.1) Time resolution for ODE solver
- `--subsample_stride` (int, default=3) Subsampling step applied to the trajectory
- `--obs_noise_log_scale` (float, default=0.05) Standard deviation of log-normal observation noise

---

### Purpose

This benchmark is designed to:

- evaluate inference on nonlinear dynamical systems,
- test robustness to multiplicative (log-normal) noise,
- provide a controlled setting for SBI methods such as BNRE + HMC,
- and serve as a bridge between simple toy models and real scientific applications.

---

### Notes
- The dataset is generated on demand and is not version-controlled.
- Regenerating with the same seed yields identical results.
- The dimensionality of the observations depends on the subsampling configuration.
- Outputs are guaranteed to be finite; failed simulations are replaced with NaNs during generation and should be filtered if necessary.