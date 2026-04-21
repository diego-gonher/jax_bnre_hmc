# SIR Benchmark

This benchmark provides a classical **epidemiological dynamical system** for simulation-based inference (SBI), based on the SIR (Susceptible–Infected–Recovered) model. It is designed to test inference pipelines on:

- nonlinear dynamical systems with conservation structure,
- multi-dimensional coupled time series,
- and multiplicative observation noise.

This benchmark complements Lotka–Volterra by introducing a system with different dynamics and constraints.

---

## Generative Model

The system evolves according to the SIR equations:

$$
\frac{dS}{dt} = -\beta \frac{S I}{N}
$$

$$
\frac{dI}{dt} = \beta \frac{S I}{N} - \gamma I
$$

$$
\frac{dR}{dt} = \gamma I
$$

where:

- $S(t)$: susceptible population  
- $I(t)$: infected population  
- $R(t)$: recovered population  
- $N$: total population (constant)

The parameters are:

$$
\theta = (\beta, \gamma)
$$

- $\beta$: infection rate  
- $\gamma$: recovery rate  

---

## Simulation Details

- Total population:
  $$
  N = 1{,}000{,}000
  $$

- Initial conditions:
  $$
  S_0 = N - I_0 - R_0, \quad I_0 = 1.0, \quad R_0 = 0.0
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

As in the Lotka–Volterra benchmark, we construct a **summary representation**:

1. Subsample the trajectory every `subsample_stride` time steps  
2. Flatten all three populations $(S, I, R)$ into a single vector  
3. Add **multiplicative log-normal noise**

Formally:

$$
x_{\text{sub}} = \text{subsample}(S(t), I(t), R(t))
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

- $\beta \sim \mathcal{U}(0.2, 0.6)$
- $\gamma \sim \mathcal{U}(0.08, 0.18)$

These ranges were chosen to produce realistic epidemic dynamics over the simulation window.

---

## Data Representation

Each observation is a flattened vector:

$$
x \in \mathbb{R}^{3 \times N_{\text{sub}}}
$$

where $N_{\text{sub}}$ depends on:

- total integration time (`days`)
- time resolution (`saveat`)
- subsampling stride (`subsample_stride`)

Datasets follow the standard pipeline contract:

- `theta_*`: shape `(N, 2)`
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
  --out datasets/sir/sir_no_masks.h5
```

#### Available Arguments
- `--out` (str, required) Output path to HDF5 file
- `--n_train` (int, default=10000) Number of training samples
- `--n_val` (int, default=2000) Number of validation samples
- `--n_test` (int, default=2000) Number of test samples
- `--seed` (int, default=57038) Random seed for reproducibility
- `--N` (float, default=1_000_000.0) Total population size
- `--I0` (float, default=1.0) Initial infected population
- `--R0` (float, default=0.0) Initial recovered population
- `--days` (float, default=160.0) Total simulation time
- `--saveat` (float, default=1.0) Time resolution for ODE solver
- `--subsample_stride` (int, default=2) Subsampling step applied to all populations
- `--obs_noise_log_scale` (float, default=0.05) Standard deviation of log-normal observation noise

---

### Purpose

This benchmark is designed to:

- evaluate inference on epidemic dynamical systems,
- test robustness to multiplicative (log-normal) noise,
- provide a structured multi-dimensional SBI task,
- and complement Lotka–Volterra with a system exhibiting conservation of total population.

---

### Notes
- The dataset is generated on demand and is not version-controlled.
- Regenerating with the same seed yields identical results.
- The dimensionality of the observations depends on the subsampling configuration.
- Outputs are expected to be finite; invalid simulations are replaced with NaNs and may be filtered if needed.