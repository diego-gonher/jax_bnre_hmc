import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpyro
import corner
from sklearn.preprocessing import MinMaxScaler
numpyro.set_host_device_count(4)

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.hmc import BoxPrior, make_log_ratio_fn, make_potential_fn, run_nuts, z_to_theta
from jax_bnre_hmc.model import RatioEstimatorMLP
from jax_bnre_hmc.diagnostics import run_tarp_jax, l2_distance

# 1) rebuild the model *architecture* (must match training config)
model = RatioEstimatorMLP(hidden_dims=(50, 50, 50), activation="tanh", norm="layernorm")

# 2) load params (PyTree) and get apply_fn
best_dir = '/Users/diegogonzalez/Documents/Research/ENIGMA/jax_bnre_hmc/jax_bnre_hmc/outputs/2param/2026-01-29_18-23-10_gamma10.0/checkpoints/best/'

# Output directory should be best_dir's parent's parent, and create it if it doesn't exist
output_dir = '/'.join(best_dir.split('/')[:-3]) + '/hmc_results/'
os.makedirs(output_dir, exist_ok=True)

params = load_best_params(best_dir=best_dir)   # point this at your Hydra run checkpoints/best

# 3) Load dataset and select mock observations
# Load the dataset and preprocess it
dataset_file = f'/Users/diegogonzalez/Desktop/models/imported_models/twoparam_mocks_z5.50.hdf5'

print(f'\nLoading dataset from {dataset_file}')

with h5py.File(dataset_file, 'r') as f:
    # load the parameters
    theta = f['theta'][:]
    print(f'\ntheta shape: {theta.shape}')
    # load the mocks and the velocity bins
    x = f['x'][:]
    print(f'x shape: {x.shape}')
    f.close()

    # use min max scalers on both
    # create min max scalers
    theta_scaler = MinMaxScaler()
    x_scaler = MinMaxScaler()

    # fit and transform the parameters and mocks
    theta_scaled = theta_scaler.fit_transform(theta)
    x_scaled = x_scaler.fit_transform(x) 
    print(f'\nscaled theta shape: {theta_scaled.shape}')
    print(f'scaled x shape: {x_scaled.shape}')

    # reshape the dataset
    dataset_params_scaled_reshaped = theta_scaled.reshape(62, 9, 1000, 2)
    dataset_mocks_scaled_reshaped = x_scaled.reshape(62, 9, 1000, 22)

# helper function to do the selection of mocks to be used for inference
def uniform_param_sampling_and_mock_selection(
    params_scaled_reshaped,
    mocks_scaled_reshaped,
    N_samples=100,
    seed=17
):
    """
    Sample N parameter combinations uniformly across the prior ranges,
    find nearest grid point in the simulation dataset,
    and select one random mock for each parameter combo.

    Args:
        params_unscaled_reshaped: np.ndarray of shape (Nmfp, Nflux, Nmock, 2)
        mocks_unscaled_reshaped: np.ndarray of shape (Nmfp, Nflux, Nmock, Nbins)
        N_samples: number of uniform samples to generate
        seed: random seed for reproducibility

    Returns:
        mocks_to_infer: np.ndarray of shape (N_samples, Nbins)
        truths_to_infer: np.ndarray of shape (N_samples, 2)
    """
    np.random.seed(seed)

    Nmfp, Nflux, Nmock, _ = params_scaled_reshaped.shape
    _, _, _, Nbins = mocks_scaled_reshaped.shape

    # Flatten parameter grid to (Nmfp*Nflux, 2)
    grid_params = params_scaled_reshaped[:, :, 0, :].reshape(-1, 2)

    # Define uniform sampling ranges based on grid min/max
    param_mins = grid_params.min(axis=0)
    param_maxs = grid_params.max(axis=0)

    # Sample uniformly in that range
    uniform_samples = np.random.uniform(param_mins, param_maxs, size=(N_samples, 2))

    # For each uniform sample, find nearest neighbor in grid
    from scipy.spatial import cKDTree
    tree = cKDTree(grid_params)
    _, nn_indices = tree.query(uniform_samples, k=1)

    # Map 1D index back to (i_mfp, i_flux)
    i_mfp = nn_indices // Nflux
    i_flux = nn_indices % Nflux

    mocks_to_infer = np.empty((N_samples, Nbins))
    truths_to_infer = np.empty((N_samples, 2))

    for i in range(N_samples):
        # Randomly pick one of the 1000 mocks for that parameter combo
        j_mock = np.random.randint(0, Nmock)
        mocks_to_infer[i] = mocks_scaled_reshaped[i_mfp[i], i_flux[i], j_mock, :]
        truths_to_infer[i] = params_scaled_reshaped[i_mfp[i], i_flux[i], j_mock, :]

    return truths_to_infer, mocks_to_infer, uniform_samples

N_OBSERVATIONS = 500

# Simulate the full dataset
theta_true, x_obs, _ = uniform_param_sampling_and_mock_selection(params_scaled_reshaped=dataset_params_scaled_reshaped,
                                                                 mocks_scaled_reshaped=dataset_mocks_scaled_reshaped,
                                                                 N_samples=N_OBSERVATIONS)

# 4) prior bounds (D,) set after rescaling
prior = BoxPrior(
    low=jnp.array([0.0, 0.0], dtype=jnp.float32),
    high=jnp.array([ 1.0,  1.0], dtype=jnp.float32),
)

print("Starting inference")

posteriors_list = []

# Iterate over observations
for i in range(N_OBSERVATIONS):
    print(f"\nRunning observation {i+1}/{N_OBSERVATIONS}")
    # squeeze to get single observation
    x_obs_i = x_obs[i].squeeze()            # now shape (10,)
    theta_true_i = theta_true[i].squeeze()  # shape (2,)

    # 4) log-ratio wrapper
    log_ratio = make_log_ratio_fn(model.apply, params, x_obs_i)

    # 6) potential in unconstrained z-space
    potential = make_potential_fn(log_ratio, prior)

    # 7) init positions for chains
    num_chains = 4
    D = prior.low.shape[0]
    init_z = jnp.zeros((num_chains, D), dtype=jnp.float32)

    # 8) run NUTS
    mcmc = run_nuts(potential, jax.random.PRNGKey(0), init_z, num_warmup=4000, num_samples=4000, num_chains=num_chains)

    # 9) samples: numpyro will return z samples; map to theta
    z_samples = mcmc.get_samples(group_by_chain=False)  # (num_chains*num_samples, D)
    theta_samples, _ = jax.vmap(lambda z: z_to_theta(z, prior))(z_samples)

    posteriors_list.append(theta_samples)

    print(theta_samples.shape)
    print(mcmc.print_summary())

# Stack all posterior samples
posterior_samples = jnp.stack(posteriors_list, axis=1)  # (num_samples*num_chains, N_OBSERVATIONS, D)
# Swap axes to (N, D, S)
posterior_samples = jnp.transpose(posterior_samples, (1, 2, 0))  # (N_OBSERVATIONS, D, num_samples*num_chains)

print("All done.")
print("Posterior samples shape:", posterior_samples.shape)

# Optional: plot corner plots N_PLOTS randomly selected observations
N_PLOTS = 10
rng = np.random.default_rng(1234)
selected_indices = rng.choice(N_OBSERVATIONS, size=N_PLOTS, replace=False)  

for idx in selected_indices:
    samples = np.array(posterior_samples[idx].T)  # (D, S)
    true_params = np.array(theta_true[idx])     # (D,)

    figure = corner.corner(
        samples,
        labels=["mfp", "<F>"],
        truths=true_params,
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 12},
    )
    figure.suptitle(f"Posterior for Observation {idx}", fontsize=16)
    figure.savefig(f"{output_dir}corner_observation_{idx}.png")
    print(f"Saved corner plot for observation {idx} as {output_dir}corner_observation_{idx}.png")

# 10) Compute TARP
# posterior_samples: (N, D, S) -> (S, N, D)
posterior_samples = jnp.transpose(posterior_samples, (2, 0, 1))  # (S, N_OBSERVATIONS, D)

# Sample the Prior again for references, use a seed with JAX for reproducibility using random.uniform for the two dimensional uniform prior
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
references = jax.random.uniform(
    subkey,
    shape=(N_OBSERVATIONS, 2),
    minval=jnp.array([0.0, 0.0], dtype=jnp.float32),
    maxval=jnp.array([1.0, 1.0], dtype=jnp.float32),
)

ecp, alpha_grid = run_tarp_jax(
    posterior_samples=posterior_samples,
    thetas=theta_true,
    references=references,
    distance=l2_distance,  # defaults to l2_distance
    num_bins=30,
    z_score_theta=True,
    eps=1e-10,
)

# Make a plot with ECP vs alpha_grid, make it square aspect ratio
plt.figure(figsize=(5, 5))
plt.plot(alpha_grid, ecp, marker='o')
plt.plot([0, 1], [0, 1], 'k--', label='Ideal')
plt.xlabel('Credibility Level (α)')
plt.ylabel('Empirical Coverage Probability (ECP)')
plt.title('TARP: Empirical Coverage Probability Curve')
plt.axis('square')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.savefig(f"{output_dir}tarp_ecp_curve.png")
plt.close()

# Save posterior samples to HDF5, these are unscaled for now
with h5py.File(output_dir + "posterior_samples.h5", "w") as f:
    f.create_dataset("posterior_samples", data=np.array(posterior_samples))
    f.create_dataset("theta_true", data=np.array(theta_true))
    f.create_dataset("x_obs", data=np.array(x_obs))