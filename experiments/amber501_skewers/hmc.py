import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpyro
import corner
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull
numpyro.set_host_device_count(4)

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.hmc import ConvexHullPrior, make_log_ratio_fn, make_potential_fn, run_nuts, z_to_theta, points_in_hull, sample_uniform_in_convex_hull
from jax_bnre_hmc.model import RatioEstimatorMLP
from jax_bnre_hmc.diagnostics import run_tarp_jax, l2_distance

# 1) rebuild the model *architecture* (must match training config)
model = RatioEstimatorMLP(hidden_dims=(256, 256, 256), activation="tanh", norm="layernorm")

# 2) load params (PyTree) and get apply_fn
best_dir = '/Users/diegogonzalez/Documents/Research/ENIGMA/BNRE-HMC/jax_bnre_hmc/outputs/amber501_skewers/2026-03-15_15-53-12/checkpoints/best/'

# Output directory should be best_dir's parent's parent, and create it if it doesn't exist
output_dir = '/'.join(best_dir.split('/')[:-3]) + '/hmc_results/'
os.makedirs(output_dir, exist_ok=True)

params = load_best_params(best_dir=best_dir)   # point this at your Hydra run checkpoints/best

# 3) Load dataset and select mock observations
# Load the dataset and preprocess it
# dataset_file = f'/Users/diegogonzalez/Documents/Research/ENIGMA/jax_bnre_hmc/jax_bnre_hmc/datasets/sinusoid/sinusoid_noisy_masked_nsim20000_ntime50_seed117.h5'
dataset_file = f'/Users/diegogonzalez/Documents/Research/ENIGMA/BNRE-HMC/jax_bnre_hmc/datasets/amber501/tau_skewers_z5.00.hdf5'

print(f"\nLoading dataset from {dataset_file}")

with h5py.File(dataset_file, "r") as f:
    # load the needed arrays
    params_all = f['params'][:, :4]                  # (4509, 5)
    mocks = f['tau_skewers'][:]      # (4509, 1000, 19)

    # Split by parameter combination first
    theta_train, theta_test, x_train, x_test = train_test_split(
        params_all,
        mocks,
        test_size=0.2,
        random_state=152637,
    )

    theta_train, theta_val, x_train, x_val = train_test_split(
        theta_train,
        x_train,
        test_size=0.2,
        random_state=152637,
    )

    def expand_theta_mock_pairs(theta, x):
        """
        Convert:
            theta: (n_theta, n_params)
            x:     (n_theta, n_mocks, x_dim)

        into:
            theta_pairs: (n_theta * n_mocks, n_params)
            x_pairs:     (n_theta * n_mocks, x_dim)
        """
        n_theta, n_mocks, x_dim = x.shape
        n_params = theta.shape[1]

        theta_pairs = np.repeat(theta, n_mocks, axis=0)
        x_pairs = x.reshape(n_theta * n_mocks, x_dim)

        return theta_pairs, x_pairs


    theta_train_pairs, x_train_pairs = expand_theta_mock_pairs(theta_train, x_train)
    theta_val_pairs, x_val_pairs     = expand_theta_mock_pairs(theta_val, x_val)
    theta_test_pairs, x_test_pairs   = expand_theta_mock_pairs(theta_test, x_test)
    

    _1, theta_train, _2, x_train = train_test_split(
        theta_train_pairs,
        x_train_pairs,
        test_size=250000,
        random_state=47805,
    )

    _1, theta_val, _2, x_val = train_test_split(
        theta_val_pairs,
        x_val_pairs,
        test_size=50000,
        random_state=940856,
    )

    _1, theta_test, _2, x_test = train_test_split(
        theta_test_pairs,
        x_test_pairs,
        test_size=50000,
        random_state=496702,
    )

    print(f'\nFinal dataset splits for SBI:')
    print(f' - Train')
    print(f'    - theta shape: {theta_train.shape}')
    print(f'    - x shape: {x_train.shape}')
    print(f' - Validation')
    print(f'    - theta shape: {theta_val.shape}')
    print(f'    - x shape: {x_val.shape}')
    print(f' - Test')
    print(f'    - theta shape: {theta_test.shape}')
    print(f'    - x shape: {x_test.shape}\n')

    # use min max scalers on theta
    # create min max scaler, symmetric around 0
    theta_scaler = MinMaxScaler(feature_range=(-1,1))
    theta_scaler.fit(theta_train)

    # Transform all splits using train-fitted scalers
    theta_train = theta_scaler.transform(theta_train)
    theta_val   = theta_scaler.transform(theta_val)
    theta_test  = theta_scaler.transform(theta_test)

    # plt.hist(theta_train.ravel())
    # plt.show()

    # use a combination of scalers to make x good
    x_scaler_1 = np.log1p
    x_scaler_2 = StandardScaler()
    # x_scaler_3 = MinMaxScaler(feature_range=(-5,5)) 

    # Transform x with first scaler
    x_train = x_scaler_1(x_train)
    x_val   = x_scaler_1(x_val)
    x_test  = x_scaler_1(x_test)

    # Transform all splits using train-fitted with second scaler
    x_scaler_2.fit(x_train)
    x_train = x_scaler_2.transform(x_train)
    x_val   = x_scaler_2.transform(x_val)
    x_test  = x_scaler_2.transform(x_test)

    # scale all the parameters for the convex hull
    params_all_scaled = theta_scaler.transform(params_all)
    hull = ConvexHull(params_all_scaled)

# Randomly select N_OBSERVATIONS from the dataset for inference    
N_OBSERVATIONS = 100

# randomly select N_OBSERVATIONS with the train_test_split function
print("\nSplitting dataset into train and validation sets...")
_1, theta_true, _2, x_obs = train_test_split(theta_test, x_test, 
                                             test_size=N_OBSERVATIONS, 
                                             random_state=2401)

# 4) prior bounds (D,) set after rescaling
prior = ConvexHullPrior(
    low=jnp.asarray(hull.min_bound),
    high=jnp.asarray(hull.max_bound),
    equations=jnp.asarray(hull.equations),
)

print("Starting inference")

posteriors_list = []

# Iterate over observations
for i in range(N_OBSERVATIONS):
    print(f"\nRunning observation {i+1}/{N_OBSERVATIONS}")
    # squeeze to get single observation
    x_obs_i = x_obs[i].squeeze()           
    theta_true_i = theta_true[i].squeeze() 

    # 4) log-ratio wrapper
    log_ratio = make_log_ratio_fn(model.apply, params, x_obs_i)

    # 6) potential in unconstrained z-space
    potential = make_potential_fn(log_ratio, prior, soft_hull=False)

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
N_PLOTS = 25
rng = np.random.default_rng(1234)
selected_indices = rng.choice(N_OBSERVATIONS, size=N_PLOTS, replace=False)  

for idx in selected_indices:
    samples = np.array(posterior_samples[idx].T)  # (D, S)
    true_params = np.array(theta_true[idx])     # (D,)

    figure = corner.corner(
        samples,
        labels=["z_mid", "Delta_z", "A_z", "T_0", "<F>"],
        truths=true_params,
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 12},
    )
    figure.suptitle(f"Posterior for Observation {idx}", fontsize=16)
    figure.savefig(f"{output_dir}corner_observation_{idx}.png")
    print(f"Saved corner plot for observation {idx} as {output_dir}corner_observation_{idx}.png")
    plt.close('all')

# 10) Compute TARP
# posterior_samples: (N, D, S) -> (S, N, D)
posterior_samples = jnp.transpose(posterior_samples, (2, 0, 1))  # (S, N_OBSERVATIONS, D)

# Sample the Prior again for references, use a seed with JAX for reproducibility using random.uniform for the two dimensional uniform prior

# key = jax.random.PRNGKey(42)
# key, subkey = jax.random.split(key)
# references = jax.random.uniform(
#     subkey,
#     shape=(N_OBSERVATIONS, 5),
#     minval=jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=jnp.float32),
#     maxval=jnp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=jnp.float32),
# )

key = jax.random.PRNGKey(42)
references = sample_uniform_in_convex_hull(
    key=key,
    prior=prior,
    n_samples=N_OBSERVATIONS,
    batch_size=8192,
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