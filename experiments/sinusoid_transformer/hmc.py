import os
import jax
import jax.numpy as jnp
import numpy as np
import h5py
import numpyro
import matplotlib.pyplot as plt
import corner

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.hmc import (
    BoxPrior,
    make_log_ratio_fn,
    make_potential_fn,
    run_nuts,
    z_to_theta,
)
from jax_bnre_hmc.model import RatioEstimatorTransformer
from jax_bnre_hmc.diagnostics import run_tarp_jax, l2_distance

numpyro.set_host_device_count(4)

# -------------------------
# 1) Rebuild architecture
# -------------------------
model = RatioEstimatorTransformer(
    d_model=16,
    num_layers=2,
    num_heads=2,
    transformer_mlp_dim=128,
    transformer_activation="gelu",
    head_hidden_dims=(50, 50, 50),
    head_activation="tanh",
    head_norm="layernorm",
)

best_dir = "/Users/diegogonzalez/Documents/Research/ENIGMA/BNRE-HMC/jax_bnre_hmc/outputs/sinusoid_transformer/2026-03-13_16-13-07/checkpoints/best/"
params = load_best_params(best_dir=best_dir)

# Output directory
output_dir = '/'.join(best_dir.split('/')[:-3]) + '/hmc_results/'
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# 2) Load dataset
# -------------------------
dataset_file = "/Users/diegogonzalez/Documents/Research/ENIGMA/BNRE-HMC/jax_bnre_hmc/datasets/sinusoid/sinusoid_noisy_masked_nsim20000_ntime50_seed117.h5"

print(f"\nLoading dataset from {dataset_file}")

with h5py.File(dataset_file, "r") as f:
    theta = f["theta"][:]
    y_obs = f["y_obs"][:]
    mask = f["mask"][:].astype(np.float32)

print(f"\ntheta shape: {theta.shape}")
print(f"y_obs shape: {y_obs.shape}")
print(f"mask shape:  {mask.shape}")

# -------------------------
# 3) Reproduce training preprocessing
# -------------------------
valid = mask > 0.5
valid_counts = np.sum(valid, axis=0)
valid_sums = np.sum(y_obs * valid, axis=0)
col_means = valid_sums / np.maximum(valid_counts, 1)

y_obs_filled = np.where(valid, y_obs, col_means[None, :])

theta_scaler = MinMaxScaler(feature_range=(-1, 1))
y_scaler = MinMaxScaler(feature_range=(-1, 1))

theta_scaled = theta_scaler.fit_transform(theta).astype(np.float32)
y_obs_scaled = y_scaler.fit_transform(y_obs_filled).astype(np.float32)

# actual model input: zero masked entries
y_obs_scaled = y_obs_scaled * mask
x_tokens = np.stack([y_obs_scaled, mask], axis=-1).astype(np.float32)

print(f"\nscaled theta shape: {theta_scaled.shape}")
print(f"scaled y_obs shape: {y_obs_scaled.shape}")
print(f"x_tokens shape:     {x_tokens.shape}")

# -------------------------
# 4) Select observations
# -------------------------
N_OBSERVATIONS = 500

print("\nSelecting observations for inference...")
_, theta_true, _, x_obs = train_test_split(
    theta_scaled,
    x_tokens,
    test_size=N_OBSERVATIONS,
    random_state=2401,
)

print(f"theta_true shape: {theta_true.shape}")
print(f"x_obs shape:      {x_obs.shape}")

# -------------------------
# 5) Prior in scaled theta space
# -------------------------
prior = BoxPrior(
    low=jnp.array([-1.0, -1.0, -1.0, -1.0], dtype=jnp.float32),
    high=jnp.array([ 1.0,  1.0,  1.0,  1.0], dtype=jnp.float32),
)

print("Starting inference")

posteriors_list = []

# Iterate over observations
for i in range(N_OBSERVATIONS):
    print(f"\nRunning observation {i+1}/{N_OBSERVATIONS}")

    x_obs_i = jnp.asarray(x_obs[i], dtype=jnp.float32)          # (N, 2)
    theta_true_i = jnp.asarray(theta_true[i], dtype=jnp.float32)  # (4,)

    # sanity check: at least one valid token
    assert jnp.sum(x_obs_i[:, 1] > 0.5) > 0, f"Obs {i} has no valid tokens."

    # log-ratio wrapper
    log_ratio = make_log_ratio_fn(model.apply, params, x_obs_i)

    # potential in unconstrained z-space
    potential = make_potential_fn(log_ratio, prior)

    # optional gradient sanity check
    test_grad = jax.grad(log_ratio)(theta_true_i)
    if not bool(jnp.all(jnp.isfinite(test_grad))):
        raise ValueError(f"Non-finite log_ratio gradient for observation {i}")

    # init positions for chains
    num_chains = 4
    D = prior.low.shape[0]
    init_z = jnp.zeros((num_chains, D), dtype=jnp.float32)

    # run NUTS
    mcmc = run_nuts(
        potential_fn=potential,
        rng_key=jax.random.PRNGKey(i),
        init_z=init_z,
        num_warmup=4000,
        num_samples=4000,
        num_chains=num_chains,
    )

    # samples: numpyro returns z samples; map to theta
    z_samples = mcmc.get_samples(group_by_chain=False)  # (num_chains*num_samples, D)
    theta_samples, _ = jax.vmap(lambda z: z_to_theta(z, prior))(z_samples)

    posteriors_list.append(theta_samples)

    print(theta_samples.shape)
    mcmc.print_summary()

# Stack all posterior samples
posterior_samples = jnp.stack(posteriors_list, axis=1)  # (num_samples*num_chains, N_OBSERVATIONS, D)

# Swap axes to (N, D, S)
posterior_samples = jnp.transpose(posterior_samples, (1, 2, 0))  # (N_OBSERVATIONS, D, num_samples*num_chains)

print("All done.")
print("Posterior samples shape:", posterior_samples.shape)

# -------------------------
# 6) Corner plots
# -------------------------
N_PLOTS = min(25, N_OBSERVATIONS)
rng = np.random.default_rng(1234)
selected_indices = rng.choice(N_OBSERVATIONS, size=N_PLOTS, replace=False)

for idx in selected_indices:
    samples = np.array(posterior_samples[idx].T)   # (S, D)
    true_params = np.array(theta_true[idx])        # (D,)

    figure = corner.corner(
        samples,
        labels=["A", "f", "phi", "b"],
        truths=true_params,
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 12},
    )
    figure.suptitle(f"Posterior for Observation {idx}", fontsize=16)
    outname = f"{output_dir}corner_observation_{idx}.png"
    figure.savefig(outname)
    plt.close(figure)
    print(f"Saved corner plot for observation {idx} as {outname}")

# -------------------------
# 7) Compute TARP
# -------------------------
# posterior_samples: (N, D, S) -> (S, N, D)
posterior_samples_tarp = jnp.transpose(posterior_samples, (2, 0, 1))  # (S, N_OBSERVATIONS, D)

# Sample the prior again for references
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
references = jax.random.uniform(
    subkey,
    shape=(N_OBSERVATIONS, 4),
    minval=jnp.array([-1.0, -1.0, -1.0, -1.0], dtype=jnp.float32),
    maxval=jnp.array([ 1.0,  1.0,  1.0,  1.0], dtype=jnp.float32),
)

ecp, alpha_grid = run_tarp_jax(
    posterior_samples=posterior_samples_tarp,
    thetas=jnp.asarray(theta_true),
    references=references,
    distance=l2_distance,
    num_bins=30,
    z_score_theta=True,
    eps=1e-10,
)

# Plot ECP vs alpha_grid
plt.figure(figsize=(5, 5))
plt.plot(alpha_grid, ecp, marker="o")
plt.plot([0, 1], [0, 1], "k--", label="Ideal")
plt.xlabel("Credibility Level (α)")
plt.ylabel("Empirical Coverage Probability (ECP)")
plt.title("TARP: Empirical Coverage Probability Curve")
plt.axis("square")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.savefig(f"{output_dir}tarp_ecp_curve.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved TARP curve as {output_dir}tarp_ecp_curve.png")

# -------------------------
# 8) Save posterior samples
# -------------------------
with h5py.File(output_dir + "posterior_samples.h5", "w") as f:
    f.create_dataset("posterior_samples", data=np.array(posterior_samples_tarp))  # (S, N, D)
    f.create_dataset("theta_true", data=np.array(theta_true))
    f.create_dataset("x_obs", data=np.array(x_obs))

print(f"Saved posterior samples to {output_dir}posterior_samples.h5")
