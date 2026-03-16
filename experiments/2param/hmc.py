from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpyro
import corner
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.hmc import BoxPrior, make_log_ratio_fn, make_potential_fn, run_nuts, z_to_theta
from jax_bnre_hmc.model import RatioEstimatorMLP
from jax_bnre_hmc.diagnostics import run_tarp_jax, l2_distance

numpyro.set_host_device_count(4)


def uniform_param_sampling_and_mock_selection(
    params_scaled_reshaped,
    mocks_scaled_reshaped,
    N_samples=100,
    seed=17,
):
    """Sample N parameter combinations, find nearest grid point, select one mock per combo."""
    np.random.seed(seed)
    Nmfp, Nflux, Nmock, _ = params_scaled_reshaped.shape
    _, _, _, Nbins = mocks_scaled_reshaped.shape
    grid_params = params_scaled_reshaped[:, :, 0, :].reshape(-1, 2)
    param_mins = grid_params.min(axis=0)
    param_maxs = grid_params.max(axis=0)
    uniform_samples = np.random.uniform(param_mins, param_maxs, size=(N_samples, 2))
    tree = cKDTree(grid_params)
    _, nn_indices = tree.query(uniform_samples, k=1)
    i_mfp = nn_indices // Nflux
    i_flux = nn_indices % Nflux
    mocks_to_infer = np.empty((N_samples, Nbins))
    truths_to_infer = np.empty((N_samples, 2))
    for i in range(N_samples):
        j_mock = np.random.randint(0, Nmock)
        mocks_to_infer[i] = mocks_scaled_reshaped[i_mfp[i], i_flux[i], j_mock, :]
        truths_to_infer[i] = params_scaled_reshaped[i_mfp[i], i_flux[i], j_mock, :]
    return truths_to_infer, mocks_to_infer, uniform_samples


@hydra.main(config_path="../../configs/2param", config_name="hmc", version_base="1.3")
def main(cfg: DictConfig):
    dataset_file = cfg.data.get("dataset_file")
    if dataset_file is None:
        raise ValueError("data.dataset_file must be set for 2param HMC (path to HDF5 with 'theta' and 'x')")
    dataset_file = str(dataset_file)

    run_dir = Path(cfg.run_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else run_dir / "hmc_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = OmegaConf.load(run_dir / "config.yaml")
    ckpt_dirname = run_cfg.train.get("checkpoint_dirname", "checkpoints")
    best_dir = run_dir / ckpt_dirname / "best"

    model = RatioEstimatorMLP(
        hidden_dims=tuple(run_cfg.model.hidden_dims),
        activation=str(run_cfg.model.activation),
        norm=str(run_cfg.model.norm),
    )
    params = load_best_params(best_dir=best_dir)

    print(f"\nLoading dataset from {dataset_file}")
    with h5py.File(dataset_file, "r") as f:
        theta = f["theta"][:]
        x = f["x"][:]
    theta_scaler = MinMaxScaler()
    x_scaler = MinMaxScaler()
    theta_scaled = theta_scaler.fit_transform(theta)
    x_scaled = x_scaler.fit_transform(x)
    dataset_params_scaled_reshaped = theta_scaled.reshape(62, 9, 1000, 2)
    dataset_mocks_scaled_reshaped = x_scaled.reshape(62, 9, 1000, 22)

    obs_seed = int(cfg.get("observation_selection_seed", cfg.seed))
    n_obs = int(cfg.n_observations)
    theta_true, x_obs, _ = uniform_param_sampling_and_mock_selection(
        params_scaled_reshaped=dataset_params_scaled_reshaped,
        mocks_scaled_reshaped=dataset_mocks_scaled_reshaped,
        N_samples=n_obs,
        seed=obs_seed,
    )

    prior = BoxPrior(
        low=jnp.array(cfg.prior.low, dtype=jnp.float32),
        high=jnp.array(cfg.prior.high, dtype=jnp.float32),
    )

    num_chains = int(cfg.num_chains)
    posteriors_list = []
    print("Starting inference")
    for i in range(n_obs):
        print(f"\nRunning observation {i+1}/{n_obs}")
        x_obs_i = x_obs[i].squeeze()
        log_ratio = make_log_ratio_fn(model.apply, params, x_obs_i)
        potential = make_potential_fn(log_ratio, prior)
        D = prior.low.shape[0]
        init_z = jnp.zeros((num_chains, D), dtype=jnp.float32)
        mcmc = run_nuts(
            potential,
            jax.random.PRNGKey(int(cfg.seed) + i),
            init_z,
            num_warmup=int(cfg.num_warmup),
            num_samples=int(cfg.num_samples),
            num_chains=num_chains,
        )
        z_samples = mcmc.get_samples(group_by_chain=False)
        theta_samples, _ = jax.vmap(lambda z: z_to_theta(z, prior))(z_samples)
        posteriors_list.append(theta_samples)
        print(theta_samples.shape)
        print(mcmc.print_summary())

    posterior_samples = jnp.stack(posteriors_list, axis=1)
    posterior_samples = jnp.transpose(posterior_samples, (1, 2, 0))
    print("All done.")
    print("Posterior samples shape:", posterior_samples.shape)

    n_plots = min(int(cfg.n_plots), n_obs)
    rng = np.random.default_rng(1234)
    selected_indices = rng.choice(n_obs, size=n_plots, replace=False)
    corner_labels = list(cfg.get("corner_labels", ["mfp", "<F>"]))
    for idx in selected_indices:
        samples = np.array(posterior_samples[idx].T)
        true_params = np.array(theta_true[idx])
        figure = corner.corner(
            samples,
            labels=corner_labels,
            truths=true_params,
            show_titles=True,
            title_fmt=".3f",
            title_kwargs={"fontsize": 12},
        )
        figure.suptitle(f"Posterior for Observation {idx}", fontsize=16)
        figure.savefig(output_dir / f"corner_observation_{idx}.png")
        print(f"Saved corner plot for observation {idx}")
        plt.close(figure)

    posterior_samples = jnp.transpose(posterior_samples, (2, 0, 1))
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    references = jax.random.uniform(
        subkey,
        shape=(n_obs, 2),
        minval=jnp.array(cfg.prior.low, dtype=jnp.float32),
        maxval=jnp.array(cfg.prior.high, dtype=jnp.float32),
    )
    ecp, alpha_grid = run_tarp_jax(
        posterior_samples=posterior_samples,
        thetas=theta_true,
        references=references,
        distance=l2_distance,
        num_bins=30,
        z_score_theta=True,
        eps=1e-10,
    )
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
    plt.savefig(output_dir / "tarp_ecp_curve.png")
    plt.close()

    with h5py.File(output_dir / "posterior_samples.h5", "w") as f:
        f.create_dataset("posterior_samples", data=np.array(posterior_samples))
        f.create_dataset("theta_true", data=np.array(theta_true))
        f.create_dataset("x_obs", data=np.array(x_obs))


if __name__ == "__main__":
    main()
