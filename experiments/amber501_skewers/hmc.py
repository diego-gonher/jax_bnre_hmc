from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import h5py
import numpyro
import corner
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.datasets import load_hdf5_dataset
from jax_bnre_hmc.hmc import ConvexHullPrior, make_log_ratio_fn, make_potential_fn, run_nuts, z_to_theta, sample_uniform_in_convex_hull
from jax_bnre_hmc.model import RatioEstimatorMLP
from jax_bnre_hmc.diagnostics import run_tarp_jax, l2_distance

numpyro.set_host_device_count(4)


@hydra.main(config_path="../../configs/amber501_skewers", config_name="hmc", version_base="1.3")
def main(cfg: DictConfig):
    dataset_file = cfg.data.get("dataset_file")
    if dataset_file is None:
        raise ValueError(
            "data.dataset_file must be set for sinusoid HMC "
            "(path to HDF5 with 'theta_train', 'x_train', 'theta_val', 'x_val', 'theta_test', 'x_test')"
        )
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
    loaded = load_hdf5_dataset(dataset_file, validate=True)
    splits = loaded.splits

    theta_train_raw = splits.theta_train
    x_train_raw = splits.x_train
    theta_val_raw = splits.theta_val
    x_val_raw = splits.x_val
    theta_test_raw = splits.theta_test
    x_test_raw = splits.x_test

    theta_scaler = MinMaxScaler(feature_range=(-1, 1))
    theta_scaler.fit(theta_train_raw)
    theta_train = theta_scaler.transform(theta_train_raw)
    theta_val = theta_scaler.transform(theta_val_raw)
    theta_test = theta_scaler.transform(theta_test_raw)

    x_train = np.log1p(x_train_raw)
    x_val = np.log1p(x_val_raw)
    x_test = np.log1p(x_test_raw)
    x_scaler = StandardScaler()
    x_scaler.fit(x_train)
    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)

    params_all_scaled = theta_scaler.transform(np.concatenate((theta_train, theta_val, theta_test), axis=0))
    hull = ConvexHull(params_all_scaled)
    prior = ConvexHullPrior(
        low=jnp.asarray(hull.min_bound),
        high=jnp.asarray(hull.max_bound),
        equations=jnp.asarray(hull.equations),
    )

    n_obs = int(cfg.n_observations)
    seed = int(cfg.seed)
    _, theta_true, _, x_obs = train_test_split(
        theta_test, x_test, test_size=n_obs, random_state=seed
    )

    num_chains = int(cfg.num_chains)
    posteriors_list = []
    print("Starting inference")
    for i in range(n_obs):
        print(f"\nRunning observation {i+1}/{n_obs}")
        x_obs_i = x_obs[i].squeeze()
        log_ratio = make_log_ratio_fn(model.apply, params, x_obs_i)
        potential = make_potential_fn(log_ratio, prior, soft_hull=False)
        D = prior.low.shape[0]
        init_z = jnp.zeros((num_chains, D), dtype=jnp.float64)
        mcmc = run_nuts(
            potential,
            jax.random.PRNGKey(seed + i),
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
    corner_labels = list(cfg.get("corner_labels", ["z_mid", "Delta_z", "A_z", "T_0"]))
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
        plt.close("all")

    posterior_samples = jnp.transpose(posterior_samples, (2, 0, 1))
    key = jax.random.PRNGKey(42)
    references = sample_uniform_in_convex_hull(
        key=key, prior=prior, n_samples=n_obs, batch_size=8192
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
