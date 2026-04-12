from __future__ import annotations

import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import matplotlib.pyplot as plt
import h5py
import corner
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.spatial import ConvexHull


@hydra.main(config_path="../../configs/amber501_p1d", config_name="hmc", version_base="1.3")
def main(cfg: DictConfig):
    device = cfg.get("device", "cpu")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    elif device == "single_gpu":
        os.environ["JAX_PLATFORMS"] = "cuda"
    else:
        raise ValueError(f"Unknown device={device!r}. Use 'cpu' or 'single_gpu'.")

    import jax
    import jax.numpy as jnp
    import numpyro

    if device == "cpu":
        numpyro.set_host_device_count(4)

    from jax_bnre_hmc.checkpointing import load_best_params, resolve_run_train_config_path
    from jax_bnre_hmc.hmc import (
        ConvexHullPrior,
        make_log_ratio_fn,
        make_potential_fn,
        run_nuts,
        z_to_theta,
        sample_uniform_in_convex_hull,
    )
    from jax_bnre_hmc.model import RatioEstimatorMLP
    from jax_bnre_hmc.plotting import plot_tarp_ecp_curve
    from jax_bnre_hmc.plot_style import apply_plot_style
    from jax_bnre_hmc.diagnostics import run_tarp_jax, l2_distance

    print(f"JAX backend: {jax.default_backend()}")
    apply_plot_style()
    dataset_file = cfg.data.get("dataset_file")
    if dataset_file is None:
        raise ValueError(
            "data.dataset_file must be set for amber501_p1d HMC (path to HDF5 with 'params', 'subset_of_mock_datasets')"
        )
    dataset_file = str(dataset_file)

    run_dir = Path(cfg.run_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else run_dir / "hmc_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hmc.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    run_cfg = OmegaConf.load(resolve_run_train_config_path(run_dir))
    ckpt_dirname = run_cfg.train.get("checkpoint_dirname", "checkpoints")
    best_dir = run_dir / ckpt_dirname / "best"

    model = RatioEstimatorMLP(
        hidden_dims=tuple(run_cfg.model.hidden_dims),
        activation=str(run_cfg.model.activation),
        norm=str(run_cfg.model.norm),
    )
    params = load_best_params(best_dir=best_dir)

    print(f"\nLoading dataset from {dataset_file}")
    test_frac = float(cfg.get("data_split_test_fraction", 0.2))
    val_frac = float(cfg.get("data_split_validation_fraction", 0.2))

    with h5py.File(dataset_file, "r") as f:
        params_all = f["params"][:, :5]
        mocks = f["subset_of_mock_datasets"][:]

    theta_train, theta_test, x_train, x_test = train_test_split(
        params_all, mocks, test_size=test_frac, random_state=152637
    )
    theta_train, theta_val, x_train, x_val = train_test_split(
        theta_train, x_train, test_size=val_frac, random_state=152637
    )

    def expand_theta_mock_pairs(theta, x):
        n_theta, n_mocks, x_dim = x.shape
        theta_pairs = np.repeat(theta, n_mocks, axis=0)
        x_pairs = x.reshape(n_theta * n_mocks, x_dim)
        return theta_pairs, x_pairs

    theta_train_pairs, x_train_pairs = expand_theta_mock_pairs(theta_train, x_train)
    theta_val_pairs, x_val_pairs = expand_theta_mock_pairs(theta_val, x_val)
    theta_test_pairs, x_test_pairs = expand_theta_mock_pairs(theta_test, x_test)

    _, theta_train, _, x_train = train_test_split(
        theta_train_pairs, x_train_pairs, test_size=250000, random_state=47805
    )
    _, theta_val, _, x_val = train_test_split(
        theta_val_pairs, x_val_pairs, test_size=50000, random_state=940856
    )
    _, theta_test, _, x_test = train_test_split(
        theta_test_pairs, x_test_pairs, test_size=50000, random_state=496702
    )

    theta_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    theta_scaler.fit(theta_train)
    x_scaler.fit(x_train)
    theta_train = theta_scaler.transform(theta_train)
    theta_val = theta_scaler.transform(theta_val)
    theta_test = theta_scaler.transform(theta_test)
    x_train = x_scaler.transform(x_train)
    x_val = x_scaler.transform(x_val)
    x_test = x_scaler.transform(x_test)

    params_all_scaled = theta_scaler.transform(params_all)
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
        init_z = jnp.zeros((num_chains, D), dtype=jnp.float32)
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
    corner_labels = list(cfg.get("corner_labels", ["z_mid", "Delta_z", "A_z", "T_0", "<F>"]))
    for idx in selected_indices:
        samples = np.array(posterior_samples[idx].T)
        true_params = np.array(theta_true[idx])
        figure = corner.corner(
            samples,
            labels=corner_labels,
            truths=true_params,
            show_titles=False,
        )
        figure.savefig(output_dir / f"corner_observation_{idx}.pdf")
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
    plot_tarp_ecp_curve(output_dir, alpha_grid, ecp)

    with h5py.File(output_dir / "posterior_samples.h5", "w") as f:
        f.create_dataset("posterior_samples", data=np.array(posterior_samples))
        f.create_dataset("theta_true", data=np.array(theta_true))
        f.create_dataset("x_obs", data=np.array(x_obs))


if __name__ == "__main__":
    main()
