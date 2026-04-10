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

from jax_bnre_hmc.checkpointing import load_best_params, resolve_run_train_config_path
from jax_bnre_hmc.datasets import load_hdf5_dataset
from jax_bnre_hmc.hmc import (
    ConvexHullPrior,
    make_log_ratio_fn,
    make_potential_fn,
    run_nuts,
    z_to_theta,
    sample_uniform_in_convex_hull,
)
from jax_bnre_hmc.model import RatioEstimatorTransformer
from jax_bnre_hmc.plotting import plot_tarp_ecp_curve
from jax_bnre_hmc.plot_style import apply_plot_style
from jax_bnre_hmc.diagnostics import run_tarp_jax, l2_distance

numpyro.set_host_device_count(4)


@hydra.main(
    config_path="../../configs/amber501_flux_skewers_transformer",
    config_name="hmc",
    version_base="1.3",
)
def main(cfg: DictConfig):
    apply_plot_style()
    dataset_file = cfg.data.get("dataset_file")
    if dataset_file is None:
        raise ValueError(
            "data.dataset_file must be set for amber501_flux_skewers_transformer experiment "
            "(path to HDF5 with theta_train, x_train, x_train_mask; "
            "theta_val, x_val, x_val_mask; theta_test, x_test, x_test_mask)"
        )
    dataset_file = str(dataset_file)

    run_dir = Path(cfg.run_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else run_dir / "hmc_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hmc.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    run_cfg = OmegaConf.load(resolve_run_train_config_path(run_dir))
    ckpt_dirname = run_cfg.train.get("checkpoint_dirname", "checkpoints")
    best_dir = run_dir / ckpt_dirname / "best"

    m = run_cfg.model
    model = RatioEstimatorTransformer(
        d_model=int(m.d_model),
        num_layers=int(m.num_layers),
        num_heads=int(m.num_heads),
        transformer_mlp_dim=int(m.transformer_mlp_dim),
        transformer_activation=str(m.transformer_activation),
        head_hidden_dims=tuple(m.head_hidden_dims),
        head_activation=str(m.head_activation),
        head_norm=str(m.head_norm),
    )
    params = load_best_params(best_dir=best_dir)

    # -----------------------------
    # Load dataset
    # -----------------------------
    print(f"\nLoading dataset from {dataset_file}")
    loaded = load_hdf5_dataset(dataset_file, validate=True)
    splits = loaded.splits

    theta_train_raw = splits.theta_train
    x_train_raw = splits.x_train
    mask_train_raw = splits.mask_train
    theta_val_raw = splits.theta_val
    x_val_raw = splits.x_val
    mask_val_raw = splits.mask_val
    theta_test_raw = splits.theta_test
    x_test_raw = splits.x_test
    mask_test_raw = splits.mask_test

    if mask_train_raw is None or mask_val_raw is None or mask_test_raw is None:
        raise ValueError(
            "Amber501 flux skewers transformer dataset must provide x_train_mask, x_val_mask, "
            "x_test_mask in the HDF5 file, with shapes matching x_train, x_val, x_test."
        )

    print("\nLoaded dataset splits (raw):")
    print(f" - theta_train shape: {theta_train_raw.shape}")
    print(f" - x_train shape:     {x_train_raw.shape}")
    print(f" - mask_train shape:  {mask_train_raw.shape}")
    print(f" - theta_val shape:   {theta_val_raw.shape}")
    print(f" - x_val shape:       {x_val_raw.shape}")
    print(f" - mask_val shape:    {mask_val_raw.shape}")
    print(f" - theta_test shape:  {theta_test_raw.shape}")
    print(f" - x_test shape:      {x_test_raw.shape}")
    print(f" - mask_test shape:   {mask_test_raw.shape}")

    # -----------------------------
    # Convert tau -> flux
    # -----------------------------
    x_train_raw = np.exp(-x_train_raw)
    x_val_raw = np.exp(-x_val_raw)
    x_test_raw = np.exp(-x_test_raw)

    # -----------------------------
    # Scale theta and x (with masks)
    # -----------------------------

    # Theta: same MinMaxScaler(-1, 1) as amber501_flux_skewers_transformer train.py
    theta_scaler = MinMaxScaler(feature_range=(-1, 1))
    theta_train = theta_scaler.fit_transform(theta_train_raw).astype(np.float32)

    # X: StandardScaler pipeline, fit only on filled valid train tokens
    mask_train = mask_train_raw.astype(np.float32)
    valid_train = mask_train > 0.5
    valid_counts_train = np.sum(valid_train, axis=0)
    valid_sums_train = np.sum(x_train_raw * valid_train, axis=0)
    col_means_train = valid_sums_train / np.maximum(valid_counts_train, 1)
    x_train_filled = np.where(valid_train, x_train_raw, col_means_train[None, :])

    x_scaler = StandardScaler()
    x_scaler.fit(x_train_filled)

    x_train_scaled = x_scaler.transform(x_train_filled).astype(np.float32)
    x_train_scaled = x_train_scaled * mask_train
    x_train_tokens = np.stack([x_train_scaled, mask_train], axis=-1).astype(np.float32)

    print(f"\nscaled theta_train shape: {theta_train.shape}")
    print(f"scaled x_train shape:     {x_train_scaled.shape}")
    print(f"x_train_tokens shape:     {x_train_tokens.shape}")

    def _transform_split(
        theta_raw: np.ndarray,
        x_raw: np.ndarray,
        mask_raw: np.ndarray,
    ):
        mask_split = mask_raw.astype(np.float32)
        valid_split = mask_split > 0.5
        valid_counts_split = np.sum(valid_split, axis=0)
        valid_sums_split = np.sum(x_raw * valid_split, axis=0)
        col_means_split = valid_sums_split / np.maximum(valid_counts_split, 1)
        x_split_filled = np.where(valid_split, x_raw, col_means_split[None, :])

        theta_scaled_split = theta_scaler.transform(theta_raw).astype(np.float32)
        x_scaled_split = x_scaler.transform(x_split_filled).astype(np.float32)
        x_scaled_split = x_scaled_split * mask_split
        x_tokens_split = np.stack([x_scaled_split, mask_split], axis=-1).astype(np.float32)
        return theta_scaled_split, x_tokens_split

    theta_val, x_val = _transform_split(theta_val_raw, x_val_raw, mask_val_raw)
    theta_test, x_test = _transform_split(theta_test_raw, x_test_raw, mask_test_raw)

    # Convex hull prior built from scaled theta across all splits
    params_all_scaled = np.concatenate((theta_train, theta_val, theta_test), axis=0)
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
    plot_tarp_ecp_curve(output_dir, alpha_grid, ecp)

    with h5py.File(output_dir / "posterior_samples.h5", "w") as f:
        f.create_dataset("posterior_samples", data=np.array(posterior_samples))
        f.create_dataset("theta_true", data=np.array(theta_true))
        f.create_dataset("x_obs", data=np.array(x_obs))


if __name__ == "__main__":
    main()

