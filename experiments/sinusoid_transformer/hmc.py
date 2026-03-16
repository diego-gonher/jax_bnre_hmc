from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
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


@hydra.main(config_path="../../configs/sinusoid_transformer", config_name="hmc", version_base="1.3")
def main(cfg: DictConfig):
    dataset_file = cfg.data.get("dataset_file")
    if dataset_file is None:
        raise ValueError(
            "data.dataset_file must be set for sinusoid_transformer HMC (path to HDF5 with 'theta', 'y_obs', 'mask')"
        )
    dataset_file = str(dataset_file)

    run_dir = Path(cfg.run_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else run_dir / "hmc_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = OmegaConf.load(run_dir / "config.yaml")
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

    print(f"\nLoading dataset from {dataset_file}")
    with h5py.File(dataset_file, "r") as f:
        theta = f["theta"][:]
        y_obs = f["y_obs"][:]
        mask = f["mask"][:].astype(np.float32)

    valid = mask > 0.5
    valid_counts = np.sum(valid, axis=0)
    valid_sums = np.sum(y_obs * valid, axis=0)
    col_means = valid_sums / np.maximum(valid_counts, 1)
    y_obs_filled = np.where(valid, y_obs, col_means[None, :])

    theta_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    theta_scaled = theta_scaler.fit_transform(theta).astype(np.float32)
    y_obs_scaled = y_scaler.fit_transform(y_obs_filled).astype(np.float32)
    y_obs_scaled = y_obs_scaled * mask
    x_tokens = np.stack([y_obs_scaled, mask], axis=-1).astype(np.float32)

    n_obs = int(cfg.n_observations)
    seed = int(cfg.seed)
    _, theta_true, _, x_obs = train_test_split(
        theta_scaled, x_tokens, test_size=n_obs, random_state=seed
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
        x_obs_i = jnp.asarray(x_obs[i], dtype=jnp.float32)
        assert jnp.sum(x_obs_i[:, 1] > 0.5) > 0, f"Obs {i} has no valid tokens."
        log_ratio = make_log_ratio_fn(model.apply, params, x_obs_i)
        potential = make_potential_fn(log_ratio, prior)
        test_grad = jax.grad(log_ratio)(jnp.asarray(theta_true[i], dtype=jnp.float32))
        if not bool(jnp.all(jnp.isfinite(test_grad))):
            raise ValueError(f"Non-finite log_ratio gradient for observation {i}")
        D = prior.low.shape[0]
        init_z = jnp.zeros((num_chains, D), dtype=jnp.float32)
        mcmc = run_nuts(
            potential_fn=potential,
            rng_key=jax.random.PRNGKey(seed + i),
            init_z=init_z,
            num_warmup=int(cfg.num_warmup),
            num_samples=int(cfg.num_samples),
            num_chains=num_chains,
        )
        z_samples = mcmc.get_samples(group_by_chain=False)
        theta_samples, _ = jax.vmap(lambda z: z_to_theta(z, prior))(z_samples)
        posteriors_list.append(theta_samples)
        print(theta_samples.shape)
        mcmc.print_summary()

    posterior_samples = jnp.stack(posteriors_list, axis=1)
    posterior_samples = jnp.transpose(posterior_samples, (1, 2, 0))
    print("All done.")
    print("Posterior samples shape:", posterior_samples.shape)

    n_plots = min(int(cfg.n_plots), n_obs)
    rng = np.random.default_rng(1234)
    selected_indices = rng.choice(n_obs, size=n_plots, replace=False)
    corner_labels = list(cfg.get("corner_labels", ["A", "f", "phi", "b"]))
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
        outname = output_dir / f"corner_observation_{idx}.png"
        figure.savefig(outname)
        plt.close(figure)
        print(f"Saved corner plot for observation {idx} as {outname}")

    posterior_samples_tarp = jnp.transpose(posterior_samples, (2, 0, 1))
    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    references = jax.random.uniform(
        subkey,
        shape=(n_obs, len(cfg.prior.low)),
        minval=jnp.array(cfg.prior.low, dtype=jnp.float32),
        maxval=jnp.array(cfg.prior.high, dtype=jnp.float32),
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
    plt.savefig(output_dir / "tarp_ecp_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    with h5py.File(output_dir / "posterior_samples.h5", "w") as f:
        f.create_dataset("posterior_samples", data=np.array(posterior_samples_tarp))
        f.create_dataset("theta_true", data=np.array(theta_true))
        f.create_dataset("x_obs", data=np.array(x_obs))
    print(f"Saved posterior samples to {output_dir}posterior_samples.h5")


if __name__ == "__main__":
    main()
