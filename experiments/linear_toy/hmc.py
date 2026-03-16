from __future__ import annotations

import os
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

from jax_bnre_hmc.checkpointing import load_best_params
from jax_bnre_hmc.hmc import BoxPrior, make_log_ratio_fn, make_potential_fn, run_nuts, z_to_theta
from jax_bnre_hmc.model import RatioEstimatorMLP
from jax_bnre_hmc.diagnostics import run_tarp_jax, l2_distance

numpyro.set_host_device_count(4)


def simulate_linear_dataset(
    key: jax.Array,
    n: int,
    n_points: int,
    sigma: float,
    m_low: float,
    m_high: float,
    b_low: float,
    b_high: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Simulator: y = m * x + b + Normal(0, sigma). Returns theta (n, 2), x (n, n_points)."""
    key_m, key_b, key_noise = jax.random.split(key, 3)
    m = jax.random.uniform(key_m, (n,), minval=m_low, maxval=m_high)
    b = jax.random.uniform(key_b, (n,), minval=b_low, maxval=b_high)
    theta = jnp.stack([m, b], axis=-1)
    x_grid = jnp.arange(n_points, dtype=jnp.float32)
    y_clean = m[:, None] * x_grid[None, :] + b[:, None]
    noise = sigma * jax.random.normal(key_noise, (n, n_points))
    y_noisy = y_clean + noise
    return theta.astype(jnp.float32), y_noisy.astype(jnp.float32)


@hydra.main(config_path="../../configs/linear_toy", config_name="hmc", version_base="1.3")
def main(cfg: DictConfig):
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

    key = jax.random.PRNGKey(int(cfg.seed))
    sp = cfg.get("simulator_prior", {})
    theta_true, x_obs = simulate_linear_dataset(
        key=key,
        n=int(cfg.n_observations),
        n_points=int(cfg.data.get("n_points", 10)),
        sigma=float(cfg.data.get("sigma", 0.1)),
        m_low=float(sp.get("m_low", 0.0)),
        m_high=float(sp.get("m_high", 1.0)),
        b_low=float(sp.get("b_low", 0.0)),
        b_high=float(sp.get("b_high", 1.0)),
    )

    prior = BoxPrior(
        low=jnp.array(cfg.prior.low, dtype=jnp.float32),
        high=jnp.array(cfg.prior.high, dtype=jnp.float32),
    )

    n_obs = int(cfg.n_observations)
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

    with h5py.File(output_dir / "posterior_samples.h5", "w") as f:
        f.create_dataset("posterior_samples", data=np.array(posterior_samples))
        f.create_dataset("theta_true", data=np.array(theta_true))
        f.create_dataset("x_obs", data=np.array(x_obs))

    n_plots = min(int(cfg.n_plots), n_obs)
    rng = np.random.default_rng(1234)
    selected_indices = rng.choice(n_obs, size=n_plots, replace=False)
    corner_labels = list(cfg.get("corner_labels", ["m", "b"]))

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


if __name__ == "__main__":
    main()
