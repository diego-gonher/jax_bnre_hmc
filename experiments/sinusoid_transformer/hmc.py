from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import h5py
import numpyro
import matplotlib.pyplot as plt
import corner
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import MinMaxScaler

from jax_bnre_hmc.checkpointing import load_best_params, resolve_run_train_config_path
from jax_bnre_hmc.datasets import load_hdf5_dataset
from jax_bnre_hmc.hmc import (
    BoxPrior,
    aggregate_nuts_divergences,
    make_log_ratio_fn,
    make_potential_fn,
    run_nuts,
    write_hmc_summary,
    z_to_theta,
)
from jax_bnre_hmc.model import RatioEstimatorTransformer
from jax_bnre_hmc.diagnostics import (
    check_sbc,
    plot_sbc_rank_histograms,
    run_sbc_from_samples,
    run_tarp_jax,
    l2_distance,
)

numpyro.set_host_device_count(4)


@hydra.main(config_path="../../configs/sinusoid_transformer", config_name="hmc", version_base="1.3")
def main(cfg: DictConfig):
    run_dir = Path(cfg.run_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else run_dir / "hmc_results"
    try:
        dataset_file = cfg.data.get("dataset_file")
        if dataset_file is None:
            raise ValueError(
                "data.dataset_file must be set for sinusoid_transformer HMC "
                "(path to HDF5 with theta_train, x_train, x_train_mask; theta_test, x_test, x_test_mask; etc.)"
            )
        dataset_file = str(dataset_file)

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

        print(f"\nLoading dataset from {dataset_file}")
        loaded = load_hdf5_dataset(dataset_file, validate=True)
        splits = loaded.splits

        theta_train_raw = splits.theta_train
        x_train_raw = splits.x_train
        mask_train_raw = splits.mask_train
        theta_test_raw = splits.theta_test
        x_test_raw = splits.x_test
        mask_test_raw = splits.mask_test

        if mask_train_raw is None or mask_test_raw is None:
            raise ValueError(
                "Sinusoid transformer HMC requires x_train_mask and x_test_mask in the HDF5 file."
            )

        # Fit scalers on train (same logic as train.py)
        mask_train = mask_train_raw.astype(np.float64)
        valid_train = mask_train > 0.5
        valid_counts_train = np.sum(valid_train, axis=0)
        valid_sums_train = np.sum(x_train_raw * valid_train, axis=0)
        col_means_train = valid_sums_train / np.maximum(valid_counts_train, 1)
        y_train_filled = np.where(valid_train, x_train_raw, col_means_train[None, :])

        theta_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
        theta_scaler.fit_transform(theta_train_raw)
        y_scaler.fit_transform(y_train_filled)

        # Transform test split
        mask_test = mask_test_raw.astype(np.float64)
        valid_test = mask_test > 0.5
        valid_counts_test = np.sum(valid_test, axis=0)
        valid_sums_test = np.sum(x_test_raw * valid_test, axis=0)
        col_means_test = valid_sums_test / np.maximum(valid_counts_test, 1)
        y_test_filled = np.where(valid_test, x_test_raw, col_means_test[None, :])

        theta_test_scaled = theta_scaler.transform(theta_test_raw).astype(np.float64)
        y_test_scaled = y_scaler.transform(y_test_filled).astype(np.float64)
        y_test_scaled = y_test_scaled * mask_test
        x_test_tokens = np.stack([y_test_scaled, mask_test], axis=-1).astype(np.float64)

        n_obs = int(cfg.n_observations)
        seed = int(cfg.seed)
        rng = np.random.default_rng(seed)
        n_test = x_test_tokens.shape[0]
        if n_obs > n_test:
            raise ValueError(f"n_observations ({n_obs}) must be <= test set size ({n_test}).")
        indices = rng.choice(n_test, size=n_obs, replace=False)
        theta_true = theta_test_scaled[indices]
        x_obs = x_test_tokens[indices]

        prior = BoxPrior(
            low=jnp.array(cfg.prior.low, dtype=jnp.float64),
            high=jnp.array(cfg.prior.high, dtype=jnp.float64),
        )

        num_chains = int(cfg.num_chains)
        posteriors_list = []
        mcmc_runs = []
        print("Starting inference")
        for i in range(n_obs):
            print(f"\nRunning observation {i+1}/{n_obs}")
            x_obs_i = jnp.asarray(x_obs[i], dtype=jnp.float64)
            assert jnp.sum(x_obs_i[:, 1] > 0.5) > 0, f"Obs {i} has no valid tokens."
            log_ratio = make_log_ratio_fn(model.apply, params, x_obs_i)
            potential = make_potential_fn(log_ratio, prior)
            test_grad = jax.grad(log_ratio)(jnp.asarray(theta_true[i], dtype=jnp.float64))
            if not bool(jnp.all(jnp.isfinite(test_grad))):
                raise ValueError(f"Non-finite log_ratio gradient for observation {i}")
            D = prior.low.shape[0]
            init_z = jnp.zeros((num_chains, D), dtype=jnp.float64)
            mcmc = run_nuts(
                potential_fn=potential,
                rng_key=jax.random.PRNGKey(seed + i),
                init_z=init_z,
                num_warmup=int(cfg.num_warmup),
                num_samples=int(cfg.num_samples),
                num_chains=num_chains,
            )
            mcmc_runs.append(mcmc)
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
            minval=jnp.array(cfg.prior.low, dtype=jnp.float64),
            maxval=jnp.array(cfg.prior.high, dtype=jnp.float64),
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
        alpha_np = np.asarray(alpha_grid)
        ecp_np = np.asarray(ecp)
        tarp_mae = float(np.mean(np.abs(ecp_np - alpha_np)))
        tarp_iae = float(np.trapz(np.abs(ecp_np - alpha_np), alpha_np))
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
        print(f"Saved posterior samples to {output_dir / 'posterior_samples.h5'}")

        total_div = aggregate_nuts_divergences(mcmc_runs)

        sbc_key = jax.random.PRNGKey(seed)
        ranks, dap_samples = run_sbc_from_samples(
            np.asarray(theta_true, dtype=np.float64),
            np.asarray(posterior_samples_tarp, dtype=np.float64),
            reduce_fns="marginals",
            rng_key=sbc_key,
        )
        sbc_num_samples = int(np.asarray(posterior_samples_tarp).shape[0])
        prior_key = jax.random.PRNGKey(seed + 1)
        prior_samples = jax.random.uniform(
            prior_key,
            shape=np.asarray(theta_true).shape,
            minval=jnp.array(cfg.prior.low, dtype=jnp.float64),
            maxval=jnp.array(cfg.prior.high, dtype=jnp.float64),
        )
        sbc_check = check_sbc(
            ranks,
            np.asarray(prior_samples, dtype=np.float64),
            np.asarray(dap_samples, dtype=np.float64),
            num_posterior_samples=sbc_num_samples,
        )
        ks_pvals = np.asarray(sbc_check["ks_pvals"], dtype=np.float64)
        ks_pval_min = float(np.min(ks_pvals))
        ks_pval_mean = float(np.mean(ks_pvals))
        print("\nSBC results")
        print(f"ks_pvals: {ks_pvals}")
        print(f"ks_pval_min: {ks_pval_min}")
        print(f"ks_pval_mean: {ks_pval_mean}")

        sbc_labels = corner_labels if len(corner_labels) == ranks.shape[1] else None
        fig_sbc, _ = plot_sbc_rank_histograms(
            ranks,
            sbc_num_samples,
            labels=sbc_labels,
            ks_pvals=ks_pvals,
            output_path=output_dir / "sbc_rank_histograms.png",
        )
        plt.close(fig_sbc)

        metrics_lines = [
            "HMC metrics",
            "-----------",
            f"divergences: {total_div}",
            "",
            "SBC metrics",
            "-----------",
            f"ks_pvals: {np.array2string(ks_pvals, separator=', ')}",
            f"ks_pval_min: {ks_pval_min}",
            f"ks_pval_mean: {ks_pval_mean}",
            "",
        ]
        (output_dir / "hmc_metrics.txt").write_text("\n".join(metrics_lines))

        n_obs_f = float(n_obs)
        num_chains_f = float(num_chains)
        div_per_obs = float(total_div) / n_obs_f
        div_per_obs_chain = float(total_div) / (n_obs_f * num_chains_f)
        write_hmc_summary(
            output_dir,
            status="ok",
            divergences_total=total_div,
            divergences_per_observation=div_per_obs,
            divergences_per_observation_per_chain=div_per_obs_chain,
            sbc_ks_pval_min=ks_pval_min,
            sbc_ks_pval_mean=ks_pval_mean,
            tarp_mae=tarp_mae,
            tarp_iae=tarp_iae,
            posterior_samples_path="posterior_samples.h5",
        )
    except Exception as e:
        run_dir = Path(cfg.run_dir).resolve()
        output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else run_dir / "hmc_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        msg = str(e) if str(e) else type(e).__name__
        write_hmc_summary(output_dir, status="error", message=msg)
        raise


if __name__ == "__main__":
    main()
