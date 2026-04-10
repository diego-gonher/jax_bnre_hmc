from __future__ import annotations

from pathlib import Path

import hydra
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import h5py
from omegaconf import DictConfig, OmegaConf
import corner

from sklearn.preprocessing import MinMaxScaler

from jax_bnre_hmc.checkpointing import load_best_params, resolve_run_train_config_path
from jax_bnre_hmc.datasets import load_hdf5_dataset
from jax_bnre_hmc.diagnostics import (
    check_sbc,
    run_sbc_from_samples,
    run_tarp_jax,
    l2_distance,
)
from jax_bnre_hmc.hmc import (
    BoxPrior,
    aggregate_nuts_divergences,
    make_log_ratio_fn,
    make_potential_fn,
    run_nuts,
    write_hmc_summary,
    z_to_theta,
)
from jax_bnre_hmc.model import RatioEstimatorMLP
from jax_bnre_hmc.plotting import plot_sbc_rank_histograms, plot_tarp_ecp_curve
from jax_bnre_hmc.plot_style import apply_plot_style


numpyro.set_host_device_count(4)


@hydra.main(config_path="../../configs/sinusoid", config_name="hmc", version_base="1.3")
def main(cfg: DictConfig):
    apply_plot_style()
    run_dir = Path(cfg.run_dir).resolve()
    output_dir = Path(cfg.output_dir).resolve() if cfg.output_dir else run_dir / "hmc_results"
    try:
        dataset_file = cfg.data.get("dataset_file")
        if dataset_file is None:
            raise ValueError(
                "data.dataset_file must be set for sinusoid HMC "
                "(path to HDF5 with 'theta_train', 'x_train', 'theta_val', 'x_val', 'theta_test', 'x_test')"
            )
        dataset_file = str(dataset_file)

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
        loaded = load_hdf5_dataset(dataset_file, validate=True)
        splits = loaded.splits

        theta_train_raw = splits.theta_train
        x_train_raw = splits.x_train
        theta_test_raw = splits.theta_test
        x_test_raw = splits.x_test

        # Same scaling as train.py: MinMaxScaler(-1, 1) fit on train only, then transform test
        theta_scaler = MinMaxScaler(feature_range=(-1, 1))
        x_scaler = MinMaxScaler(feature_range=(-1, 1))
        theta_scaler.fit(theta_train_raw)
        x_scaler.fit(x_train_raw)

        theta_test = np.asarray(theta_scaler.transform(theta_test_raw), dtype=np.float64)
        x_test = np.asarray(x_scaler.transform(x_test_raw), dtype=np.float64)

        print(f"theta_test shape (scaled): {theta_test.shape}")
        print(f"x_test shape (scaled):     {x_test.shape}")

        n_obs = int(cfg.n_observations)
        seed = int(cfg.seed)
        if n_obs > theta_test.shape[0]:
            raise ValueError(
                f"Requested n_observations={n_obs} but test split only has {theta_test.shape[0]} samples."
            )

        # Select a deterministic subset of the test split
        rng = np.random.default_rng(seed)
        indices = rng.choice(theta_test.shape[0], size=n_obs, replace=False)
        theta_true = theta_test[indices]
        theta_true_unscaled = theta_test_raw[indices]
        x_obs = x_test[indices]

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
            x_obs_i = x_obs[i].squeeze()
            log_ratio = make_log_ratio_fn(model.apply, params, x_obs_i)
            potential = make_potential_fn(log_ratio, prior)
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
            mcmc_runs.append(mcmc)
            z_samples = mcmc.get_samples(group_by_chain=False)
            theta_samples, _ = jax.vmap(lambda z: z_to_theta(z, prior))(z_samples)
            posteriors_list.append(theta_samples)
            print(theta_samples.shape)
            print(mcmc.print_summary())

        posterior_samples = jnp.stack(posteriors_list, axis=1)
        posterior_samples = jnp.transpose(posterior_samples, (1, 2, 0))
        posterior_samples_unscaled = np.stack(
            [
                theta_scaler.inverse_transform(
                    np.asarray(posterior_samples[i].T, dtype=np.float64)
                )
                for i in range(n_obs)
            ],
            axis=0,
        )
        print("All done.")
        print("Posterior samples shape:", posterior_samples.shape)

        # Use parameter labels from metadata if available
        theta_names = None
        if loaded.metadata is not None and loaded.metadata.theta_names:
            theta_names = [j.replace(',', '') for j in loaded.metadata.theta_names.split(' ')]

        n_plots = min(int(cfg.n_plots), n_obs)

        corner_labels = theta_names

        for idx in range(n_plots):
            samples = np.asarray(posterior_samples_unscaled[idx], dtype=np.float64)
            true_params = np.asarray(theta_true_unscaled[idx], dtype=np.float64)
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

        posterior_samples_scaled = jnp.transpose(posterior_samples, (2, 0, 1))
        posterior_samples_unscaled_snd = np.transpose(posterior_samples_unscaled, (1, 0, 2))
        key = jax.random.PRNGKey(42)
        key, subkey = jax.random.split(key)
        references = jax.random.uniform(
            subkey,
            shape=(n_obs, len(cfg.prior.low)),
            minval=jnp.array(cfg.prior.low, dtype=jnp.float64),
            maxval=jnp.array(cfg.prior.high, dtype=jnp.float64),
        )
        ecp, alpha_grid = run_tarp_jax(
            posterior_samples=posterior_samples_scaled,
            thetas=theta_true,
            references=references,
            distance=l2_distance,
            num_bins=30,
            z_score_theta=True,
            eps=1e-10,
        )
        alpha_np = np.asarray(alpha_grid)
        ecp_np = np.asarray(ecp)
        tarp_mae = float(np.mean(np.abs(ecp_np - alpha_np)))
        tarp_iae = float(np.trapezoid(np.abs(ecp_np - alpha_np), alpha_np))
        plot_tarp_ecp_curve(output_dir, alpha_grid, ecp)

        with h5py.File(output_dir / "posterior_samples.h5", "w") as f:
            f.create_dataset("posterior_samples", data=posterior_samples_unscaled_snd)
            f.create_dataset("theta_true", data=np.asarray(theta_true_unscaled, dtype=np.float64))
            f.create_dataset("x_obs", data=np.array(x_obs))

        total_div = aggregate_nuts_divergences(mcmc_runs)

        sbc_key = jax.random.PRNGKey(seed)
        ranks, dap_samples = run_sbc_from_samples(
            np.asarray(theta_true, dtype=np.float64),
            np.asarray(posterior_samples_scaled, dtype=np.float64),
            reduce_fns="marginals",
            rng_key=sbc_key,
        )
        sbc_num_samples = int(np.asarray(posterior_samples_scaled).shape[0])
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
