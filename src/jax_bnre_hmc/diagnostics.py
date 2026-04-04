from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def l2_distance(x: jnp.ndarray, y: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Broadcastable L2 distance.
    x: (..., D)
    y: (..., D)
    returns: (...) distances
    """
    return jnp.sqrt(jnp.sum((x - y) ** 2, axis=axis))


def run_tarp_jax(
    posterior_samples: jnp.ndarray,
    thetas: jnp.ndarray,
    references: jnp.ndarray,
    distance: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray] = l2_distance,
    num_bins: Optional[int] = 30,
    z_score_theta: bool = False,
    eps: float = 1e-10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JAX implementation of sbi's _run_tarp.

    Args:
        posterior_samples: (S, N, D) posterior samples for each observation.
        thetas:           (N, D) true parameters.
        references:       (N, D) reference points (one per observation).
        distance:         broadcastable distance fn. Should support shapes:
                          distance((N,D),(S,N,D)) -> (S,N)
                          distance((N,D),(N,D))   -> (N,)
        num_bins:         histogram bins for coverage values (in [0,1]).
        z_score_theta:    if True, min-max normalize theta and samples using
                          lo/hi computed from thetas (like sbi).
        eps:              numerical stability constant.

    Returns:
        ecp:       (num_bins+1,) expected coverage probability curve.
        alpha_grid:(num_bins+1,) bin edges in [0,1] (like torch.histogram)
    """
    if posterior_samples.ndim != 3:
        raise ValueError(f"posterior_samples must have shape (S,N,D), got {posterior_samples.shape}")
    if thetas.ndim != 2:
        raise ValueError(f"thetas must have shape (N,D), got {thetas.shape}")
    if references.shape != thetas.shape:
        raise ValueError(f"references must have same shape as thetas. got {references.shape} vs {thetas.shape}")

    S, N, D = posterior_samples.shape
    if thetas.shape != (N, D):
        raise ValueError(f"thetas must have shape (N,D) matching posterior_samples. got {thetas.shape} vs {(N,D)}")

    if num_bins is None:
        num_bins = max(1, N // 10)

    # Optional min-max normalization (matches sbi's behavior, despite the name z_score_theta)
    if z_score_theta:
        lo = jnp.min(thetas, axis=0, keepdims=True)  # (1,D)
        hi = jnp.max(thetas, axis=0, keepdims=True)  # (1,D)
        scale = (hi - lo) + eps
        posterior_samples = (posterior_samples - lo) / scale
        thetas = (thetas - lo) / scale
        references = (references - lo) / scale

    # Distances:
    # sample_dists: (S,N) = dist(ref_i, sample_{s,i})
    # theta_dists:  (N,)  = dist(ref_i, theta_i)
    sample_dists = distance(references, posterior_samples)  # expect (S,N)
    theta_dists = distance(references, thetas)              # expect (N,)

    # Ensure shapes are as expected (helpful when swapping distance fns)
    if sample_dists.shape != (S, N):
        raise ValueError(f"distance(references, posterior_samples) must return (S,N). got {sample_dists.shape}")
    if theta_dists.shape != (N,):
        raise ValueError(f"distance(references, thetas) must return (N,). got {theta_dists.shape}")

    # Coverage values per observation i:
    # fraction of posterior samples closer to ref than true theta is.
    coverage_values = jnp.mean(sample_dists < theta_dists[None, :], axis=0)  # (N,)

    # Histogram over coverage values in [0,1]
    # Use fixed range to avoid edge weirdness and to match the idea that coverage is in [0,1].
    hist, alpha_grid = jnp.histogram(coverage_values, bins=num_bins, range=(0.0, 1.0), density=True)

    # Empirical CDF (ECP): cumulative integral over histogram bins
    # With density=True, hist integrates to 1 over the range, but numerical issues can happen.
    ecp = jnp.cumsum(hist) / (jnp.sum(hist) + eps)  # (num_bins,)
    ecp = jnp.concatenate([jnp.zeros((1,), dtype=ecp.dtype), ecp], axis=0)  # (num_bins+1,)

    return ecp, alpha_grid


# --- Simulation-based calibration (SBC) ---------------------------------------


ReduceFn = Callable[[np.ndarray], np.ndarray]


def _prepare_sbc_reduce_fns(
    reduce_fns: Union[str, ReduceFn, List[ReduceFn]],
    d: int,
) -> List[ReduceFn]:
    if isinstance(reduce_fns, str):
        if reduce_fns != "marginals":
            raise ValueError(
                'reduce_fns as str must be "marginals"; otherwise pass a callable or list of callables.'
            )
        return [lambda theta, j=j: np.asarray(theta[:, j], dtype=np.float64).ravel() for j in range(d)]
    if callable(reduce_fns):
        return [reduce_fns]
    if isinstance(reduce_fns, list):
        if not reduce_fns:
            raise ValueError("reduce_fns list must be non-empty.")
        return reduce_fns
    raise TypeError(
        "reduce_fns must be 'marginals', a callable, or a list of callables; "
        f"got {type(reduce_fns)!r}."
    )


def run_sbc_from_samples(
    theta_true: np.ndarray,
    posterior_samples: np.ndarray,
    reduce_fns: Union[str, ReduceFn, List[ReduceFn]] = "marginals",
    rng_key: jax.Array | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulation-based calibration (SBC) ranks from posterior samples (no sbi / torch).

    For each observation ``i`` and each 1D reduction, the rank is the count of posterior
    draws whose reduced value is **strictly less** than the reduced true parameter
    (same convention as Talts et al. / sbi when marginals use ``theta[:, j]``).

    Args:
        theta_true: Ground-truth parameters, shape ``(N, D)``.
        posterior_samples: Posterior samples, shape ``(S, N, D)``.
        reduce_fns: ``"marginals"`` (one function per parameter dimension),
            a single ``callable(theta_batch) -> 1d``, or a list of such callables.
        rng_key: Optional JAX PRNG key for reproducible DAP (data-averaged posterior)
            row sampling. If provided, indices are drawn with ``jax.random.randint`` and
            DAP is deterministic for a fixed key. If ``None``, indices use
            ``numpy.random.randint`` (non-deterministic unless NumPy is seeded elsewhere).

    Returns:
        ranks: Integer array of shape ``(N, K)`` where ``K`` is the number of reductions.
        dap_samples: Data-averaged posterior draw per row, shape ``(N, D)``, each row
            one sample drawn uniformly at random from ``posterior_samples[:, i, :]``.
    """
    theta_true = np.asarray(theta_true, dtype=np.float64)
    posterior_samples = np.asarray(posterior_samples, dtype=np.float64)

    if theta_true.ndim != 2:
        raise ValueError(f"theta_true must have shape (N, D), got {theta_true.shape}")
    if posterior_samples.ndim != 3:
        raise ValueError(
            f"posterior_samples must have shape (S, N, D), got {posterior_samples.shape}"
        )

    n, d = theta_true.shape
    s, n_ps, d_ps = posterior_samples.shape
    if n_ps != n or d_ps != d:
        raise ValueError(
            f"posterior_samples (S, N, D) must match theta_true (N, D); "
            f"got N={n_ps}, D={d_ps} vs N={n}, D={d}."
        )

    fns = _prepare_sbc_reduce_fns(reduce_fns, d)
    k = len(fns)
    ranks = np.empty((n, k), dtype=np.int64)

    for i in range(n):
        post_i = posterior_samples[:, i, :]  # (S, D)
        true_i = theta_true[i : i + 1, :]  # (1, D)
        for j, rf in enumerate(fns):
            reduced_samples = np.asarray(rf(post_i), dtype=np.float64).ravel()
            reduced_true = np.asarray(rf(true_i), dtype=np.float64).ravel()
            if reduced_samples.shape != (s,):
                raise ValueError(
                    f"reduce_fn must map (S, D) to length S; got shape {reduced_samples.shape} for S={s}."
                )
            if reduced_true.size != 1:
                raise ValueError(
                    f"reduce_fn must map (1, D) to a scalar; got shape {reduced_true.shape}."
                )
            ranks[i, j] = int(np.sum(reduced_samples < reduced_true))

    row_ix = np.arange(n, dtype=np.int64)
    if rng_key is not None:
        pick = jr.randint(rng_key, (n,), 0, s, dtype=jnp.int64)
        pick = np.asarray(pick, dtype=np.int64)
    else:
        pick = np.random.randint(0, s, size=n, dtype=np.int64)
    dap_samples = np.asarray(posterior_samples)[pick, row_ix, :]

    return ranks, dap_samples


def check_uniformity_frequentist(
    ranks: np.ndarray,
    num_posterior_samples: int,
) -> np.ndarray:
    """Kolmogorov–Smirnov test of each rank column vs ``Uniform(0, num_posterior_samples)``.

    Uses the same continuous null as sbi: CDF ``x / num_posterior_samples`` on ``[0, M]``
    with ``M = num_posterior_samples``.

    Args:
        ranks: Shape ``(N, K)``.
        num_posterior_samples: ``S`` (number of posterior draws used in ranking).

    Returns:
        p-values, shape ``(K,)``.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.ndim != 2:
        raise ValueError(f"ranks must be 2D (N, K), got {ranks.shape}")
    m = int(num_posterior_samples)
    if m <= 0:
        raise ValueError("num_posterior_samples must be positive.")

    u = stats.uniform(loc=0.0, scale=float(m))
    pvals = np.empty(ranks.shape[1], dtype=np.float64)
    for j in range(ranks.shape[1]):
        col = ranks[:, j]
        _, pvals[j] = stats.kstest(col, u.cdf)
    return pvals


def check_sbc(
    ranks: np.ndarray,
    prior_samples: np.ndarray,
    dap_samples: np.ndarray,
    num_posterior_samples: int,
) -> dict[str, np.ndarray]:
    """Frequentist SBC checks (KS uniformity of ranks only; no C2ST yet).

    ``prior_samples`` and ``dap_samples`` are part of the API for future checks
    (e.g. prior vs DAP); currently only rank uniformity is computed.

    Args:
        ranks: ``(N, K)`` SBC ranks.
        prior_samples: ``(N, D)`` prior draws (same ``N`` as ranks).
        dap_samples: ``(N, D)`` data-averaged posterior samples.
        num_posterior_samples: ``S`` used in ``run_sbc_from_samples``.

    Returns:
        ``{"ks_pvals": array of shape (K,)}``.
    """
    ranks = np.asarray(ranks)
    prior_samples = np.asarray(prior_samples)
    dap_samples = np.asarray(dap_samples)
    if ranks.ndim != 2:
        raise ValueError(f"ranks must be (N, K), got {ranks.shape}")
    n = ranks.shape[0]
    if prior_samples.shape[0] != n or dap_samples.shape[0] != n:
        raise ValueError(
            "prior_samples and dap_samples must have the same number of rows N as ranks."
        )
    if prior_samples.ndim != 2 or dap_samples.ndim != 2:
        raise ValueError("prior_samples and dap_samples must be 2D (N, D).")
    if prior_samples.shape != dap_samples.shape:
        raise ValueError(
            f"prior_samples and dap_samples must have the same shape; got "
            f"{prior_samples.shape} vs {dap_samples.shape}."
        )

    return {
        "ks_pvals": check_uniformity_frequentist(ranks, num_posterior_samples),
    }


def plot_sbc_rank_histograms(
    ranks: np.ndarray,
    num_posterior_samples: int,
    labels: Optional[List[str]] = None,
    ks_pvals: Optional[np.ndarray] = None,
    bins: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[plt.Figure, np.ndarray]:
    """Histogram of SBC ranks per dimension (marginals), with uniform reference line.

    Args:
        ranks: Shape ``(N, D)`` (one column per marginal / reduction).
        num_posterior_samples: ``S`` (same as in ``run_sbc_from_samples``); x-axis is ``[0, S]``.
        labels: Optional length-``D`` names for subplot titles.
        ks_pvals: Optional length-``D`` p-values shown in titles.
        bins: Number of histogram bins; default ``min(30, S + 1)``.
        figsize: Figure size; default scales with ``D``.
        output_path: If set, ``savefig`` to this path.

    Returns:
        ``(fig, axes)`` with ``axes`` shape ``(D,)`` (flattened row of subplots).
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    if ranks.ndim != 2:
        raise ValueError(f"ranks must have shape (N, D), got {ranks.shape}")
    n, d = ranks.shape
    s = int(num_posterior_samples)
    if s <= 0:
        raise ValueError("num_posterior_samples must be positive.")

    n_bins = int(bins) if bins is not None else min(30, s + 1)
    n_bins = max(1, n_bins)
    expected_count = n / n_bins

    if figsize is None:
        figsize = (max(3.2 * d, 3.5), 3.2)

    fig, axes_2d = plt.subplots(1, d, figsize=figsize, squeeze=False, sharey=True)
    axes = axes_2d.ravel()

    for j in range(d):
        ax = axes[j]
        ax.hist(
            ranks[:, j],
            bins=n_bins,
            range=(0.0, float(s)),
            color="steelblue",
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(expected_count, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlim(0.0, float(s))
        ax.set_xlabel("rank")
        if j == 0:
            ax.set_ylabel("count")

        title_parts: List[str] = []
        if labels is not None and j < len(labels):
            title_parts.append(str(labels[j]))
        else:
            title_parts.append(f"dim {j}")
        if ks_pvals is not None and j < len(ks_pvals):
            title_parts.append(f"KS p={float(ks_pvals[j]):.3g}")
        ax.set_title("\n".join(title_parts), fontsize=10)

    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig, axes


if __name__ == "__main__":
    _rng = np.random.default_rng(0)
    _N, _D, _S = 50, 3, 200
    _theta = _rng.normal(size=(_N, _D))
    _post = _rng.normal(size=(_S, _N, _D)) + _theta[np.newaxis, :, :]
    _ranks, _dap = run_sbc_from_samples(_theta, _post, reduce_fns="marginals")
    assert _ranks.shape == (_N, _D)
    assert _dap.shape == (_N, _D)
    _p = check_uniformity_frequentist(_ranks, _S)
    assert _p.shape == (_D,)
    _out = check_sbc(_ranks, _theta, _dap, _S)
    assert _out["ks_pvals"].shape == (_D,)
    _key = jax.random.PRNGKey(7)
    _, _dap_a = run_sbc_from_samples(_theta, _post, rng_key=_key)
    _, _dap_b = run_sbc_from_samples(_theta, _post, rng_key=_key)
    np.testing.assert_array_equal(_dap_a, _dap_b)
