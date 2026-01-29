from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp


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
