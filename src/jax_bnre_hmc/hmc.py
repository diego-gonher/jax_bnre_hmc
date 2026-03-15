# src/jax_bnre_hmc/hmc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS

Array = jax.Array


@dataclass(frozen=True)
class BoxPrior:
    low: jnp.ndarray
    high: jnp.ndarray

    def __post_init__(self):
        object.__setattr__(self, "low", jnp.asarray(self.low))
        object.__setattr__(self, "high", jnp.asarray(self.high))
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have same shape")
        if not bool(jnp.all(self.high > self.low)):
            raise ValueError("All high must be > low")


@dataclass(frozen=True)
class ConvexHullPrior:
    low: jnp.ndarray
    high: jnp.ndarray
    equations: jnp.ndarray  # (n_facets, D+1)

    def __post_init__(self):
        object.__setattr__(self, "low", jnp.asarray(self.low))
        object.__setattr__(self, "high", jnp.asarray(self.high))
        object.__setattr__(self, "equations", jnp.asarray(self.equations))

        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have same shape")
        if not bool(jnp.all(self.high > self.low)):
            raise ValueError("All high must be > low")

        d = self.low.shape[0]
        if self.equations.ndim != 2 or self.equations.shape[1] != d + 1:
            raise ValueError("equations must have shape (n_facets, D+1)")


def z_to_theta(z: Array, prior) -> tuple[Array, Array]:
    """Unconstrained -> bounding box via sigmoid."""
    u = jax.nn.sigmoid(z)
    theta = prior.low + (prior.high - prior.low) * u
    return theta, u


def samples_z_to_theta(z_samples: Array, prior) -> Array:
    """Vectorized z -> theta for arrays with shape (..., D)."""
    u = jax.nn.sigmoid(z_samples)
    return prior.low + (prior.high - prior.low) * u


def theta_to_z(theta: Array, prior, eps: float = 1e-6) -> Array:
    """Bounding-box coordinates -> unconstrained z."""
    u = (theta - prior.low) / (prior.high - prior.low)
    u = jnp.clip(u, eps, 1.0 - eps)
    return jnp.log(u) - jnp.log1p(-u)


def logabsdet_dtheta_dz(u: Array, prior, eps: float = 1e-12) -> Array:
    """log |det dtheta/dz| for theta = low + (high-low)*sigmoid(z)."""
    return jnp.sum(
        jnp.log(prior.high - prior.low)
        + jnp.log(jnp.clip(u, eps, 1.0))
        + jnp.log(jnp.clip(1.0 - u, eps, 1.0))
    )


@jax.jit
def point_in_hull(theta: Array, equations: Array, tolerance: float = 1e-12) -> Array:
    vals = equations[:, :-1] @ theta + equations[:, -1]
    return jnp.all(vals <= tolerance)


@jax.jit
def hull_violation(theta: Array, equations: Array) -> Array:
    vals = equations[:, :-1] @ theta + equations[:, -1]
    return jnp.maximum(vals, 0.0)


@jax.jit
def soft_hull_log_prior(
    theta: Array,
    equations: Array,
    barrier_scale: float = 100.0,
    power: float = 2.0,
) -> Array:
    v = hull_violation(theta, equations)
    return -barrier_scale * jnp.sum(v**power)


def make_log_ratio_fn(
    apply_fn: Callable,
    params,
    x_obs: Array,
) -> Callable[[Array], Array]:
    """Return log r(theta, x_obs) for a fixed observation."""
    x_obs = jnp.asarray(x_obs, dtype=jnp.float32)

    @jax.jit
    def log_ratio(theta: Array) -> Array:
        theta = jnp.asarray(theta, dtype=jnp.float32)
        logits = apply_fn(params, theta[None, :], x_obs[None, ...])
        return jnp.squeeze(logits, axis=0)

    return log_ratio


def make_potential_fn(
    log_ratio_fn: Callable[[Array], Array],
    prior,
    tolerance: float = 1e-12,
    soft_hull: bool = False,
    barrier_scale: float = 100.0,
    barrier_power: float = 2.0,
) -> Callable[[Array], Array]:
    """Create NUTS potential in unconstrained z-space."""

    if isinstance(prior, BoxPrior):

        @jax.jit
        def potential(z: Array) -> Array:
            theta, u = z_to_theta(z, prior)
            ladj = logabsdet_dtheta_dz(u, prior)
            lr = log_ratio_fn(theta)
            return -(lr + ladj)

        return potential

    elif isinstance(prior, ConvexHullPrior):

        if soft_hull:
            @jax.jit
            def potential(z: Array) -> Array:
                theta, u = z_to_theta(z, prior)
                ladj = logabsdet_dtheta_dz(u, prior)
                lr = log_ratio_fn(theta)
                lp = soft_hull_log_prior(
                    theta,
                    prior.equations,
                    barrier_scale=barrier_scale,
                    power=barrier_power,
                )
                return -(lr + lp + ladj)

            return potential

        else:
            @jax.jit
            def potential(z: Array) -> Array:
                theta, u = z_to_theta(z, prior)
                ladj = logabsdet_dtheta_dz(u, prior)
                lr = log_ratio_fn(theta)
                inside = point_in_hull(theta, prior.equations, tolerance=tolerance)
                lp = jnp.where(inside, 0.0, -jnp.inf)
                return -(lr + lp + ladj)

            return potential

    else:
        raise TypeError(f"Unsupported prior type: {type(prior)}")


def run_nuts(
    potential_fn: Callable[[Array], Array],
    rng_key: Array,
    init_z: Array,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    chain_method: str = "parallel",
    **nuts_kwargs,
):
    """Run NUTS with a custom potential_fn."""
    kernel = NUTS(potential_fn=potential_fn, **nuts_kwargs)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
    )
    mcmc.run(rng_key, init_params=init_z)
    return mcmc

@jax.jit
def points_in_hull(points: Array, equations: Array, tolerance: float = 1e-12) -> Array:
    """
    Batched convex hull membership test.

    Args:
        points:
            Array of shape (N, D).
        equations:
            Hull facet equations of shape (n_facets, D+1), as returned by
            scipy.spatial.ConvexHull.equations.
        tolerance:
            Numerical tolerance for the facet inequalities.

    Returns:
        Boolean mask of shape (N,), where True means the point is inside
        the convex hull.
    """
    A = equations[:, :-1]   # (n_facets, D)
    b = equations[:, -1]    # (n_facets,)
    vals = points @ A.T + b[None, :]   # (N, n_facets)
    return jnp.all(vals <= tolerance, axis=1)


def sample_uniform_in_convex_hull(
    key: Array,
    prior: ConvexHullPrior,
    n_samples: int,
    batch_size: int = 4096,
    tolerance: float = 1e-12,
) -> Array:
    """
    Sample approximately uniformly from a convex hull using rejection sampling
    from the hull bounding box.

    Args:
        key:
            JAX PRNG key.
        prior:
            ConvexHullPrior defining the bounding box and hull equations.
        n_samples:
            Number of accepted samples to return.
        batch_size:
            Number of proposals drawn per rejection-sampling round.
        tolerance:
            Numerical tolerance for hull membership.

    Returns:
        samples:
            Array of shape (n_samples, D).
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    accepted_chunks = []
    n_collected = 0
    dim = prior.low.shape[0]

    while n_collected < n_samples:
        key, subkey = jax.random.split(key)

        proposals = jax.random.uniform(
            subkey,
            shape=(batch_size, dim),
            minval=prior.low,
            maxval=prior.high,
            dtype=prior.low.dtype,
        )

        mask = points_in_hull(proposals, prior.equations, tolerance=tolerance)
        accepted = proposals[mask]

        if accepted.shape[0] > 0:
            accepted_chunks.append(accepted)
            n_collected += accepted.shape[0]

    return jnp.concatenate(accepted_chunks, axis=0)[:n_samples]
