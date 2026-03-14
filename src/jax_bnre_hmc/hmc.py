# src/jax_bnre_hmc/hmc.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS


Array = jax.Array


@dataclass(frozen=True)
class BoxPrior:
    low: jnp.ndarray   # (D,)
    high: jnp.ndarray  # (D,)

    def __post_init__(self):
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have same shape")
        if not jnp.all(self.high > self.low):
            raise ValueError("All high must be > low")


def z_to_theta(z: Array, prior: BoxPrior) -> tuple[Array, Array]:
    """Unconstrained -> box via sigmoid."""
    u = jax.nn.sigmoid(z)                       # (D,) in (0,1)
    theta = prior.low + (prior.high - prior.low) * u
    return theta, u


def logabsdet_dtheta_dz(u: Array, prior: BoxPrior, eps: float = 1e-12) -> Array:
    """log |det dtheta/dz| for theta = low + (high-low)*sigmoid(z).

    Uses u = sigmoid(z). Adds eps for numerical safety.
    """
    # dtheta/dz = (high-low) * u * (1-u)
    return jnp.sum(
        jnp.log(prior.high - prior.low)
        + jnp.log(jnp.clip(u, eps, 1.0))
        + jnp.log(jnp.clip(1.0 - u, eps, 1.0))
    )


def make_log_ratio_fn(
    apply_fn: Callable,
    params,
    x_obs: Array,
) -> Callable[[Array], Array]:
    """Create a compiled log-ratio function for a fixed observation.

    Given a trained neural ratio estimator and a single observation x_obs,
    this function returns a callable:

        log_ratio(theta) -> scalar

    that evaluates the learned log density ratio:

        log r(theta, x_obs)

    where r(theta, x) approximates p(x | theta) / p(x).

    The returned function is JIT-compiled with JAX so that repeated
    evaluations during HMC are efficient.

    Args:
        apply_fn:
            Flax model apply function with signature

                apply_fn(params, theta_batch, x_batch) -> logits

            where logits has shape (B,) and represents the estimated
            log-ratio for each batch element.

        params:
            Trained model parameters (Flax PyTree).

        x_obs:
            A single observed dataset. Shape depends on the model:

            - MLP ratio estimator: (x_dim,)
            - Transformer ratio estimator: (N_tokens, token_dim)

    Returns:
        A function

            log_ratio(theta: Array) -> Array

        that takes a single parameter vector of shape (D,) and returns a
        scalar log-ratio value.

    Notes:
        - Internally the function reshapes inputs to batch size 1 because
          the neural ratio estimator expects batched inputs.
        - The function is JIT-compiled so the first call incurs a compilation
          cost, but subsequent calls (e.g. during NUTS) are fast.
    """
    x_obs = jnp.asarray(x_obs, dtype=jnp.float32)

    @jax.jit
    def log_ratio(theta: Array) -> Array:
        theta = jnp.asarray(theta, dtype=jnp.float32)

        # Add batch dimension expected by the neural network
        logits = apply_fn(params, theta[None, :], x_obs[None, ...])

        # Remove batch dimension -> scalar
        return jnp.squeeze(logits, axis=0)

    return log_ratio


def make_potential_fn(
    log_ratio_fn: Callable[[Array], Array],
    prior: BoxPrior,
    x_obs: Array | None = None,   # kept for signature symmetry; not used if baked into log_ratio_fn
) -> Callable[[Array], Array]:
    """Create a compiled potential function for NUTS in unconstrained space.

    This function constructs the scalar potential energy function used by
    Hamiltonian Monte Carlo. The sampler operates in an unconstrained
    parameter space `z`, which is mapped to the bounded parameter space
    `theta` defined by the prior.

    Specifically:

        z  --sigmoid-->  u in (0, 1)
        u  --affine-->   theta in [low, high]

    The potential energy corresponds to the negative log-posterior (up to
    a constant):

        U(z) = - log r(theta, x_obs) - log |det dtheta/dz|

    where:
        - log r(theta, x_obs) is the learned log density ratio
        - the Jacobian term accounts for the change of variables from z to theta

    The returned function is JIT-compiled so that repeated evaluations
    during NUTS are efficient.

    Args:
        log_ratio_fn:
            Function returning the scalar log density ratio

                log_ratio_fn(theta) -> log r(theta, x_obs)

            typically created by `make_log_ratio_fn`.

        prior:
            BoxPrior defining the lower and upper bounds of the parameter
            space in which theta lives.

        x_obs:
            Unused argument kept for API symmetry with earlier versions
            where the observation was passed directly to the potential.
            The observation is now captured inside `log_ratio_fn`.

    Returns:
        A function

            potential(z: Array) -> Array

        that computes the scalar potential energy for a given unconstrained
        parameter vector `z`. This function is suitable for use with

            numpyro.infer.NUTS(potential_fn=potential)

    Notes:
        - The transformation `z -> theta` ensures that HMC operates in an
          unconstrained space while respecting the box prior bounds.
        - The log-determinant Jacobian term ensures the posterior density
          is correctly adjusted under this transformation.
        - The function is JIT-compiled, so the first call incurs compilation
          overhead but subsequent evaluations during NUTS are fast.
    """

    @jax.jit
    def potential(z: Array) -> Array:
        theta, u = z_to_theta(z, prior)
        ladj = logabsdet_dtheta_dz(u, prior)
        lr = log_ratio_fn(theta)
        return -(lr + ladj)

    return potential


def run_nuts(
    potential_fn: Callable[[Array], Array],
    rng_key: Array,
    init_z: Array,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    **nuts_kwargs,
):
    """Run NUTS with a custom potential_fn."""
    kernel = NUTS(potential_fn=potential_fn, **nuts_kwargs)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)
    mcmc.run(rng_key, init_params=init_z)
    return mcmc
