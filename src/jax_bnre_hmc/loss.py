from __future__ import annotations

import jax
import jax.numpy as jnp

from .data import Batch


def nre_loss_from_logits(logits_joint: jnp.ndarray, logits_marginal: jnp.ndarray) -> jnp.ndarray:
    """
    Label-free NRE objective with joint labeled 1 and marginal labeled 0:

      L = E_joint[softplus(-logit)] + E_marg[softplus(logit)]
    """
    return jnp.mean(jax.nn.softplus(-logits_joint)) + jnp.mean(jax.nn.softplus(logits_marginal))


def nre_loss(apply_fn, params, joint: Batch, marginal: Batch) -> jnp.ndarray:
    """
    NRE loss function to be applied to a given forward pass from a model (estimator).

    """
    logits_joint = apply_fn(params, joint.theta, joint.x)
    logits_marg = apply_fn(params, marginal.theta, marginal.x)
    return nre_loss_from_logits(logits_joint, logits_marg)
