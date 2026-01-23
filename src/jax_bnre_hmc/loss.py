from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from .data import Batch


def nre_loss_from_logits(logits_joint: jnp.ndarray, logits_marginal: jnp.ndarray) -> jnp.ndarray:
    """
    Label-free NRE objective with joint labeled 1 and marginal labeled 0:

      L = E_joint[softplus(-logit)] + E_marg[softplus(logit)]
    """
    return jnp.mean(jax.nn.softplus(-logits_joint)) + jnp.mean(jax.nn.softplus(logits_marginal))


def nre_loss(apply_fn, params, joint: Batch, marginal: Batch) -> jnp.ndarray:
    """NRE loss applied to a model forward pass."""
    logits_joint = apply_fn(params, joint.theta, joint.x)
    logits_marg = apply_fn(params, marginal.theta, marginal.x)
    return nre_loss_from_logits(logits_joint, logits_marg)


def nre_loss_bce_style_from_logits(logits_joint: jnp.ndarray, logits_marg: jnp.ndarray) -> jnp.ndarray:
    """
    Same objective as nre_loss_from_logits, written as BCE-with-logits over
    concatenated joint+marginal examples. Useful for direct comparison to SBI.
    """
    logits = jnp.concatenate([logits_joint, logits_marg], axis=0)
    labels = jnp.concatenate(
        [jnp.ones_like(logits_joint), jnp.zeros_like(logits_marg)], axis=0
    )
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def bnre_balance_from_logits(logits_joint: jnp.ndarray, logits_marg: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute BNRE balance penalty from logits.
    
    Returns:
        penalty: (balance - 1.0)^2 where balance = mean(sigmoid(logits_joint)) + mean(sigmoid(logits_marg))
        balance: mean(sigmoid(logits_joint)) + mean(sigmoid(logits_marg))
    """
    d_joint = jax.nn.sigmoid(logits_joint)
    d_marg = jax.nn.sigmoid(logits_marg)
    balance = jnp.mean(d_joint) + jnp.mean(d_marg)
    penalty = (balance - 1.0) ** 2
    return penalty, balance
