from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from .data import Batch


def nre_loss_from_logits(logits_joint: jnp.ndarray, logits_marginal: jnp.ndarray) -> jnp.ndarray:
    """Compute Neural Ratio Estimation (NRE) loss from logits.
    
    Implements the label-free NRE objective where joint samples are implicitly
    labeled as positive (1) and marginal samples as negative (0):
    
        L = E_joint[softplus(-logit)] + E_marg[softplus(logit)]
    
    Args:
        logits_joint: Logits for joint (theta, x) pairs. Shape: (batch_size,).
        logits_marginal: Logits for marginal (theta, x_shuffled) pairs. Shape: (batch_size,).
    
    Returns:
        Scalar NRE loss value.
    """
    return jnp.mean(jax.nn.softplus(-logits_joint)) + jnp.mean(jax.nn.softplus(logits_marginal))


def nre_loss(apply_fn, params, joint: Batch, marginal: Batch) -> jnp.ndarray:
    """Compute NRE loss by applying the model and computing loss from logits.
    
    Args:
        apply_fn: Model forward function.
        params: Model parameters.
        joint: Batch of joint (theta, x) pairs.
        marginal: Batch of marginal (theta, x_shuffled) pairs.
    
    Returns:
        Scalar NRE loss value.
    """
    logits_joint = apply_fn(params, joint.theta, joint.x)
    logits_marg = apply_fn(params, marginal.theta, marginal.x)
    return nre_loss_from_logits(logits_joint, logits_marg)


def nre_loss_bce_style_from_logits(logits_joint: jnp.ndarray, logits_marg: jnp.ndarray) -> jnp.ndarray:
    """Compute NRE loss in BCE-with-logits style.
    
    Reformulates the NRE objective as binary cross-entropy with logits over
    concatenated joint and marginal examples. This is mathematically equivalent
    to nre_loss_from_logits but written in a form that's useful for direct
    comparison with SBI implementations.
    
    Args:
        logits_joint: Logits for joint (theta, x) pairs. Shape: (batch_size,).
        logits_marg: Logits for marginal (theta, x_shuffled) pairs. Shape: (batch_size,).
    
    Returns:
        Scalar BCE-style NRE loss value.
    """
    logits = jnp.concatenate([logits_joint, logits_marg], axis=0)
    labels = jnp.concatenate(
        [jnp.ones_like(logits_joint), jnp.zeros_like(logits_marg)], axis=0
    )
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))


def bnre_balance_from_logits(logits_joint: jnp.ndarray, logits_marg: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Balanced Neural Ratio Estimation (BNRE) balance penalty.
    
    Computes the balance term B = mean(sigmoid(logits_joint)) + mean(sigmoid(logits_marg))
    and the penalty (B - 1.0)^2. This penalty encourages the classifier to be balanced,
    i.e., to assign equal average probability to joint and marginal samples.
    
    Args:
        logits_joint: Logits for joint (theta, x) pairs. Shape: (batch_size,).
        logits_marg: Logits for marginal (theta, x_shuffled) pairs. Shape: (batch_size,).
    
    Returns:
        A tuple containing:
            - penalty: The balance penalty (balance - 1.0)^2.
            - balance: The balance value mean(sigmoid(logits_joint)) + mean(sigmoid(logits_marg)).
    """
    d_joint = jax.nn.sigmoid(logits_joint)
    d_marg = jax.nn.sigmoid(logits_marg)
    balance = jnp.mean(d_joint) + jnp.mean(d_marg)
    penalty = (balance - 1.0) ** 2
    return penalty, balance
