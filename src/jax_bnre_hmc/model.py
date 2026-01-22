from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


_ACTIVATIONS: dict[str, Callable[[jnp.ndarray], jnp.ndarray]] = {
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "silu": jax.nn.silu,
}


def get_activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    try:
        return _ACTIVATIONS[name]
    except KeyError as e:
        raise ValueError(f"Unknown activation '{name}'. Choose from {sorted(_ACTIVATIONS.keys())}.") from e


class RatioEstimatorMLP(nn.Module):
    """f(theta, x) -> logit using an MLP over concat(theta, x)."""
    hidden_dims: tuple[int, ...] = (50, 50, 50)
    activation: str = "tanh"
    norm: str = "layernorm"   # "layernorm" or "none"

    def setup(self):
        self.act = get_activation(self.activation)

    @nn.compact
    def __call__(self, theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        # Expect theta: (B, theta_dim), x: (B, x_dim)
        z = jnp.concatenate([theta, x], axis=-1)

        h = z
        for d in self.hidden_dims:
            h = nn.Dense(d)(h)
            if self.norm == "layernorm":
                h = nn.LayerNorm()(h)
            h = self.act(h)

        logit = nn.Dense(1)(h)  # (B, 1)
        return jnp.squeeze(logit, axis=-1)  # (B,)


class ResidualBlock(nn.Module):
    """A simple pre-activation residual MLP block with LayerNorm."""
    width: int
    activation: str = "relu"

    def setup(self):
        self.act = get_activation(self.activation)

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        # Pre-activation style:
        # h -> LN -> act -> Dense -> LN -> act -> Dense -> + skip
        y = nn.LayerNorm()(h)
        y = self.act(y)
        y = nn.Dense(self.width)(y)

        y = nn.LayerNorm()(y)
        y = self.act(y)
        y = nn.Dense(self.width)(y)

        return h + y


class RatioEstimatorResNet(nn.Module):
    """
    f(theta, x) -> logit using a ResNet-style MLP over concat(theta, x).

    This is an MLP "ResidualNet" (like SBI's), not a CNN ResNet:
      - input projection to width
      - num_blocks residual blocks
      - output head to 1 logit
    """
    hidden_features: int = 50
    num_blocks: int = 2
    activation: str = "relu"

    def setup(self):
        # Validate activation early
        _ = get_activation(self.activation)

    @nn.compact
    def __call__(self, theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        z = jnp.concatenate([theta, x], axis=-1)

        # Project to residual width
        h = nn.Dense(self.hidden_features)(z)

        # Residual blocks
        for _ in range(int(self.num_blocks)):
            h = ResidualBlock(width=self.hidden_features, activation=self.activation)(h)

        # A final normalization + activation before the head is often helpful
        h = nn.LayerNorm()(h)
        h = get_activation(self.activation)(h)

        logit = nn.Dense(1)(h)
        return jnp.squeeze(logit, axis=-1)