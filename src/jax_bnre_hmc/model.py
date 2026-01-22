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