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
    """Get activation function by name.
    
    Args:
        name: Name of the activation function. Must be one of: "tanh", "relu", "gelu", "silu".
    
    Returns:
        The activation function.
    
    Raises:
        ValueError: If the activation name is not recognized.
    """
    try:
        return _ACTIVATIONS[name]
    except KeyError as e:
        raise ValueError(f"Unknown activation '{name}'. Choose from {sorted(_ACTIVATIONS.keys())}.") from e


class RatioEstimatorMLP(nn.Module):
    """Multi-layer perceptron for neural ratio estimation.
    
    Implements f(theta, x) -> logit using an MLP over concatenated (theta, x).
    The network consists of fully connected layers with optional layer normalization
    and activation functions.
    
    Attributes:
        hidden_dims: Tuple of hidden layer dimensions. Default: (50, 50, 50).
        activation: Activation function name ("tanh", "relu", "gelu", or "silu"). Default: "tanh".
        norm: Normalization type, either "layernorm" or "none". Default: "layernorm".
    """
    hidden_dims: tuple[int, ...] = (50, 50, 50)
    activation: str = "tanh"
    norm: str = "layernorm"   # "layernorm" or "none"

    def setup(self):
        self.act = get_activation(self.activation)

    @nn.compact
    def __call__(self, theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the ratio estimator.
        
        Args:
            theta: Parameter batch of shape (B, theta_dim).
            x: Observation batch of shape (B, x_dim).
        
        Returns:
            Logits of shape (B,) representing the ratio estimate.
        """
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
    """Pre-activation residual MLP block with LayerNorm.
    
    Implements a residual connection with pre-activation normalization.
    Architecture: h -> LN -> act -> Dense -> LN -> act -> Dense -> + skip.
    
    Attributes:
        width: Width of the hidden layers.
        activation: Activation function name. Default: "relu".
    """
    width: int
    activation: str = "relu"

    def setup(self):
        self.act = get_activation(self.activation)

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the residual block.
        
        Args:
            h: Input features of shape (B, width).
        
        Returns:
            Output features of shape (B, width) with residual connection applied.
        """
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
    """ResNet-style MLP for neural ratio estimation.
    
    Implements f(theta, x) -> logit using a residual network architecture.
    This is an MLP "ResidualNet" (similar to SBI's), not a CNN ResNet.
    Architecture:
      - Input projection to hidden_features
      - num_blocks residual blocks
      - Output head to 1 logit
    
    Attributes:
        hidden_features: Width of hidden layers. Default: 50.
        num_blocks: Number of residual blocks. Default: 2.
        activation: Activation function name. Default: "relu".
    """
    hidden_features: int = 50
    num_blocks: int = 2
    activation: str = "relu"

    def setup(self):
        # Validate activation early
        _ = get_activation(self.activation)

    @nn.compact
    def __call__(self, theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass of the ResNet ratio estimator.
        
        Args:
            theta: Parameter batch of shape (B, theta_dim).
            x: Observation batch of shape (B, x_dim).
        
        Returns:
            Logits of shape (B,) representing the ratio estimate.
        """
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