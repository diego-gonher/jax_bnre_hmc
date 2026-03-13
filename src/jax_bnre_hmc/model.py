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


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> jnp.ndarray:
    """Create deterministic sinusoidal positional encodings.

    Args:
        seq_len: Sequence length N.
        d_model: Embedding dimension.

    Returns:
        Positional encoding array of shape (1, N, d_model).
    """
    position = jnp.arange(seq_len, dtype=jnp.float32)[:, None]  # (N, 1)
    div_term = jnp.exp(
        -jnp.log(10000.0) * jnp.arange(0, d_model, 2, dtype=jnp.float32) / d_model
    )  # (ceil(d_model/2),)

    pe = jnp.zeros((seq_len, d_model), dtype=jnp.float32)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[: (d_model // 2)]))

    return pe[None, :, :]  # (1, N, d_model)


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm transformer encoder block.

    Attributes:
        d_model: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden width of the feed-forward subnetwork.
        activation: Activation function name.
    """
    d_model: int
    num_heads: int
    mlp_dim: int
    activation: str = "gelu"

    def setup(self):
        self.act = get_activation(self.activation)

    @nn.compact
    def __call__(self, h: jnp.ndarray, attention_mask: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            h: Input tensor of shape (B, N, d_model).
            attention_mask: Boolean mask of shape (B, 1, N, N), where True means
                attention is allowed and False means masked out.

        Returns:
            Output tensor of shape (B, N, d_model).
        """
        # Pre-LN self-attention block
        y = nn.LayerNorm()(h)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            dropout_rate=0.0,
            deterministic=True,
        )(y, y, mask=attention_mask)
        h = h + y

        # Pre-LN feed-forward block
        y = nn.LayerNorm()(h)
        y = nn.Dense(self.mlp_dim)(y)
        y = self.act(y)
        y = nn.Dense(self.d_model)(y)
        h = h + y

        return h


class RatioEstimatorTransformer(nn.Module):
    """Transformer-based ratio estimator for masked 1D observations.

    This model expects observations x with shape (B, N, 2), where each token is:
        [y_i, m_i]

    The observation is embedded with a transformer encoder, pooled into a single
    vector with mask-aware mean pooling, concatenated with theta, and passed through
    an MLP head to produce a scalar logit.

    Attributes:
        d_model: Transformer embedding dimension.
        num_layers: Number of transformer encoder blocks.
        num_heads: Number of attention heads.
        transformer_mlp_dim: Hidden width in the transformer feed-forward blocks.
        transformer_activation: Activation used inside transformer blocks.
        head_hidden_dims: Hidden dimensions of the MLP head after concatenating theta and z.
        head_activation: Activation function for the MLP head.
        head_norm: Normalization type for the MLP head ("layernorm" or "none").
        eps: Small constant for numerical stability in masked pooling.
    """
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    transformer_mlp_dim: int = 256
    transformer_activation: str = "gelu"

    head_hidden_dims: tuple[int, ...] = (50, 50, 50)
    head_activation: str = "tanh"
    head_norm: str = "layernorm"

    eps: float = 1e-8

    def setup(self):
        self.head_act = get_activation(self.head_activation)
        # Validate transformer activation early
        _ = get_activation(self.transformer_activation)

    @nn.compact
    def __call__(self, theta: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            theta: Parameter batch of shape (B, theta_dim).
            x: Observation batch of shape (B, N, 2), with tokens [y_i, m_i].

        Returns:
            Logits of shape (B,).
        """
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, N, 2), got {x.shape}")
        if x.shape[-1] != 2:
            raise ValueError(f"x must have last dimension 2 for [y_i, m_i], got {x.shape}")

        x = jnp.asarray(x, dtype=jnp.float32)

        # Extract mask channel
        # valid: (B, N), boolean
        valid = x[..., 1] > 0.5

        # Token projection from 2 -> d_model
        h = nn.Dense(self.d_model)(x)  # (B, N, d_model)

        # Add deterministic sinusoidal positional encoding
        seq_len = x.shape[1]
        pe = sinusoidal_positional_encoding(seq_len, self.d_model)  # (1, N, d_model)
        h = h + pe

        # Build self-attention mask: allow attention only between valid tokens
        # shape: (B, 1, N, N)
        attention_mask = valid[:, None, :, None] & valid[:, None, None, :]

        # Transformer encoder
        for _ in range(self.num_layers):
            h = TransformerEncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                mlp_dim=self.transformer_mlp_dim,
                activation=self.transformer_activation,
            )(h, attention_mask)

        # Mask-aware mean pooling
        valid_f = valid.astype(jnp.float32)  # (B, N)
        denom = jnp.sum(valid_f, axis=1, keepdims=True) + self.eps  # (B, 1)
        z = jnp.sum(h * valid_f[..., None], axis=1) / denom  # (B, d_model)

        # Concatenate theta with pooled embedding
        h_head = jnp.concatenate([theta, z], axis=-1)

        # MLP head
        for d in self.head_hidden_dims:
            h_head = nn.Dense(d)(h_head)
            if self.head_norm == "layernorm":
                h_head = nn.LayerNorm()(h_head)
            h_head = self.head_act(h_head)

        logit = nn.Dense(1)(h_head)
        return jnp.squeeze(logit, axis=-1)
        