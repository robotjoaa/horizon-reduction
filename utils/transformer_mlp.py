from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp


def default_init(scale: float = 1.0):
    """Default kernel initializer (fan_avg, uniform)."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


@dataclass(frozen=True)
class TransformerSpec:
    """Configuration for the Transformer feature extractor."""

    proj_dim: int  # Tr
    sequence_length: int  # Tℓ
    token_dim: int  # Tk
    num_layers: int  # Tn
    mlp_dim: int  # Tm

    def __post_init__(self):
        if self.sequence_length * self.token_dim != self.proj_dim:
            raise ValueError('proj_dim must equal sequence_length * token_dim')


TRANSFORMER_SPECS: Mapping[str, TransformerSpec] = {
    'small': TransformerSpec(proj_dim=2048, sequence_length=16, token_dim=128, num_layers=4, mlp_dim=128),
    'large': TransformerSpec(proj_dim=2048, sequence_length=8, token_dim=256, num_layers=10, mlp_dim=1024),
}


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    token_dim: int
    mlp_dim: int
    num_heads: int = 4
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    kernel_init: Any = default_init()

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        # Multi-head self-attention with pre-normalization.
        residual = x
        y = nn.LayerNorm()(x)
        y = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.token_dim,
            out_features=self.token_dim,
            dropout_rate=self.attention_dropout_rate,
            deterministic=deterministic,
            kernel_init=self.kernel_init,
        )(y)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        x = residual + y

        # Position-wise feed-forward network.
        residual = x
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim, kernel_init=self.kernel_init)(y)
        y = nn.gelu(y)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(self.token_dim, kernel_init=self.kernel_init)(y)
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate)(y, deterministic=deterministic)
        return residual + y


class TransformerMLP(nn.Module):
    """Transformer-based drop-in replacement for MLP modules."""

    hidden_dims: Sequence[int]
    transformer_variant: str = 'small'
    transformer_spec: Optional[TransformerSpec] = None
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False
    num_heads: int = 4
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0

    def _resolve_spec(self) -> TransformerSpec:
        if self.transformer_spec is not None:
            return self.transformer_spec
        if self.transformer_variant not in TRANSFORMER_SPECS:
            raise ValueError(f'Unknown transformer_variant "{self.transformer_variant}".')
        return TRANSFORMER_SPECS[self.transformer_variant]

    @nn.compact
    def __call__(self, x, *, deterministic: bool = True):
        if not self.hidden_dims:
            raise ValueError('TransformerMLP expects at least one entry in hidden_dims (the output size).')
        else :
            spec = TransformerSpec(
                    proj_dim = self.hidden_dims[0],
                    sequence_length = self.hidden_dims[1],
                    token_dim = self.hidden_dims[2],
                    num_layers = self.hidden_dims[3],
                    mlp_dim = self.hidden_dims[4],
                )
            #print(self.hidden_dims)
            #assert len(self.hidden_dims) == 6 # proj_dim, sequence_length, token_dim, num_layers, mlp_dim, last_dense
            #proj_dim, sequence_length, token_dim, num_layers, mlp_dim = self.hidden_dims
            # spec = self._resolve_spec()
            

        # Project inputs to Tr and reshape into (Tℓ, Tk).
        y = nn.Dense(spec.proj_dim, kernel_init=self.kernel_init)(x)
        new_shape = (*y.shape[:-1], spec.sequence_length, spec.token_dim)
        y = jnp.reshape(y, new_shape)

        # Stack Transformer encoder blocks.
        for i in range(spec.num_layers):
            y = TransformerBlock(
                token_dim=spec.token_dim,
                mlp_dim=spec.mlp_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                kernel_init=self.kernel_init,
                name=f'transformer_block_{i}',
            )(y, deterministic=deterministic)

        # Concatenate token embeddings for downstream dense layers.
        y = nn.LayerNorm()(y)
        features = jnp.reshape(y, (*y.shape[:-2], spec.proj_dim))
        #print(features.shape)

        self.sow('intermediates', 'feature', features) # used for ensemble
        
        if len(self.hidden_dims) == 6 : # 5 default, but GCValue adds 1 at the end
            outputs = nn.Dense(self.hidden_dims[-1], kernel_init=self.kernel_init)(features)
        else : 
            outputs = features

        if self.activate_final:
            outputs = self.activations(outputs)
            if self.layer_norm:
                outputs = nn.LayerNorm()(outputs)

        return outputs