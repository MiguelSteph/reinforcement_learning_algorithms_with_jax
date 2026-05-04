import flax.linen as nn
from jax import numpy as jnp

class QNetwork(nn.Module):
    dropout_rate: float
    action_size: int
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(inputs)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
 
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)

        x = x.reshape(-1)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(self.action_size)(x) 
        return x
