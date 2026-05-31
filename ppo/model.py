import flax.linen as nn
from jax import numpy as jnp


class ActorCritic(nn.Module):
    action_size: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs.astype(jnp.float32) / 255.0
        x = nn.Conv(features=16, kernel_size=(8, 8), strides=(4, 4), padding='VALID')(x)
        x = nn.relu(x)

        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2), padding='VALID')(x)
        x = nn.relu(x)

        x = x.reshape(x.shape[0], -1)

        x = nn.Dense(256)(x)
        x = nn.relu(x)

        logits = nn.Dense(self.action_size)(x)
        log_probabilities = nn.log_softmax(logits)

        value = nn.Dense(1)(x)

        return log_probabilities, value
