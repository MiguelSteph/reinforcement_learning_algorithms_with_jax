import jax
from jax import numpy as jnp
from typing import Tuple
from functools import partial
import optax
from dqn_types import DQNConfig, DQNState
from model import QNetwork


class Agent():
    def __init__(
        self, 
        dqn_config: DQNConfig
    ):
        self.network = QNetwork(
            dropout_rate = dqn_config.dropout_rate,
            action_size = dqn_config.n_actions,
        )
        self.temperature = dqn_config.temperature


    def init(
        self,
        rng_key: jax.Array,
        obs_shape: Tuple[int, int, int],
        learning_rate: float,
    ) -> DQNState:
        """Initialise network parameters and optimizer state."""
        net_key, rng_key = jax.random.split(rng_key, 2)
 
        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        online_params = self.network.init(net_key, dummy_obs)
        target_params = jax.tree.map(lambda x: x.copy(), online_params)

        return DQNState.create(
            apply_fn = self.network.apply,
            params = online_params,
            tx = optax.adam(learning_rate),
            target_params = target_params,
            rng_key = rng_key,
        )


    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        dqn_state: DQNState,
        obs: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.int32:
        q_values = dqn_state.apply_fn(dqn_state.params, jnp.expand_dims(obs, 0), train=False)[0]
        adjusted_q_values = q_values / self.temperature
        probs = jax.nn.softmax(adjusted_q_values)
        action = jax.random.choice(key, q_values.shape[-1], p=probs)
        return action.astype(jnp.int32)
