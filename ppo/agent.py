import jax
from jax import numpy as jnp
from flax.training import train_state
from typing import Tuple, Any
from functools import partial
import optax
from ppo_types import NetworkConfig
from model import ActorCritic


class Agent():
    def __init__(
        self,
        network_config: NetworkConfig
    ):
        self.action_size = network_config.n_actions
        self.network = ActorCritic(
            action_size = network_config.n_actions,
        )

    def init(
        self,
        rng_key: jax.Array,
        obs_shape: Tuple[int, int, int],
        learning_rate: float,
    ) -> train_state.TrainState:
        """Initialise network parameters and optimizer state."""
        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        params = self.network.init(rng_key, dummy_obs)

        lr_schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0.0,
            transition_steps=170_000,
        )
        return train_state.TrainState.create(
            apply_fn = self.network.apply,
            params = params,
            tx = optax.chain(
                optax.clip_by_global_norm(0.5),
                optax.adam(lr_schedule, eps=1e-5),
            )
        )

    @partial(jax.jit, static_argnums=(0,))
    def run_policy(
        self,
        state: train_state.TrainState,
        obs: jnp.ndarray,
    ) -> jax.Array:
        return state.apply_fn(state.params, obs) # log_probs, values

    @partial(jax.jit, static_argnums=(0,))
    def select_actions(
        self,
        log_probs: jnp.ndarray,
        key: jax.Array,
    ) -> jax.Array:
        return jax.random.categorical(key, log_probs)

    @partial(jax.jit, static_argnums=(0,))
    def select_greedy_actions(
        self,
        log_probs: jnp.ndarray,
    ) -> jax.Array:
        return jnp.argmax(log_probs, axis=-1)
