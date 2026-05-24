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
            action_size = dqn_config.n_actions,
        )

    def init(
        self,
        rng_key: jax.Array,
        obs_shape: Tuple[int, int, int],
        learning_rate: float,
        warm_up_steps: int,
        decay_steps: int,
    ) -> DQNState:
        """Initialise network parameters and optimizer state."""
        def create_learning_rate_scheduler(base_lr: jnp.float32,
                                        warmup_epochs: int,
                                        cosine_epochs: int) -> Any:
            warmup_fn = optax.linear_schedule(init_value=0, end_value=base_lr,
                                            transition_steps=warmup_epochs)
            cosine_fn = optax.cosine_decay_schedule(init_value=base_lr,
                                                    decay_steps=cosine_epochs)
            schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],
                                            boundaries=[warmup_epochs])
            return schedule_fn

        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        online_params = self.network.init(rng_key, dummy_obs)
        target_params = jax.tree.map(lambda x: x.copy(), online_params)

        scheduler = create_learning_rate_scheduler(learning_rate,
                                                    warm_up_steps,
                                                    decay_steps)

        # return DQNState.create(
        #     apply_fn = self.network.apply,
        #     params = online_params,
        #     tx = optax.chain(
        #       optax.clip_by_global_norm(10.0),
        #       optax.adam(scheduler),
        #     ),
        #     target_params = target_params,
        # )
        return DQNState.create(
            apply_fn = self.network.apply,
            params = online_params,
            tx = optax.adam(scheduler),
            target_params = target_params,
        )

    @partial(jax.jit, static_argnums=(0,))
    def select_action(
        self,
        dqn_state: DQNState,
        obs: jnp.ndarray,
        key: jax.Array,
        epsilon: float,
    ) -> jnp.int32:
        q_values = dqn_state.apply_fn(dqn_state.params, jnp.expand_dims(obs, 0))[0]
        greedy_action = jnp.argmax(q_values).astype(jnp.int32)
        explore_key, uniform_key = jax.random.split(key)
        random_action = jax.random.randint(explore_key, shape=(), minval=0, maxval=q_values.shape[-1]).astype(jnp.int32)
        return jnp.where(jax.random.uniform(uniform_key) < epsilon, random_action, greedy_action)

    @partial(jax.jit, static_argnums=(0,))
    def select_greedy_action(
        self,
        dqn_state: DQNState,
        obs: jnp.ndarray,
    ) -> jnp.int32:
        q_values = dqn_state.apply_fn(dqn_state.params, jnp.expand_dims(obs, 0))[0]
        return jnp.argmax(q_values).astype(jnp.int32)
