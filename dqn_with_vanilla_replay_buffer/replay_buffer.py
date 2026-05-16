import jax
from jax import numpy as jnp
from functools import partial
from dqn_types import BufferState, Transition, TransitionBatch


class ReplayBuffer:
    """Fixed size replay buffer to store experiences."""

    def __init__(self, capacity: int, obs_dim: list[int]):
        self.capacity = capacity
        self.obs_dim  = obs_dim

    def init(self) -> BufferState:
        """Return an empty BufferState."""
        return BufferState(
            obs = jnp.zeros((self.capacity, *self.obs_dim), dtype = jnp.uint8),
            action = jnp.zeros((self.capacity,), dtype =jnp.int32),
            reward = jnp.zeros((self.capacity,), dtype = jnp.float32),
            next_obs = jnp.zeros((self.capacity, *self.obs_dim), dtype = jnp.uint8),
            done = jnp.zeros((self.capacity,), dtype = jnp.bool_),
            cursor = jnp.int32(0),
            size = jnp.int32(0),
        )

    @partial(jax.jit, static_argnums=(0,), donate_argnames=('state',))
    def add(self, state: BufferState, transition: Transition) -> BufferState:
        """Insert one transition into the buffer (overwrites oldest if full)."""
        idx = state.cursor
        return BufferState(
            obs = state.obs.at[idx].set(transition.obs),
            action = state.action.at[idx].set(transition.action),
            reward = state.reward.at[idx].set(transition.reward),
            next_obs = state.next_obs.at[idx].set(transition.next_obs),
            done = state.done.at[idx].set(transition.done),
            cursor = (idx + 1) % self.capacity,
            size = jnp.minimum(state.size + 1, self.capacity),
        )

    @partial(jax.jit, static_argnums=(0,3))
    def sample(self, state: BufferState, key: jax.Array, batch_size: int) -> TransitionBatch:
        """Retrieve transitions at the given indices."""
        indices = jax.random.choice(
            key, self.capacity, shape=(batch_size,), replace=False
        )
        return TransitionBatch(
            obs      = state.obs[indices],
            action   = state.action[indices],
            reward   = state.reward[indices],
            next_obs = state.next_obs[indices],
            done     = state.done[indices],
        )

    def is_ready(self, state: BufferState, min_size: int) -> bool:
        return int(state.size) >= min_size
