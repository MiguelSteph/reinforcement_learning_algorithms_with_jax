import jax
import numpy as np
from jax import numpy as jnp
from dqn_types import BufferState, Transition, TransitionBatch


class ReplayBuffer:
    """Fixed size replay buffer backed by CPU numpy arrays.

    Observations live in CPU RAM. Only the sampled batch is transferred
    to the GPU via jnp.array() on each training step.
    """

    def __init__(self, capacity: int, obs_dim: list[int]):
        self.capacity = capacity
        self.obs_dim  = obs_dim

    def init(self) -> BufferState:
        return BufferState(
            obs      = np.zeros((self.capacity, *self.obs_dim), dtype=np.uint8),
            action   = np.zeros((self.capacity,), dtype=np.int32),
            reward   = np.zeros((self.capacity,), dtype=np.float32),
            next_obs = np.zeros((self.capacity, *self.obs_dim), dtype=np.uint8),
            done     = np.zeros((self.capacity,), dtype=np.bool_),
            cursor   = np.array(0, dtype=np.int32),
            size     = np.array(0, dtype=np.int32),
        )

    def add(self, state: BufferState, transition: Transition) -> BufferState:
        idx = int(state.cursor)
        state.obs[idx] = transition.obs
        state.action[idx] = transition.action
        state.reward[idx] = transition.reward
        state.next_obs[idx] = transition.next_obs
        state.done[idx] = transition.done
        state.cursor = (idx + 1) % self.capacity
        state.size = min(int(state.size) + 1, self.capacity)
        return state

    def sample(self, state: BufferState, key: jax.Array, batch_size: int) -> TransitionBatch:
        indices = np.array(jax.random.choice(key, self.capacity, shape=(batch_size,), replace=False))
        return TransitionBatch(
            obs = jnp.array(state.obs[indices]),
            action = jnp.array(state.action[indices]),
            reward = jnp.array(state.reward[indices]),
            next_obs = jnp.array(state.next_obs[indices]),
            done = jnp.array(state.done[indices]),
        )

    def is_ready(self, state: BufferState, min_size: int) -> bool:
        return int(state.size) >= min_size
