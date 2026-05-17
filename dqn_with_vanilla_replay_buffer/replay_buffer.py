import numpy as np
import jax
from jax import numpy as jnp
from dqn_types import Transition, TransitionBatch


class ReplayBuffer:
    """Fixed-size replay buffer backed by NumPy arrays."""

    def __init__(self, capacity: int, obs_dim: list[int], seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_dim), dtype=np.uint8)
        self.action = np.zeros((capacity,), dtype=np.int32)
        self.reward = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_dim), dtype=np.uint8)
        self.done = np.zeros((capacity,), dtype=np.bool_)
        self.cursor = 0
        self.size = 0
        self._rng = np.random.default_rng(seed)

    def add(self, transition: Transition) -> None:
        self.obs[self.cursor] = transition.obs
        self.action[self.cursor] = transition.action
        self.reward[self.cursor] = transition.reward
        self.next_obs[self.cursor] = transition.next_obs
        self.done[self.cursor] = transition.done
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        indices = self._rng.choice(self.size, size=batch_size, replace=False)
        return TransitionBatch(
            obs = jnp.array(self.obs[indices]),
            action = jnp.array(self.action[indices]),
            reward = jnp.array(self.reward[indices]),
            next_obs = jnp.array(self.next_obs[indices]),
            done = jnp.array(self.done[indices]),
        )

    def is_ready(self, min_size: int) -> bool:
        return self.size >= min_size
