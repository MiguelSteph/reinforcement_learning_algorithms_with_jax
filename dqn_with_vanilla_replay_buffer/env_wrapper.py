import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation


class CarRacingEnvWrapper:
    """Wraps CarRacing-v3 with frame stacking, returning (H, W, n_stack*C) uint8 observations."""

    def __init__(self, n_stack: int = 4):
        env = gym.make("CarRacing-v3", continuous=False)
        self._env = FrameStackObservation(env, stack_size=n_stack)

    def _process_obs(self, obs) -> np.ndarray:
        frames = np.array(obs) # (4, 96, 96, 3)
        return frames.transpose(1, 2, 0, 3).reshape(96, 96, -1) # (96, 96, 12)

    def reset(self, seed: int | None = None):
        obs, info = self._env.reset(seed=seed)
        return self._process_obs(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info

    def close(self):
        self._env.close()

    @property
    def action_space(self):
        return self._env.action_space
