import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

gym.register_envs(ale_py)


class EnvWrapper:
    def __init__(self, env_id: str = "ALE/Breakout-v5", n_stack: int = 4):
        env = gym.make(env_id)
        env = AtariPreprocessing(env)
        self._env = FrameStackObservation(env, stack_size=n_stack)

    def _process_obs(self, obs) -> np.ndarray:
        return np.array(obs).transpose(1, 2, 0)  # (4,84,84) → (84,84,4)

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
