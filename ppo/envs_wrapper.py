import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo

gym.register_envs(ale_py)


class FireOnResetWrapper(gym.Wrapper):
    """Sends FIRE (action 1) on every reset so the ball is always in play."""
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.reset(**kwargs)
        return obs, info


class EnvsWrapper:
    def __init__(self,
                 env_id: str = "ALE/Breakout-v5",
                 num_envs: int = 8,
                 n_stack: int = 4,
                 terminal_on_life_loss: bool = True):
        def make_env():
            def create_new_env():
                env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
                env = AtariPreprocessing(env, terminal_on_life_loss=terminal_on_life_loss)
                env = FireOnResetWrapper(env)
                return FrameStackObservation(env, stack_size=n_stack)
            return create_new_env

        self._envs = gym.vector.AsyncVectorEnv(
                [make_env() for i in range(num_envs)],
            )

        self._num_envs = num_envs
        self._terminal_on_life_loss = terminal_on_life_loss

    def _process_obs(self, obs) -> np.ndarray:
        return np.array(obs).transpose(0, 2, 3, 1)  # (8, 4, 84,84) → (8, 84, 84, 4)

    def reset_envs(self, seed: int | None = None):
        obs, info = self._envs.reset(seed=seed)
        return self._process_obs(obs), info

    def step_envs(self, actions: list[int]):
        obs, rewards, terminated, truncated, info = self._envs.step(actions)
        return self._process_obs(obs), rewards, terminated, truncated, info

    def close(self):
        self._envs.close()
