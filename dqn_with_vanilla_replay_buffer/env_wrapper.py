import numpy as np
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo

gym.register_envs(ale_py)


class EnvWrapper:
    def __init__(self, 
                 env_id: str = "ALE/Breakout-v5", 
                 n_stack: int = 6, 
                 terminal_on_life_loss: bool = True,
                 video_folder: str|None = None):
        env = gym.make(env_id, frameskip=1, render_mode="rgb_array")
        env = AtariPreprocessing(env, terminal_on_life_loss=terminal_on_life_loss)
        if video_folder is not None:
            env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: True)
        self._env = FrameStackObservation(env, stack_size=n_stack)
        self._terminal_on_life_loss = terminal_on_life_loss

    def _process_obs(self, obs) -> np.ndarray:
        return np.array(obs).transpose(1, 2, 0)  # (4,84,84) → (84,84,4)

    def reset(self, seed: int | None = None):
        obs, info = self._env.reset(seed=seed)
        obs, _, terminated, truncated, info = self._env.step(1)  # FIRE
        if terminated or truncated:
          obs, info = self.reset(seed=seed)
        self._lives = info.get('lives', 0)
        return self._process_obs(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not self._terminal_on_life_loss and info.get('lives', 0) < self._lives and not (terminated or truncated):
            obs, _, terminated, truncated, info = self._env.step(1)  # FIRE after life loss
        return self._process_obs(obs), reward, terminated, truncated, info

    def close(self):
        self._env.close()

    @property
    def action_space(self):
        return self._env.action_space
