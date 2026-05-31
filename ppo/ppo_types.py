import jax
import numpy as np
from typing import Any, NamedTuple
from dataclasses import dataclass

PyTree = Any

class NetworkConfig(NamedTuple):
    n_actions: int = config.data.n_actions

@dataclass
class EnvsTransition:
    """Information about one transition of all the environments"""
    obs: np.ndarray               # (num_envs, 84, 84, 4)
    actions: np.ndarray           # (num_envs,)
    rewards: np.ndarray           # (num_envs,)
    terminated: np.ndarray        # (num_envs,)
    truncated: np.ndarray         # (num_envs,)
    log_probs: np.ndarray         # (num_envs,)
    values: np.ndarray            # (num_envs,)
    advantages: np.ndarray = None # (num_envs,)
    returns: np.ndarray = None    # (num_envs,)

class BatchSamples(NamedTuple):
    obs: jax.Array            # (batch_size, 84, 84, 4)
    actions: jax.Array        # (batch_size,)
    old_log_probs: jax.Array  # (batch_size,)
    advantages: jax.Array     # (batch_size,)
    returns: jax.Array        # (batch_size,)

class TrainSamples(NamedTuple):
    obs: np.ndarray            # (num_samples, 84, 84, 4)
    actions: np.ndarray        # (num_samples,)
    old_log_probs: np.ndarray  # (num_samples,)
    advantages: np.ndarray     # (num_samples,)
    returns: np.ndarray        # (num_samples,)

class TrainerConfig(NamedTuple):
    """Hyperparameters and architecture settings for the DQN agent."""
    # Environment
    obs_shape: tuple[int, int, int] = config.data.obs_shape
    n_actions: int = config.data.n_actions

    # Training
    learning_rate: float = config.training.learning_rate
    gamma: float = config.training.gamma
    gae_lambda: float = config.training.gae_lambda
    clip_coef: float = config.training.clip_coef
    batch_size: int = config.training.batch_size
    num_envs: int = config.training.num_envs
    eval_num_envs: int = config.training.eval_num_envs
    steps_per_env: int = config.training.steps_per_env
    ent_coef: int = config.training.ent_coef
    vf_coef: int = config.training.vf_coef
    num_rollouts: int = config.training.num_rollouts
    rollouts_per_eval: int = config.training.rollouts_per_eval
    num_epochs_per_rollout: int = config.training.num_epochs_per_rollout

    # Output
    log_metric_path: str = config.log.log_metric_path
    log_checkpoint_path: str = config.log.log_checkpoint_path
