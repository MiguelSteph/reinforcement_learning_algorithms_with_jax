import jax
from typing import NamedTuple, Any
from jax import numpy as jnp
from flax.training import train_state
from configs import get_configs


config = get_configs()

class Transition(NamedTuple):
    """A single environment transition."""
    obs:      jnp.ndarray
    action:   jnp.int32
    reward:   jnp.float32
    next_obs: jnp.ndarray
    done:     jnp.bool_

class TransitionBatch(NamedTuple):
    """A batch of transitions returned by the replay buffer sample method."""
    obs:      jnp.ndarray  # (batch_size, *obs_dim)
    action:   jnp.ndarray  # (batch_size,)
    reward:   jnp.ndarray  # (batch_size,)
    next_obs: jnp.ndarray  # (batch_size, *obs_dim)
    done:     jnp.ndarray  # (batch_size,)

class BufferState(NamedTuple):
    """All mutable state carried by the buffer (fully JAX-compatible)."""
    obs:        jnp.ndarray   # (capacity, *obs_dim) uint8
    action:     jnp.ndarray   # (capacity,)
    reward:     jnp.ndarray   # (capacity,)
    next_obs:   jnp.ndarray   # (capacity, *obs_dim) uint8
    done:       jnp.ndarray   # (capacity,)
    cursor:     jnp.int32   # int32 — next write position
    size:       jnp.int32   # int32 — current valid entries

class DQNConfig(NamedTuple):
    """Architecture settings for the DQN Network."""
    n_actions: int = config.data.n_actions

class TrainerConfig(NamedTuple):
    """Hyperparameters and architecture settings for the DQN agent."""
    # Environment
    obs_shape: tuple[int, int, int] = config.data.obs_shape
    n_actions: int = config.data.n_actions

    # Exploration
    epsilon_start:       float = config.policy.epsilon_start
    epsilon_end:         float = config.policy.epsilon_end
    epsilon_decay_steps: int   = config.policy.epsilon_decay_steps

    # Training
    learning_rate: float = config.training.learning_rate
    gamma: float = config.training.gamma
    batch_size: int = config.training.batch_size
    target_sync_freq: int = config.training.target_sync_freq
    update_every: int = config.training.update_every
    n_episodes: int = config.training.n_episodes
    max_t_per_episode: int = config.training.max_t_per_episode
    check_reward_every: int = config.training.check_reward_every
    max_total_gradient_steps: int = config.training.max_total_gradient_steps

    # Replay buffer
    buffer_capacity: int = config.buffer.buffer_capacity

    # Output
    log_metric_path: str = config.log.log_metric_path
    log_checkpoint_path: str = config.log.log_checkpoint_path

PyTree = Any

class DQNState(train_state.TrainState):
    target_params: PyTree
