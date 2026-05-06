import os
import jax
from jax import numpy as jnp
from typing import Any, Tuple, Dict
from functools import partial
from tensorboardX import SummaryWriter
import datetime
import orbax.checkpoint as ocp
from tqdm.auto import tqdm
from agent import Agent
from dqn_types import TrainerConfig, BufferState, DQNState, Transition, TransitionBatch, PyTree
from replay_buffer import ReplayBuffer
from env_wrapper import EnvWrapper

class AgentTrainer():
    def __init__(
        self, 
        agent: Agent,
        trainer_config: TrainerConfig, 
        initial_rng_key: jax.Array
    ):
        self.cfg = trainer_config
        self.rng_key = initial_rng_key
        self.buffer = ReplayBuffer(self.cfg.buffer_capacity, self.cfg.obs_shape)
        self.t_update = 0
        self.t_target_sync = 0
        self.agent = agent
        self.buffer_ready = False 

        # Create Summary writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = self.cfg.log_metric_path + '/' + current_time + '/train'
        reward_log_dir = self.cfg.log_metric_path + '/' + current_time + '/reward'
        self.train_summary_writer = SummaryWriter(train_log_dir)
        self.reward_summary_writer = SummaryWriter(reward_log_dir)

        # Create the checkpoint manager
        ckp_options = ocp.CheckpointManagerOptions(
            max_to_keep = 1, 
            best_fn = lambda metrics: metrics['reward'],
            best_mode = 'max'
        )
        self.ckp_mngr = ocp.CheckpointManager(
            os.path.abspath(self.cfg.log_checkpoint_path),
            options=ckp_options
        )

    def init(self) -> BufferState:
        buffer_state = self.buffer.init()
        return buffer_state

    def play_full_episode(self, env: EnvWrapper, dqn_state: DQNState) -> float:
        rewards = []
        obs, _ = env.reset()
        for _ in range(self.cfg.max_t_per_episode):
            self.rng_key, sample_rng_key = jax.random.split(self.rng_key, 2)
            action = self.agent.select_greedy_action(dqn_state, obs)
            obs, reward, terminated, truncated, *_ = env.step(int(action))
            rewards.append(reward)
            done = terminated or truncated
            if done:
                break

        discounted_reward_sum = 0
        for t in range(len(rewards) - 1, -1, -1):
            discounted_reward_sum = rewards[t] + self.cfg.gamma * discounted_reward_sum
        return discounted_reward_sum

    def run_train_loop(
        self,
        env: EnvWrapper,
        dqn_state: DQNState,
    ) -> DQNState:
        buffer_state = self.init()
        for episode in tqdm(range(1, self.cfg.n_episodes + 1)):
            # Reset the environment
            print(f"Episode {episode} started")
            obs, _ = env.reset()
            for _ in range(self.cfg.max_t_per_episode):
                self.rng_key, sample_rng_key = jax.random.split(self.rng_key, 2)
                action = self.agent.select_action(dqn_state, obs, sample_rng_key)
                next_obs, reward, terminated, truncated, *_ = env.step(int(action))
                done = terminated or truncated
                transition = Transition(obs, action, reward, next_obs, done)
                dqn_state, buffer_state = self.step(dqn_state, buffer_state, transition)
                obs = next_obs
                if done:
                    break

            if episode % self.cfg.check_reward_every == 0:
                full_episode_discounted_reward = self.play_full_episode(env, dqn_state)
                print(f"Reward checking: Episode: {episode}, Total discounted reward: {full_episode_discounted_reward}")
                self.reward_summary_writer.add_scalar('reward', full_episode_discounted_reward, episode)

                self.ckp_mngr.save(
                    episode, 
                    args=ocp.args.StandardSave(dqn_state), 
                    metrics={'reward': full_episode_discounted_reward,}
                )
        return dqn_state

    def step(self, dqn_state: DQNState, buffer_state: BufferState, transition: Transition) -> Tuple[DQNState, BufferState]:
        # Save transition in the replay buffer
        buffer_state = self.buffer.add(buffer_state, transition)

        self.t_update = (self.t_update + 1) % self.cfg.update_every
        self.t_target_sync = (self.t_target_sync + 1) % self.cfg.target_sync_freq
        if not self.buffer_ready:
          self.buffer_ready = bool(self.buffer.is_ready(buffer_state, self.cfg.min_buffer_size))
        new_dqn_state = dqn_state
        if self.t_update == 0 and self.buffer_ready:
            self.rng_key, sample_rng_key = jax.random.split(self.rng_key, 2)
            indices = jax.random.choice(
                sample_rng_key, int(buffer_state.size), shape=(self.cfg.batch_size,), replace=False
            )
            sample_batch = self.buffer.sample(buffer_state, indices)
            new_dqn_state, metrics = self.train_step(dqn_state, sample_batch)
            # self.train_summary_writer.add_scalar('loss', float(metrics['loss']), int(dqn_state.step))

        if self.t_target_sync == 0:
            new_target_params = jax.tree.map(lambda x: x.copy(), new_dqn_state.params)
            new_dqn_state = new_dqn_state.replace(target_params=new_target_params)

        return new_dqn_state, buffer_state

    @partial(jax.jit,
             static_argnums=(0,),
            )
    def train_step(self, dqn_state: DQNState, batch: TransitionBatch) -> Tuple[DQNState, Dict]:
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(self._loss)(
            dqn_state.params,
            dqn_state.target_params,
            dqn_state.apply_fn,
            batch,
        )
        new_state = dqn_state.apply_gradients(grads=grads)
        return new_state, {"loss": loss}

    def _loss(self,
              online_params: PyTree,
              target_params: PyTree,
              apply_fn: Any,
              batch: TransitionBatch) -> PyTree:

        def q_online(x):
            return apply_fn(
                online_params, x
            )

        def q_target(x):
            return apply_fn(target_params, x)

        with jax.named_scope("computing_q_values"):
            q_values = q_online(batch.obs) # (B, n_actions)
            # Gather Q(s, a) for the taken action
            q_taken = q_values[jnp.arange(q_values.shape[0]), batch.action]  # (B,)

        with jax.named_scope("computing_q_target"):
            q_next = q_target(batch.next_obs) # (B, n_actions)
            # TD target: r + gamma * max_a Q_target(s', a)  — masked by done
            q_next_max = jnp.max(q_next, axis=-1) # (B,)
            done = batch.done.astype(jnp.int32)
            td_target  = batch.reward + self.cfg.gamma * q_next_max * (1 - done)
            # Stop gradient through the target
            td_target = jax.lax.stop_gradient(td_target)

        return jnp.mean((q_taken - td_target) ** 2)
