import os
import jax
import numpy as np
from jax import numpy as jnp
from flax.training import train_state
from typing import Any, Tuple, Dict
from functools import partial
from tensorboardX import SummaryWriter
import datetime
import orbax.checkpoint as ocp
from tqdm.auto import tqdm
from dataclasses import dataclass
from ppo_types import EnvsTransition, PyTree, TrainerConfig, TrainSamples, BatchSamples
from agent import Agent
from envs_wrapper import EnvsWrapper


class AgentTrainer():
    def __init__(
        self,
        agent: Agent,
        trainer_config: TrainerConfig,
        initial_rng_key: jax.Array
    ):
        self.cfg = trainer_config
        self.rng_key = initial_rng_key
        self.rng_key, data_rng_key = jax.random.split(self.rng_key, 2)
        train_data_seed = int(jax.random.randint(data_rng_key, shape=(), minval=0, maxval=2**31 - 1))
        self._rng = np.random.default_rng(train_data_seed)
        self.agent = agent
        self.envs_wrapper = EnvsWrapper(terminal_on_life_loss=False)
        self.eval_envs_wrapper = EnvsWrapper(num_envs=self.cfg.eval_num_envs, terminal_on_life_loss=False)
        self._total_steps = 0

        # Create Summary writer
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = self.cfg.log_metric_path + '/' + current_time + '/train'
        eval_log_dir = self.cfg.log_metric_path + '/' + current_time + '/eval'
        self.train_summary_writer = SummaryWriter(train_log_dir)
        self.eval_summary_writer = SummaryWriter(eval_log_dir)

        # Create the checkpoint manager
        ckp_options = ocp.CheckpointManagerOptions(
            max_to_keep = 3,
            best_fn = lambda metrics: metrics['avg_rewards_sum'],
            best_mode = 'max'
        )
        self.ckp_mngr = ocp.CheckpointManager(
            os.path.abspath(self.cfg.log_checkpoint_path),
            options=ckp_options
        )

    def run_train_loop(self, state: train_state.TrainState) -> train_state.TrainState:
        obs, _ = self.envs_wrapper.reset_envs()
        for num_rollout in tqdm(range(1, self.cfg.num_rollouts + 1)):
            if num_rollout % 50 == 0:
                print(f"Rollout {num_rollout} started")
            obs, transitions = self._collect_env_transitions(state, obs)
            train_samples = self._get_train_samples(transitions)
            for epoch in range(self.cfg.num_epochs_per_rollout):
                batch_samples = self._get_batch_train_samples(train_samples)
                losses = []
                for batch in batch_samples:
                    state, metrics = self.train_step(state, batch)
                    losses.append(float(metrics['loss']))
                mean_loss = sum(losses) / len(losses)
                curr_epoch = (num_rollout - 1) * self.cfg.num_epochs_per_rollout + epoch
                self.train_summary_writer.add_scalar('loss', mean_loss, curr_epoch)
                # if curr_epoch % 50 == 0:
                #     print(f"Training: Epoch: {curr_epoch}, Avg loss: {mean_loss}")
            
            if num_rollout % self.cfg.rollouts_per_eval == 0:
                avg_rewards_sum = self.run_eval(state)
                self.eval_summary_writer.add_scalar('avg_rewards_sum', avg_rewards_sum, num_rollout)
                self.ckp_mngr.save(
                    num_rollout,
                    args=ocp.args.StandardSave(state),
                    metrics={'avg_rewards_sum': float(avg_rewards_sum),}
                )
                print(f"Evaluation: Rollout: {num_rollout}, Avg rewards sum: {avg_rewards_sum}")

        self.ckp_mngr.wait_until_finished()
        self.envs_wrapper.close()
        self.eval_envs_wrapper.close()
        
        return state

    @partial(jax.jit,
             static_argnums=(0,),
            )
    def train_step(self, state: train_state.TrainState, batch: BatchSamples) -> Tuple[train_state.TrainState, Dict]:
        # Compute loss and gradients
        loss, grads = jax.value_and_grad(self._loss)(
            state.params,
            state.apply_fn,
            batch,
        )
        new_state = state.apply_gradients(grads=grads)
        return new_state, {"loss": loss}

    def _loss(self, params: PyTree, apply_fn: Any, batch: BatchSamples) -> PyTree:
        log_probs, values = apply_fn(params, batch.obs)
        probs = jnp.exp(log_probs)
        values = values[:, 0]

        value_loss = jnp.mean((values - batch.returns) ** 2)

        entropy_loss = jnp.sum(-probs * log_probs, axis=-1).mean()

        advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
        log_probs_act_taken = log_probs[jnp.arange(log_probs.shape[0]), batch.actions.astype(jnp.int32)]
        ratios = jnp.exp(log_probs_act_taken - batch.old_log_probs)
        pg_loss_1 = ratios * advantages
        pg_loss_2 = jnp.clip(ratios, 1 - self.cfg.clip_coef, 1 + self.cfg.clip_coef) * advantages
        pg_loss = -jnp.minimum(pg_loss_1, pg_loss_2).mean()

        return pg_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy_loss

    def run_eval(self, state: train_state.TrainState) -> np.float32:
        obs, _ = self.eval_envs_wrapper.reset_envs()
        episode_rewards = np.zeros(self.cfg.eval_num_envs, dtype=np.float32)
        episode_done = np.zeros(self.cfg.eval_num_envs, dtype=bool)
        step = 0

        while not np.all(episode_done) and step < 10_000:
            log_probs, _ = self.agent.run_policy(state, jnp.array(obs))
            actions = self.agent.select_greedy_actions(log_probs)
            actions = jax.device_get(actions)
            obs, rewards, terminated, truncated, _ = self.eval_envs_wrapper.step_envs(actions)

            episode_rewards += rewards * (1 - episode_done)
            episode_done |= (terminated | truncated)
            step += 1
        return np.mean(episode_rewards)

    def _get_batch_train_samples(self, train_samples: TrainSamples) -> list[BatchSamples]:
        num_samples = self.cfg.steps_per_env * self.cfg.num_envs
        num_batchs = num_samples // self.cfg.batch_size
        permutation = self._rng.permutation(num_samples)
        indexes = permutation.reshape((num_batchs, self.cfg.batch_size))
        batches = []
        for i in range(num_batchs):
            batch_indexes = indexes[i]
            batch_sample = BatchSamples(
                obs=train_samples.obs[batch_indexes, ...],
                actions=train_samples.actions[batch_indexes],
                old_log_probs=train_samples.old_log_probs[batch_indexes],
                advantages=train_samples.advantages[batch_indexes],
                returns=train_samples.returns[batch_indexes],
            )
            batches.append(batch_sample)
        return batches

    def _get_train_samples(self, transitions: list[EnvsTransition]) -> TrainSamples:
        num_samples = self.cfg.steps_per_env * self.cfg.num_envs
        train_samples = TrainSamples(
            obs=np.zeros((num_samples,) + self.cfg.obs_shape, dtype=np.float32),
            actions=np.zeros((num_samples,), dtype=np.float32),
            old_log_probs=np.zeros((num_samples,), dtype=np.float32),
            advantages=np.zeros((num_samples,), dtype=np.float32),
            returns=np.zeros((num_samples,), dtype=np.float32),
        )
        index = 0
        
        for t in range(self.cfg.steps_per_env):
            start = index
            end = index + self.cfg.num_envs
            transition = transitions[t]
            train_samples.obs[start:end] = transition.obs
            train_samples.actions[start:end] = transition.actions
            train_samples.old_log_probs[start:end] = transition.log_probs
            train_samples.advantages[start:end] = transition.advantages
            train_samples.returns[start:end] = transition.returns
            index = end
        return train_samples

    def _collect_env_transitions(self, state: train_state.TrainState, obs: np.ndarray) -> tuple[np.ndarray, list[EnvsTransition]]:
        transitions = []
        # Collect the env steps
        for _ in range(self.cfg.steps_per_env + 1):
            self.rng_key, sample_rng_key = jax.random.split(self.rng_key)
            log_probs, values = self.agent.run_policy(state, jnp.array(obs))
            actions = self.agent.select_actions(log_probs, sample_rng_key)
            actions, log_probs, values = jax.device_get((actions, log_probs, values))
            next_obs, rewards, terminated, truncated, _ = self.envs_wrapper.step_envs(actions)
            log_probs = log_probs[np.arange(log_probs.shape[0]), actions]

            transition = EnvsTransition(
                obs=obs,
                actions=actions,
                rewards=np.sign(rewards),
                terminated=terminated,
                truncated=truncated,
                log_probs=log_probs,
                values=values,
            )
            transitions.append(transition)
            obs = next_obs

        # Compute the gae
        for t in reversed(range(self.cfg.steps_per_env)):
            transition = transitions[t]
            done = transition.terminated | transition.truncated
            values = transition.values[:, 0]
            next_values = transitions[t+1].values[:, 0]
            next_values = next_values * (1 - done)            
            td_error = transition.rewards + self.cfg.gamma * next_values - values

            if t == self.cfg.steps_per_env-1:
                transition.advantages = td_error
            else:
                advantages = td_error + self.cfg.gamma * self.cfg.gae_lambda * transitions[t+1].advantages * (1 - done)
                transition.advantages = advantages
            
            transition.returns = transition.advantages + values
        return obs, transitions
