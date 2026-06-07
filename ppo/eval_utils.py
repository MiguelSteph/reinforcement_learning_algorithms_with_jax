import jax
import numpy as np
from agent import Agent
from envs_wrapper import EnvsWrapper
from flax.training import train_state
from jax import numpy as jnp

def run_eval(agent: Agent, 
             state: train_state.TrainState, 
             num_envs: int,
             terminal_on_life_loss: bool = False,
             video_folder: str | None = None):
    eval_envs = EnvsWrapper(num_envs=num_envs,
                            terminal_on_life_loss=terminal_on_life_loss,
                            video_folder=video_folder)
    obs, _ = eval_envs.reset_envs()
    episode_rewards = np.zeros(num_envs, dtype=np.float32)
    episode_done = np.zeros(num_envs, dtype=bool)
    step = 0

    while not np.all(episode_done) and step < 20_000:
        log_probs, _ = agent.run_policy(state, jnp.array(obs))
        actions = agent.select_greedy_actions(log_probs)
        actions = jax.device_get(actions)
        obs, rewards, terminated, truncated, _ = eval_envs.step_envs(actions)

        episode_rewards += rewards * (1 - episode_done)
        episode_done |= (terminated | truncated)
        step += 1
    eval_envs.close()
    
    print("EVALUATION RESULT")
    print(f"    - Mean sum rewards: {np.mean(episode_rewards)}")
    print(f"    - Min sum rewards: {np.min(episode_rewards)}")
    print(f"    - Max sum rewards: {np.max(episode_rewards)}")
