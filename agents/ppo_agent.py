import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback # For more control if needed
from typing import Dict, Any, Tuple, List, Optional
from torch import nn

class PPOAgent:
    def __init__(self, env_name: str, agent_config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        self.env_name = env_name
        self.agent_config = agent_config if agent_config else {}
        self.seed = seed
        self.env = DummyVecEnv([lambda: gym.make(self.env_name)])

        ppo_defaults = {
            "policy": "MlpPolicy",
            "learning_rate": 3e-4,
            "n_steps": 2048, 
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "tensorboard_log": None, 
            "verbose": 0,
        }
        ppo_defaults.update(self.agent_config)

        self.model = PPO(
            env=self.env,
            seed=self.seed,
            **ppo_defaults
        )
        print(f"PPO Agent initialized for {self.env_name} with policy: {self.model.policy}")

    def get_model_parameters(self) -> List[np.ndarray]:
        if self.model.policy is None:
            raise ValueError("Model policy is not initialized.")
    
        params = []
        for param_tensor in self.model.policy.state_dict().values():
            params.append(param_tensor.cpu().numpy())
        return params

    def set_model_parameters(self, parameters: List[np.ndarray]):
        """Sets model parameters from a list of NumPy arrays."""
        if self.model.policy is None:
            raise ValueError("Model policy is not initialized.")

        params_dict = self.model.policy.state_dict()
        # Ensure the number of parameter groups matches
        if len(parameters) != len(params_dict):
            raise ValueError(f"Mismatched number of parameter tensors: got {len(parameters)}, expected {len(params_dict)}")

        new_state_dict = {}
        for i, (name, current_tensor) in enumerate(params_dict.items()):
            param_np = parameters[i]
            # Ensure shapes match
            if current_tensor.shape != param_np.shape:
                raise ValueError(f"Shape mismatch for parameter {name}: current {current_tensor.shape}, new {param_np.shape}")
            new_state_dict[name] = torch.from_numpy(param_np).to(self.model.device)
        
        self.model.policy.load_state_dict(new_state_dict)
        print("PPO Agent parameters updated.")

    def train(self, total_timesteps: int) -> Dict[str, float]:

        if self.model.policy is None:
            raise ValueError("Model policy is not initialized.")

        self.model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False, 
        )

        ep_info_buffer = self.model.ep_info_buffer
        mean_reward = safe_mean([ep_info['r'] for ep_info in ep_info_buffer]) if ep_info_buffer else 0.0
        mean_ep_length = safe_mean([ep_info['l'] for ep_info in ep_info_buffer]) if ep_info_buffer else 0.0

        metrics = {
            "mean_reward_local_train": mean_reward,
            "mean_ep_length_local_train": mean_ep_length,
            "timesteps_trained_this_round": total_timesteps,
        }
        print(f"PPO Agent trained for {total_timesteps} steps. Metrics: {metrics}")
        return metrics


    def evaluate(self, n_eval_episodes: int = 10) -> Dict[str, float]:
        if self.model.policy is None:
            raise ValueError("Model policy is not initialized.")

        eval_env = DummyVecEnv([lambda: gym.make(self.env_name)])
        
        episode_rewards = []
        episode_lengths = []

        for _ in range(n_eval_episodes):
            obs = eval_env.reset()
            done = False
            current_episode_reward = 0.0
            current_episode_length = 0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = eval_env.step(action)
                current_episode_reward += reward[0] 
                current_episode_length += 1
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)
        
        eval_env.close()

        metrics = {
            "mean_reward_eval": np.mean(episode_rewards) if episode_rewards else 0.0,
            "std_reward_eval": np.std(episode_rewards) if episode_rewards else 0.0,
            "mean_ep_length_eval": np.mean(episode_lengths) if episode_lengths else 0.0,
        }
        print(f"PPO Agent evaluation. Metrics: {metrics}")
        return metrics

    def get_policy_architecture(self) -> Dict[str, Any]:
        if self.model.policy:
            return {"policy_type": str(type(self.model.policy)), "state_dict_keys": list(self.model.policy.state_dict().keys())}
        return {}