# hfrl_project/clients/ppo_flower_client.py
import flwr as fl
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import time

from agents.ppo_agent import PPOAgent

def sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64, np.floating)):
            sanitized[key] = float(value)
        elif isinstance(value, (np.int32, np.int64, np.integer)):
            sanitized[key] = int(value)
        # elif isinstance(value, np.ndarray):
            # sanitized[key] = value.tolist()
        else:
            sanitized[key] = value
    return sanitized

class PPOFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: str, env_name: str, ppo_agent_config: Optional[Dict] = None,
                 local_training_timesteps: int = 2048, local_eval_episodes: int = 5, seed: Optional[int]=None):
        self.client_id = client_id
        self.env_name = env_name
        self.local_training_timesteps = local_training_timesteps
        self.local_eval_episodes = local_eval_episodes
        
        self.agent = PPOAgent(env_name=env_name, agent_config=ppo_agent_config, seed=seed)
        print(f"Client {self.client_id}: PPO agent initialized for {self.env_name}.")

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        print(f"Client {self.client_id}: Sending parameters to server.")
        return self.agent.get_model_parameters()

    def set_parameters(self, parameters: List[np.ndarray]):
        print(f"Client {self.client_id}: Receiving parameters from server.")
        self.agent.set_model_parameters(parameters)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        print(f"Client {self.client_id}: Starting local training (fit). Config: {config}")
        start_time = time.time()

        self.set_parameters(parameters) 

        local_timesteps_this_round = self.local_training_timesteps
        training_metrics_from_agent = self.agent.train(total_timesteps=local_timesteps_this_round)

        updated_parameters = self.agent.get_model_parameters()
        num_samples_or_timesteps = training_metrics_from_agent.get("timesteps_trained_this_round", local_timesteps_this_round)

        fit_duration = time.time() - start_time
        metrics_to_return = {
            "client_id": self.client_id,
            "fit_duration": float(fit_duration),
            **training_metrics_from_agent
        }
        
        sanitized_final_metrics = sanitize_metrics(metrics_to_return)
        
        print(f"Client {self.client_id}: Local training finished in {fit_duration:.2f}s. Metrics: {sanitized_final_metrics}")
        
        return updated_parameters, int(num_samples_or_timesteps), sanitized_final_metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        print(f"Client {self.client_id}: Starting local evaluation (evaluate). Config: {config}")
        start_time = time.time()

        self.set_parameters(parameters)

        eval_metrics_from_agent = self.agent.evaluate(n_eval_episodes=self.local_eval_episodes)

        raw_loss = -eval_metrics_from_agent.get("mean_reward_eval", 0.0)
        loss = float(raw_loss) 
        
        num_eval_examples = self.local_eval_episodes
        
        eval_duration = time.time() - start_time
        metrics_to_return = {
            "client_id": self.client_id,
            "eval_duration": float(eval_duration),
            **eval_metrics_from_agent
        }
        
        sanitized_final_metrics = sanitize_metrics(metrics_to_return)

        print(f"Client {self.client_id}: Local evaluation finished in {eval_duration:.2f}s. Loss: {loss}, Metrics: {sanitized_final_metrics}")

        return loss, num_eval_examples, sanitized_final_metrics