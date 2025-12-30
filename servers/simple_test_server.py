import flwr as fl
import numpy as np
from logging import INFO, WARNING, ERROR
try:
    from agents.ppo_agent import PPOAgent
except ImportError:

    import sys
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    try:
        from agents.ppo_agent import PPOAgent
    except ImportError as e:
        PPOAgent = None 
        fl.common.logger.log(ERROR, f"Failed to import PPOAgent: {e}. "
                                   "Ensure the 'agents' module is accessible. "
                                   "The server cannot determine initial parameters without it.")


def get_initial_ppo_parameters(env_name="CartPole-v1", ppo_config_override=None):
    """
    Initializes a temporary PPO agent to extract its model parameters.
    This defines the model structure for FedAvg.
    """
    if PPOAgent is None:
        fl.common.logger.log(ERROR, "PPOAgent class not available. Cannot determine initial parameters.")
        return None

    fl.common.logger.log(INFO, f"Attempting to initialize a temporary PPO agent for env '{env_name}' to get initial parameter structure...")
    

    temp_agent_config = {
        "policy": "MlpPolicy",
        "n_steps": 256, 
        "batch_size": 64,
        "n_epochs": 1,
        "verbose": 0,
    }
    if ppo_config_override:
        temp_agent_config.update(ppo_config_override)

    try:

        import gymnasium as gym
        try:
            gym.make(env_name).close()
        except Exception as e:
            fl.common.logger.log(ERROR, f"Failed to create Gym environment '{env_name}': {e}. "
                                       "Please ensure the environment is installed and registered.")
            return None


        temp_agent = PPOAgent(env_name=env_name, agent_config=temp_agent_config)
        initial_ndarrays = temp_agent.get_model_parameters()
        fl.common.logger.log(INFO, f"Successfully retrieved initial PPO parameters structure with {len(initial_ndarrays)} layers.")
        return fl.common.ndarrays_to_parameters(initial_ndarrays)
    except Exception as e:
        fl.common.logger.log(ERROR, f"Error getting initial PPO parameters using PPOAgent for env '{env_name}': {e}")
        fl.common.logger.log(ERROR, "This is critical for FedAvg. The server might not function correctly.")
        return None


def main():
    server_address = "0.0.0.0:8080" 
    num_federated_rounds = 5

    reference_env_name = "CartPole-v1"

    fraction_fit = 1.0
    min_fit_clients = 2 
    fraction_evaluate = 1.0
    min_evaluate_clients = 2 
    min_available_clients = 2 

    initial_parameters = get_initial_ppo_parameters(env_name=reference_env_name)

    if initial_parameters is None:
        fl.common.logger.log(WARNING,
            "Could not determine initial PPO model parameters. "
            "FedAvg might fail or behave unexpectedly. "
            "Ensure PPOAgent can be initialized and the environment is correct."
        )


    # Strategy: FedAvg

    def aggregate_rl_metrics(metrics: list[tuple[int, dict]]) -> dict:
        """Helper function to aggregate PPO-specific metrics."""
        aggregated = {}
        # Metrics from PPOFlowerClient: 'mean_reward_local_train', 'mean_ep_length_local_train',
        # 'mean_reward_eval', 'std_reward_eval', 'mean_ep_length_eval'
        # Also 'fit_duration', 'eval_duration'.

        for num_examples, client_metrics in metrics:
            for key, value in client_metrics.items():
                if isinstance(value, (int, float)):
                    if key not in aggregated:
                        aggregated[key] = {'sum': 0.0, 'total_examples': 0}
                    aggregated[key]['sum'] += value * num_examples
                    aggregated[key]['total_examples'] += num_examples

        final_metrics = {}
        for key, data in aggregated.items():
            if data['total_examples'] > 0:
                final_metrics[f"avg_weighted_{key}"] = data['sum'] / data['total_examples']
        return final_metrics

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        fraction_evaluate=fraction_evaluate,
        min_evaluate_clients=min_evaluate_clients,
        min_available_clients=min_available_clients,
        initial_parameters=initial_parameters, 
        fit_metrics_aggregation_fn=aggregate_rl_metrics,
        evaluate_metrics_aggregation_fn=aggregate_rl_metrics,
    )

    fl.common.logger.log(INFO, f"Starting Federated Averaging (FedAvg) server on {server_address}")
    fl.common.logger.log(INFO, f"Number of rounds: {num_federated_rounds}")
    if initial_parameters:
        fl.common.logger.log(INFO, "Initial model parameters have been set for the strategy.")
    else:
        fl.common.logger.log(WARNING, "Proceeding without pre-set initial model parameters in FedAvg strategy.")


    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_federated_rounds),
        strategy=strategy,
    )

    fl.common.logger.log(INFO, "Federated learning server finished.")

if __name__ == "__main__":
    main()