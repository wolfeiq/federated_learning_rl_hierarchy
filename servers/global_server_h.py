import flwr as fl
from flwr.common.logger import log
from logging import INFO, WARNING, ERROR, DEBUG 
import sys
import os
from typing import List, Dict, Optional, Tuple


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from agents.ppo_agent import PPOAgent 
    ppo_agent_available = True
except ImportError as e:
    log(ERROR, f"GlobalServer: Failed to import PPOAgent: {e}. Initial parameters and centralized eval might fail.")
    PPOAgent = None 
    ppo_agent_available = False

try:
    from servers.strategies.fed_rl_strategy import FedRLStrategy 
    fed_rl_strategy_available = True
except ImportError as e:
    log(ERROR, f"GlobalServer: Failed to import FedRLStrategy: {e}. Server cannot start with this strategy.")
    FedRLStrategy = None 
    fed_rl_strategy_available = False


def get_initial_ppo_parameters(env_name="CartPole-v1", ppo_config_override: Optional[Dict] = None) -> Optional[fl.common.Parameters]:
    if PPOAgent is None:
        log(ERROR, "GlobalServer: PPOAgent class not available. Cannot determine initial parameters.")
        return None
    log(INFO, f"GlobalServer: Initializing temp PPO agent for env '{env_name}' to get initial parameters.")
    
    temp_agent_config = {"policy": "MlpPolicy", "n_steps": 256, "batch_size": 64, "n_epochs": 1, "verbose": 0}
    if ppo_config_override: 
        temp_agent_config.update(ppo_config_override)
    
    try:
        import gymnasium as gym 
        try:
            env = gym.make(env_name)
            env.close()
        except Exception as e_gym:
            log(ERROR, f"GlobalServer: Failed to create Gym env '{env_name}' for initial params: {e_gym}")
            return None
            
        temp_agent = PPOAgent(env_name=env_name, agent_config=temp_agent_config)
        initial_ndarrays = temp_agent.get_model_parameters()
        log(INFO, f"GlobalServer: Retrieved initial PPO parameters structure ({len(initial_ndarrays)} layers).")
        return fl.common.ndarrays_to_parameters(initial_ndarrays)
    except Exception as e:
        log(ERROR, f"GlobalServer: Error getting initial PPO parameters: {e}", exc_info=True)
        return None
    
def evaluate_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Dict]]) -> Dict:
    
    if not all_client_metrics:
        log(WARNING, "GlobalServer (evaluate_metrics_aggregation_fn): Received empty metrics list for federated eval.")
        return {}

    aggregated_metrics = {}
    local_eval_rewards = []
    total_eval_episodes_globally = 0
    num_reporting_clients = 0

    for num_examples, metrics in all_client_metrics:
        if metrics: 
            num_reporting_clients +=1
            
            reward_key = "avg_actual_reward_local_eval" 
            if reward_key in metrics:
                try:
                    local_eval_rewards.append(float(metrics[reward_key]))
                except (ValueError, TypeError):
                    log(WARNING, f"GlobalServer (evaluate_metrics_aggregation_fn): Could not convert {reward_key} value '{metrics[reward_key]}'")
        

            episodes_key = "total_eval_episodes_in_local_sim"
            if episodes_key in metrics:
                try:
                    total_eval_episodes_globally += int(metrics[episodes_key])
                except (ValueError, TypeError):
                     log(WARNING, f"GlobalServer (evaluate_metrics_aggregation_fn): Could not convert {episodes_key} value '{metrics[episodes_key]}'")
            
    if local_eval_rewards:
        aggregated_metrics["global_avg_of_local_eval_rewards"] = sum(local_eval_rewards) / len(local_eval_rewards)
    else:
        aggregated_metrics["global_avg_of_local_eval_rewards"] = 0.0 

    aggregated_metrics["global_total_federated_eval_episodes"] = total_eval_episodes_globally
    aggregated_metrics["global_num_locals_reported_federated_eval"] = num_reporting_clients
    
    log(INFO, f"GlobalServer (evaluate_metrics_aggregation_fn): Aggregated federated eval metrics: {aggregated_metrics}")
    return aggregated_metrics


def fit_metrics_aggregation_fn(all_client_metrics: List[Tuple[int, Dict]]) -> Dict:
    if not all_client_metrics:
        log(WARNING, "GlobalServer (fit_metrics_aggregation_fn): Received empty metrics list for fit.")
        return {}

    aggregated_metrics = {}
    local_fit_sim_rewards = []
    local_fit_losses = [] 
    total_fit_timesteps_globally = 0
    num_reporting_clients = 0

    for num_examples, metrics in all_client_metrics:
        if metrics:
            num_reporting_clients += 1
            reward_key = "avg_ppo_reward_in_local_sim" 
            if reward_key in metrics:
                try:
                    local_fit_sim_rewards.append(float(metrics[reward_key]))
                except (ValueError, TypeError):
                    log(WARNING, f"GlobalServer (fit_metrics_aggregation_fn): Could not convert {reward_key} value '{metrics[reward_key]}'")

            loss_key = "final_loss_local_fit"
            if loss_key in metrics:
                try:
                    local_fit_losses.append(float(metrics[loss_key]))
                except (ValueError, TypeError):
                    log(WARNING, f"GlobalServer (fit_metrics_aggregation_fn): Could not convert {loss_key} value '{metrics[loss_key]}'")
            
            timesteps_key = "total_timesteps_in_local_sim_fit"
            if timesteps_key in metrics:
                try:
                    total_fit_timesteps_globally += int(metrics[timesteps_key])
                except (ValueError, TypeError):
                    log(WARNING, f"GlobalServer (fit_metrics_aggregation_fn): Could not convert {timesteps_key} value '{metrics[timesteps_key]}'")

    if local_fit_sim_rewards:
        aggregated_metrics["global_avg_of_local_sim_fit_rewards"] = sum(local_fit_sim_rewards) / len(local_fit_sim_rewards)
    else:
        aggregated_metrics["global_avg_of_local_sim_fit_rewards"] = 0.0

    if local_fit_losses:
        aggregated_metrics["global_avg_of_local_fit_losses"] = sum(local_fit_losses) / len(local_fit_losses)
    else:
        aggregated_metrics["global_avg_of_local_fit_losses"] = 0.0
        
    aggregated_metrics["global_total_fit_timesteps_across_locals"] = total_fit_timesteps_globally
    aggregated_metrics["global_num_locals_reported_fit"] = num_reporting_clients

    log(INFO, f"GlobalServer (fit_metrics_aggregation_fn): Aggregated fit metrics: {aggregated_metrics}")
    return aggregated_metrics

def main():
    if not fed_rl_strategy_available or FedRLStrategy is None:
        log(ERROR, "GlobalServer: FedRLStrategy class not loaded. Exiting.")
        return
    if not ppo_agent_available and PPOAgent is None:
        log(WARNING, "GlobalServer: PPOAgent class not available. Initial parameters might be missing, and centralized evaluation will be skipped if configured.")

    log(INFO, "Starting Global Server for Hierarchical FL.")
    
    global_server_address = "0.0.0.0:9090" 
    num_global_rounds = 20 
    

    central_eval_env_name_on_server = "CartPole-v1" 
    central_eval_episodes_on_server = 20 

    server_eval_ppo_config = {
        "policy": "MlpPolicy",
        "verbose": 0,

    }

    expected_local_servers = 2 
    
    initial_params = get_initial_ppo_parameters(env_name=central_eval_env_name_on_server)
    if initial_params is None:
        log(WARNING, "GlobalServer: Could not determine initial model parameters. Strategy might fail if not set by first client.")

    log(INFO, f"GlobalServer: PPOAgent available for strategy: {PPOAgent is not None}")
    log(INFO, f"GlobalServer: Central evaluation env name for strategy: {central_eval_env_name_on_server}")

    global_strategy = FedRLStrategy(
        initial_parameters=initial_params,
        min_fit_clients=expected_local_servers,      
        min_available_clients=expected_local_servers, 
        min_evaluate_clients=expected_local_servers, 
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, 
    
        central_eval_env_name=central_eval_env_name_on_server,
        central_eval_ppo_config=server_eval_ppo_config,
        central_eval_episodes=central_eval_episodes_on_server
    )

    log(INFO, f"Global Server will run for {num_global_rounds} rounds.")
    log(INFO, f"Expecting {expected_local_servers} local server(s) to connect.")
    log(INFO, f"Centralized server-side evaluation is configured for env: '{central_eval_env_name_on_server}'")

    fl.server.start_server(
        server_address=global_server_address,
        config=fl.server.ServerConfig(num_rounds=num_global_rounds),
        strategy=global_strategy,
    )
    log(INFO, "Global Server finished.")

if __name__ == "__main__":
    main()
