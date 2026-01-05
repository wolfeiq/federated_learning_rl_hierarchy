import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, EvaluateRes, FitRes 
from flwr.common.logger import log
from logging import INFO, WARNING, ERROR, DEBUG
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional, Callable
import collections.abc

project_root_for_main_process = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root_for_main_process not in sys.path:
    sys.path.insert(0, project_root_for_main_process)

PPOFlowerClient = None
FedRLStrategy_Module = None 
try:
    from clients.ppo_flower_client import PPOFlowerClient
    from servers.strategies.fed_rl_strategy import FedRLStrategy as FedRLStrategy_Module 
except ImportError as e:
    log(ERROR, f"LocalServerAsClient (in local_server_app.py): Failed to import critical modules: {e}")


def aggregate_ppo_client_fit_metrics(all_ppo_client_metrics: List[Tuple[int, Dict]]) -> Dict:
    aggregated_local_metrics = {
        "avg_reward_from_ppo_clients": 0.0, 
        "num_ppo_clients_reported_valid_reward_metric": 0
    } 
    try:
        if not all_ppo_client_metrics:
            log(WARNING, "LocalSimAggregator (FIT): Received empty list of PPO client metrics.")
            return aggregated_local_metrics
        
        ppo_client_reward_key = "mean_reward_local_train" 

        rewards = []
        clients_with_reward_key = 0
        total_clients_reporting_any_metrics = 0

        log(INFO, f"LocalSimAggregator (FIT): Processing {len(all_ppo_client_metrics)} PPO client metric sets. Expecting reward key: '{ppo_client_reward_key}'")

        for i, (num_examples, client_metrics_obj) in enumerate(all_ppo_client_metrics):
            log(DEBUG, f"LocalSimAggregator (FIT) - Client Metric Set {i}: Type={type(client_metrics_obj)}, Content={client_metrics_obj}")
            
            if client_metrics_obj and isinstance(client_metrics_obj, collections.abc.Mapping): 
                total_clients_reporting_any_metrics += 1
                key_present = ppo_client_reward_key in client_metrics_obj 
                log(DEBUG, f"LocalSimAggregator (FIT) - Client {client_metrics_obj.get('client_id', 'Unknown')}: Checking for key '{ppo_client_reward_key}'. Present: {key_present}. Available keys: {list(client_metrics_obj.keys())}")

                if key_present:
                    try:
                        reward_value = client_metrics_obj[ppo_client_reward_key]
                        rewards.append(float(reward_value))
                        clients_with_reward_key += 1
                        log(DEBUG, f"LocalSimAggregator (FIT): Successfully processed reward {reward_value} for client {client_metrics_obj.get('client_id', 'UnknownPPOClient')}")
                    except (ValueError, TypeError) as e:
                        log(WARNING, f"LocalSimAggregator (FIT): Could not convert reward value '{client_metrics_obj.get(ppo_client_reward_key)}' to float for key '{ppo_client_reward_key}'. Error: {e}")
            else:
                log(WARNING, f"LocalSimAggregator (FIT): Received None or non-Mapping client_metrics_obj at index {i}: {client_metrics_obj}")
                
        if rewards: 
            aggregated_local_metrics["avg_reward_from_ppo_clients"] = sum(rewards) / len(rewards)
        elif total_clients_reporting_any_metrics > 0:
            log(WARNING, f"LocalSimAggregator (FIT): No rewards found under key '{ppo_client_reward_key}' from any of the {total_clients_reporting_any_metrics} PPO clients that reported metrics.")
        else:
            log(WARNING, "LocalSimAggregator (FIT): No PPO clients reported any valid metrics dictionary-like object.")
        
        aggregated_local_metrics["num_ppo_clients_reported_valid_reward_metric"] = clients_with_reward_key
        aggregated_local_metrics["total_ppo_clients_provided_metrics_obj"] = total_clients_reporting_any_metrics
        log(INFO, f"LocalSimAggregator (FIT): Aggregated PPO client fit metrics: {aggregated_local_metrics}")

    except Exception as e:
        log(ERROR, f"LocalSimAggregator (FIT): CRITICAL ERROR during metric aggregation: {e}", exc_info=True)
    return aggregated_local_metrics

def aggregate_ppo_client_evaluate_metrics(all_ppo_client_metrics: List[Tuple[int, Dict]]) -> Dict:
    aggregated_local_metrics = {
        "avg_eval_reward_from_ppo_clients": 0.0, 
        "num_ppo_clients_reported_valid_eval_reward_metric": 0
    }
    try:
        if not all_ppo_client_metrics:
            log(WARNING, "LocalSimAggregator (EVAL): Received empty list of PPO client metrics.")
            return aggregated_local_metrics
        
        ppo_client_eval_reward_key = "mean_reward_eval"

        eval_rewards = []
        clients_with_eval_reward_key = 0
        total_clients_reporting_any_metrics_eval = 0

        log(INFO, f"LocalSimAggregator (EVAL): Processing {len(all_ppo_client_metrics)} PPO client metric sets. Expecting key: '{ppo_client_eval_reward_key}'")

        for i, (num_examples, client_metrics_obj) in enumerate(all_ppo_client_metrics): 
            log(DEBUG, f"LocalSimAggregator (EVAL) - Client Metric Set {i}: Type={type(client_metrics_obj)}, Content={client_metrics_obj}")
            
            if client_metrics_obj and isinstance(client_metrics_obj, collections.abc.Mapping):
                total_clients_reporting_any_metrics_eval +=1
                key_present = ppo_client_eval_reward_key in client_metrics_obj
                log(DEBUG, f"LocalSimAggregator (EVAL) - Client {client_metrics_obj.get('client_id', 'Unknown')}: Checking for key '{ppo_client_eval_reward_key}'. Present: {key_present}. Available keys: {list(client_metrics_obj.keys())}")
                
                if key_present:
                    try:
                        reward_value = client_metrics_obj[ppo_client_eval_reward_key]
                        eval_rewards.append(float(reward_value))
                        clients_with_eval_reward_key += 1
                        log(DEBUG, f"LocalSimAggregator (EVAL): Successfully processed eval reward {reward_value} for client {client_metrics_obj.get('client_id', 'UnknownPPOClient')}")
                    except (ValueError, TypeError) as e:
                        log(WARNING, f"LocalSimAggregator (EVAL): Could not convert eval reward value '{client_metrics_obj.get(ppo_client_eval_reward_key)}' to float for key '{ppo_client_eval_reward_key}'. Error: {e}")
            else:
                log(WARNING, f"LocalSimAggregator (EVAL): Received None or non-Mapping client_metrics_obj at index {i}: {client_metrics_obj}")

        if eval_rewards:
            aggregated_local_metrics["avg_eval_reward_from_ppo_clients"] = sum(eval_rewards) / len(eval_rewards)
        elif total_clients_reporting_any_metrics_eval > 0 :
            log(WARNING, f"LocalSimAggregator (EVAL): No eval rewards found under key '{ppo_client_eval_reward_key}' from any of the {total_clients_reporting_any_metrics_eval} PPO clients that reported metrics.")
        else:
            log(WARNING, "LocalSimAggregator (EVAL): No PPO clients reported any valid metrics dictionary-like object for evaluation.")

        aggregated_local_metrics["num_ppo_clients_reported_valid_eval_reward_metric"] = clients_with_eval_reward_key
        aggregated_local_metrics["total_ppo_clients_provided_metrics_obj_eval"] = total_clients_reporting_any_metrics_eval
        log(INFO, f"LocalSimAggregator (EVAL): Aggregated PPO client eval metrics: {aggregated_local_metrics}")

    except Exception as e:
        log(ERROR, f"LocalSimAggregator (EVAL): CRITICAL ERROR during metric aggregation: {e}", exc_info=True)
    return aggregated_local_metrics


class LocalServerAsClient(fl.client.NumPyClient):
    def __init__(self,
                 local_server_id: str,
                 num_ppo_clients_per_round: int,
                 ppo_client_env_name: str,
                 ppo_client_timesteps: int, 
                 ppo_client_eval_episodes: int = 5, 
                 local_aggregation_rounds: int = 1
                ):
        if PPOFlowerClient is None or FedRLStrategy_Module is None:
            log(ERROR, "LocalServerAsClient __init__: Critical modules not loaded.")
            raise ImportError("LocalServerAsClient: Critical modules not loaded.")

        self.local_server_id = local_server_id
        self.num_ppo_clients_per_round = num_ppo_clients_per_round
        self.ppo_client_env_name = ppo_client_env_name
        self.ppo_client_timesteps = ppo_client_timesteps 
        self.ppo_client_eval_episodes = ppo_client_eval_episodes 
        self.local_aggregation_rounds = local_aggregation_rounds
        self.current_model_parameters: Optional[List[np.ndarray]] = None

        self.local_strategy = FedRLStrategy_Module( 
            min_fit_clients=self.num_ppo_clients_per_round,
            min_available_clients=self.num_ppo_clients_per_round,
            fraction_fit=1.0,
            min_evaluate_clients=self.num_ppo_clients_per_round, 
            fraction_evaluate=1.0, 
            fit_metrics_aggregation_fn=aggregate_ppo_client_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_ppo_client_evaluate_metrics,
        )
        log(INFO, f"LocalServerAsClient [{self.local_server_id}] initialized with custom local metric aggregators.")

    def _get_ppo_client_fn_for_simulation(self, for_evaluation: bool = False):
        def client_fn(cid: str) -> fl.client.Client: 
            client_id = f"{self.local_server_id}_ppo_client_{cid}_{'eval' if for_evaluation else 'fit'}"
            log(DEBUG, f"LocalServer [{self.local_server_id}]: Spawning PPO Client {client_id}")
            ppo_config = {"n_steps": self.ppo_client_timesteps, "batch_size": 64, "n_epochs": 10, "verbose": 0}
            try:
                client = PPOFlowerClient(
                    client_id=client_id,
                    env_name=self.ppo_client_env_name,
                    ppo_agent_config=ppo_config,
                    local_training_timesteps=self.ppo_client_timesteps, 
                    local_eval_episodes=self.ppo_client_eval_episodes  
                )
                return client.to_client()
            except Exception as e:
                log(ERROR, f"LocalServerAsClient [{self.local_server_id}]: Failed to create PPOFlowerClient instance {client_id}. Error: {e}", exc_info=True)
                raise 
        return client_fn

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        log(INFO, f"LocalServerAsClient [{self.local_server_id}]: GlobalServer requesting parameters.")
        if self.current_model_parameters is not None:
            return self.current_model_parameters
        else:
            log(WARNING, f"LocalServerAsClient [{self.local_server_id}]: No model parameters for get_parameters. Returning empty list.")
            return [] 

    def set_parameters(self, parameters: List[np.ndarray]):
        log(INFO, f"LocalServerAsClient [{self.local_server_id}]: Received {len(parameters)} Global Server parameter arrays.")
        try:
            self.current_model_parameters = parameters
            if self.local_strategy:
                 self.local_strategy.initial_parameters = ndarrays_to_parameters(parameters)
        except Exception as e:
            log(ERROR, f"LocalServerAsClient [{self.local_server_id}]: Error setting parameters. Error: {e}", exc_info=True)

    def fit(self, parameters: List[np.ndarray], config: Dict) -> Tuple[List[np.ndarray], int, Dict]:
        log(INFO, f"LocalServerAsClient [{self.local_server_id}]: FIT call from Global Server (Round {config.get('server_round', 'N/A')}).")
        metrics_to_global_server = {"client_id": self.local_server_id, "fit_status": "started"}
        num_aggregated_examples = 0 
        current_params_to_return = parameters 
        
        try:
            self.set_parameters(parameters) 
            current_params_to_return = self.current_model_parameters 

            if self.local_strategy.initial_parameters is None: 
                log(WARNING, f"LocalServerAsClient [{self.local_server_id}]: local_strategy.initial_parameters is still None at start of fit. Setting it now.")
                self.local_strategy.initial_parameters = ndarrays_to_parameters(parameters)

            ppo_client_fn = self._get_ppo_client_fn_for_simulation(for_evaluation=False)
            log(INFO, f"LocalServerAsClient [{self.local_server_id}]: Starting local FIT simulation with {self.num_ppo_clients_per_round} PPO clients for {self.local_aggregation_rounds} local round(s).")
            hfrl_project_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            ray_init_args_for_sim = {
                "runtime_env": {"working_dir": hfrl_project_abs_path, "env_vars": {"PYTHONPATH": f"{hfrl_project_abs_path}:{os.environ.get('PYTHONPATH', '')}"}},
                "ignore_reinit_error": True
            }
            
            history = fl.simulation.start_simulation(
                client_fn=ppo_client_fn,
                num_clients=self.num_ppo_clients_per_round,
                client_resources={"num_cpus": 1, "num_gpus": 0.0}, 
                config=fl.server.ServerConfig(num_rounds=self.local_aggregation_rounds),
                strategy=self.local_strategy,
                ray_init_args=ray_init_args_for_sim
            )
            log(INFO, f"LocalServerAsClient [{self.local_server_id}]: Local FIT simulation finished.")
            metrics_to_global_server["fit_simulation_completed"] = True

            if getattr(self.local_strategy, "last_aggregated_parameters", None):
                locally_aggregated_params_ndarrays = parameters_to_ndarrays(self.local_strategy.last_aggregated_parameters)
                self.current_model_parameters = locally_aggregated_params_ndarrays
                current_params_to_return = locally_aggregated_params_ndarrays 
                log(INFO, f"LocalServerAsClient [{self.local_server_id}]: Successfully got locally aggregated parameters.")
            else:
                log(WARNING, f"LocalServerAsClient [{self.local_server_id}]: 'last_aggregated_parameters' not found on local_strategy after FIT. Using parameters received from global server for return.")

            num_aggregated_examples = self.num_ppo_clients_per_round * self.ppo_client_timesteps * self.local_aggregation_rounds
            metrics_to_global_server["total_timesteps_in_local_sim_fit"] = num_aggregated_examples

            if hasattr(history, "losses_distributed") and history.losses_distributed:
                if history.losses_distributed[-1] and isinstance(history.losses_distributed[-1], tuple) and len(history.losses_distributed[-1]) > 1:
                    metrics_to_global_server["final_loss_local_fit"] = float(history.losses_distributed[-1][1])
                else:
                    log(WARNING, f"LocalServerAsClient FIT: losses_distributed[-1] is not as expected: {history.losses_distributed[-1]}")

            fit_metrics_content = getattr(history, "metrics_distributed_fit", None)
            log(INFO, f"LocalServerAsClient [{self.local_server_id}]: FIT - Attempting to access history.metrics_distributed_fit. Found: {fit_metrics_content is not None}")

            if fit_metrics_content and isinstance(fit_metrics_content, collections.abc.Mapping) and len(fit_metrics_content) > 0: 
                log(INFO, f"LocalServerAsClient [{self.local_server_id}]: FIT history.metrics_distributed_fit found and is a non-empty mapping. Content: {fit_metrics_content}")
                fit_reward_key_from_local_agg = "avg_reward_from_ppo_clients" 
                
                if fit_reward_key_from_local_agg in fit_metrics_content:
                    reward_val_list = fit_metrics_content[fit_reward_key_from_local_agg]
                    if reward_val_list and isinstance(reward_val_list[-1], tuple) and len(reward_val_list[-1]) == 2:
                        last_round_reward = reward_val_list[-1][1] 
                        metrics_to_global_server["avg_ppo_reward_in_local_sim"] = float(last_round_reward)
                        log(INFO, f"LocalServerAsClient [{self.local_server_id}]: Extracted '{fit_reward_key_from_local_agg}' = {last_round_reward} as 'avg_ppo_reward_in_local_sim' from history.metrics_distributed_fit.")
                    else:
                        log(WARNING, f"LocalServerAsClient FIT: Metric list for '{fit_reward_key_from_local_agg}' in history.metrics_distributed_fit is empty or its last element is malformed: {reward_val_list}")
                else:
                    log.warning(f"LocalServerAsClient FIT: Key '{fit_reward_key_from_local_agg}' NOT FOUND in history.metrics_distributed_fit. Available keys: {list(fit_metrics_content.keys())}")
            else: 
                log(WARNING, f"LocalServerAsClient [{self.local_server_id}]: Attribute 'metrics_distributed_fit' was not found or was empty/None in FIT history. History object: {history}")
            
            metrics_to_global_server["fit_status"] = "completed_successfully"
            log(INFO, f"LocalServerAsClient [{self.local_server_id}]: FINAL FIT METRICS TO GLOBAL SERVER: {metrics_to_global_server}")
            return current_params_to_return, num_aggregated_examples, metrics_to_global_server

        except Exception as e:
            log(ERROR, f"LocalServerAsClient [{self.local_server_id}]: CRITICAL ERROR in FIT method: {e}", exc_info=True)
            metrics_to_global_server["fit_status"] = "failed"
            metrics_to_global_server["error_message"] = str(e)
            return self.current_model_parameters if self.current_model_parameters is not None else parameters, 0, metrics_to_global_server


    def evaluate(self, parameters: List[np.ndarray], config: Dict) -> Tuple[float, int, Dict]:
        log(INFO, f"LocalServerAsClient [{self.local_server_id}]: EVALUATE call from Global Server (Round {config.get('server_round', 'N/A')}).")
        metrics_to_global_server = {"client_id": self.local_server_id, "eval_status": "started"}
        final_loss_from_local_eval = 1.0 
        total_examples_from_local_eval = 0

        try:
            self.set_parameters(parameters) 
            if self.local_strategy.initial_parameters is None:
                self.local_strategy.initial_parameters = ndarrays_to_parameters(parameters)
            
            ppo_client_fn_eval = self._get_ppo_client_fn_for_simulation(for_evaluation=True)
            log(INFO, f"LocalServerAsClient [{self.local_server_id}]: Starting local EVALUATION simulation.")
            hfrl_project_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            ray_init_args_for_sim = {
                "runtime_env": {"working_dir": hfrl_project_abs_path, "env_vars": {"PYTHONPATH": f"{hfrl_project_abs_path}:{os.environ.get('PYTHONPATH', '')}"}},
                "ignore_reinit_error": True
            }

            history = fl.simulation.start_simulation(
                client_fn=ppo_client_fn_eval,
                num_clients=self.num_ppo_clients_per_round,
                client_resources={"num_cpus": 1, "num_gpus": 0.0},
                config=fl.server.ServerConfig(num_rounds=1), 
                strategy=self.local_strategy, 
                ray_init_args=ray_init_args_for_sim
            )
            log(INFO, f"LocalServerAsClient [{self.local_server_id}]: Local EVALUATION simulation finished.")
            metrics_to_global_server["eval_simulation_completed"] = True
            
            if hasattr(history, "losses_distributed") and history.losses_distributed:
                if history.losses_distributed[0] and isinstance(history.losses_distributed[0], tuple) and len(history.losses_distributed[0]) > 1:
                    final_loss_from_local_eval = float(history.losses_distributed[0][1]) 
                else:
                    log(WARNING, f"LocalServerAsClient EVAL: losses_distributed[0] is not as expected: {history.losses_distributed[0]}")
            
            total_examples_from_local_eval = self.num_ppo_clients_per_round * self.ppo_client_eval_episodes
            metrics_to_global_server["total_eval_episodes_in_local_sim"] = total_examples_from_local_eval
            metrics_to_global_server["loss_reported_to_global"] = final_loss_from_local_eval 
            
            eval_metrics_content = getattr(history, "metrics_distributed", None) 
            log(INFO, f"LocalServerAsClient [{self.local_server_id}]: EVAL - Attempting to access history.metrics_distributed. Found: {eval_metrics_content is not None}")

            if eval_metrics_content and isinstance(eval_metrics_content, collections.abc.Mapping) and len(eval_metrics_content) > 0:
                log(INFO, f"LocalServerAsClient [{self.local_server_id}]: EVAL history.metrics_distributed found and is a non-empty mapping. Content: {eval_metrics_content}")
                eval_reward_key_from_local_agg = "avg_eval_reward_from_ppo_clients"
                
                if eval_reward_key_from_local_agg in eval_metrics_content:
                    reward_val_list = eval_metrics_content[eval_reward_key_from_local_agg]
                    
                    if reward_val_list and isinstance(reward_val_list[0], tuple) and len(reward_val_list[0]) == 2:
                        
                        actual_eval_reward = reward_val_list[0][1] 
                        metrics_to_global_server["avg_actual_reward_local_eval"] = float(actual_eval_reward)
                        log(INFO, f"LocalServerAsClient [{self.local_server_id}]: Extracted '{eval_reward_key_from_local_agg}' = {actual_eval_reward} as 'avg_actual_reward_local_eval' from history.metrics_distributed.")
                    else:
                        log(WARNING, f"LocalServerAsClient EVAL: Metric list for '{eval_reward_key_from_local_agg}' in history.metrics_distributed is empty or its first element is malformed: {reward_val_list}")
                else:
                    log.warning(f"LocalServerAsClient EVAL: Key '{eval_reward_key_from_local_agg}' NOT FOUND in history.metrics_distributed. Available keys: {list(eval_metrics_content.keys())}")
            else: 
                log(WARNING, f"LocalServerAsClient [{self.local_server_id}]: Attribute 'metrics_distributed' was not found or was empty/None in EVALUATE history. History object: {history}")

            metrics_to_global_server["eval_status"] = "completed_successfully"
            log(INFO, f"LocalServerAsClient [{self.local_server_id}]: FINAL EVALUATE METRICS TO GLOBAL SERVER: {metrics_to_global_server}")
            return final_loss_from_local_eval, total_examples_from_local_eval, metrics_to_global_server

        except Exception as e:
            log(ERROR, f"LocalServerAsClient [{self.local_server_id}]: CRITICAL ERROR in EVALUATE method: {e}", exc_info=True)
            metrics_to_global_server["eval_status"] = "failed"
            metrics_to_global_server["error_message"] = str(e)
            return final_loss_from_local_eval, total_examples_from_local_eval, metrics_to_global_server
