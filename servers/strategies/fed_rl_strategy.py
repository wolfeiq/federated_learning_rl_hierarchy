import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, NDArrays,
    ndarrays_to_parameters, parameters_to_ndarrays, Metrics, EvaluateRes, FitIns, EvaluateIns
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from logging import INFO, WARNING, DEBUG, ERROR
import datetime
from typing import List, Tuple, Optional, Dict, Union, Callable
import numpy as np


import gymnasium as gym 

try:
    from agents.ppo_agent import PPOAgent
except ImportError:
    PPOAgent = None
    log(ERROR, "FedRLStrategy: PPOAgent could not be imported. Centralized evaluation will not be possible.")


class FedRLStrategy(fl.server.strategy.Strategy):
    def __init__(
        self,
        fraction_fit: float = 1.0,
        min_fit_clients: int = 1,
        fraction_evaluate: float = 1.0, 
        min_evaluate_clients: int = 1,  
        min_available_clients: int = 1,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Metrics]]], Metrics]] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable[[List[Tuple[int, Metrics]]], Metrics]] = None, 
        central_eval_env_name: Optional[str] = None, 
        central_eval_ppo_config: Optional[Dict] = None, 
        central_eval_episodes: int = 10, 
    ):
        super().__init__()
        if not (0.0 <= fraction_fit <= 1.0):
            raise ValueError(f"fraction_fit must be between 0.0 and 1.0 (got: {fraction_fit})")

        self.fraction_fit = fraction_fit
        self.min_fit_clients = min_fit_clients
        self.fraction_evaluate = fraction_evaluate
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        
        self.last_aggregated_parameters: Optional[Parameters] = initial_parameters
        self.central_eval_env_name = central_eval_env_name
        self.central_eval_ppo_config = central_eval_ppo_config if central_eval_ppo_config is not None else {}
        self.central_eval_episodes = central_eval_episodes
        
        if self.central_eval_env_name and PPOAgent is None:
            log(WARNING, "FedRLStrategy: Centralized evaluation environment name provided, but PPOAgent is not available. Centralized evaluation will be skipped.")
            self.central_eval_env_name = None 

    def __repr__(self) -> str:
        return f"FedRLStrategy(min_fit_clients={self.min_fit_clients}, central_eval_env='{self.central_eval_env_name}')"

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        log(INFO, "FedRLStrategy: Initializing parameters...")
        if self.initial_parameters is not None:
            log(INFO, "FedRLStrategy: Using pre-configured initial parameters.")
            self.last_aggregated_parameters = self.initial_parameters
            return self.initial_parameters
        log(WARNING, "FedRLStrategy: No initial parameters provided. Waiting for first client or manual setting.")
        return None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if parameters is None and self.initial_parameters is None and server_round == 1:
            log(WARNING, "FedRLStrategy: No parameters available for fit configuration in round 1.")
        
        config = {"server_round": server_round, "current_time": str(datetime.datetime.now())}
        
        current_params_for_fit = parameters
        if current_params_for_fit is None:
            current_params_for_fit = self.last_aggregated_parameters if self.last_aggregated_parameters is not None else self.initial_parameters
        
        if current_params_for_fit is None:
             log(ERROR, "FedRLStrategy: CRITICAL - No parameters available to send to clients for fit.")
             return []

        fit_ins = FitIns(current_params_for_fit, config)
        sample_size = int(client_manager.num_available() * self.fraction_fit)
        num_clients_to_sample = max(self.min_fit_clients, sample_size)
        clients = client_manager.sample(num_clients=num_clients_to_sample, min_num_clients=self.min_fit_clients)
        
        if not clients:
            log(WARNING, f"FedRLStrategy: Not enough clients for fit (min {self.min_fit_clients}, sampled {len(clients)}). Skipping.")
            return []
            
        log(INFO, f"FedRLStrategy: Configuring fit for {len(clients)} clients in round {server_round}.")
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            log(WARNING, "FedRLStrategy: aggregate_fit received no results. Returning last known parameters.")
            return self.last_aggregated_parameters, {}

        weights_results = []
        total_timesteps_aggregated = 0
        for client_proxy, fit_res in results:
            if fit_res.status.code == fl.common.Code.OK and fit_res.parameters and fit_res.num_examples > 0:
                weights_results.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
                total_timesteps_aggregated += fit_res.num_examples
            else:
                log(WARNING, f"Client {client_proxy.cid}: FitRes issue. Status: {fit_res.status}. Metrics: {fit_res.metrics}")
        
        if not weights_results:
            log(WARNING, "FedRLStrategy: No valid results for aggregation in aggregate_fit.")
            return self.last_aggregated_parameters, {}

        log(INFO, f"FedRLStrategy: Aggregating weights from {len(weights_results)} clients, total timesteps: {total_timesteps_aggregated}.")
        
        ndarrays_list = [res[0] for res in weights_results]
        num_examples_list = [res[1] for res in weights_results]

        weighted_ndarrays = [
            layer * num_ex for layer_group, num_ex in zip(ndarrays_list, num_examples_list) for layer in layer_group
        ]
        
        aggregated_ndarrays: List[np.ndarray] = []
        for i in range(len(ndarrays_list[0])):
            layer_sum = sum(client_ndarrays[i] * num_ex for client_ndarrays, num_ex in zip(ndarrays_list, num_examples_list))
            aggregated_ndarrays.append(layer_sum / total_timesteps_aggregated)

        aggregated_parameters = ndarrays_to_parameters(aggregated_ndarrays)
        self.last_aggregated_parameters = aggregated_parameters

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results if res.status.code == fl.common.Code.OK and res.metrics]
            if fit_metrics:
                metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        
        log(INFO, f"FedRLStrategy: Fit aggregation complete for round {server_round}. Metrics: {metrics_aggregated}")
        return aggregated_parameters, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure federated evaluation (client-side)."""
        if not self.fraction_evaluate > 0.0: 
            log(INFO, "FedRLStrategy: fraction_evaluate is 0, skipping federated evaluation.")
            return []

        params_to_eval = parameters if parameters is not None else self.last_aggregated_parameters
        if params_to_eval is None:
             log(ERROR, "FedRLStrategy: CRITICAL - No parameters for federated evaluation configuration.")
             return[]

        config = {"server_round": server_round, "current_time": str(datetime.datetime.now())}
        eval_ins = EvaluateIns(params_to_eval, config)

        sample_size = int(client_manager.num_available() * self.fraction_evaluate)
        num_clients_to_sample = max(self.min_evaluate_clients, sample_size)
        clients = client_manager.sample(num_clients=num_clients_to_sample, min_num_clients=self.min_evaluate_clients)

        if not clients:
            log(WARNING, f"FedRLStrategy: Not enough clients for federated evaluate (min {self.min_evaluate_clients}). Skipping.")
            return []
            
        log(INFO, f"FedRLStrategy: Configuring federated evaluate for {len(clients)} clients in round {server_round}.")
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            log(WARNING, "FedRLStrategy: aggregate_evaluate received no results for federated evaluation.")
            return None, {}

        total_eval_examples = 0
        weighted_loss_sum = 0.0
        valid_results_count = 0

        for client_proxy, eval_res in results:
            if eval_res.status.code == fl.common.Code.OK and eval_res.loss is not None and eval_res.num_examples > 0:
                weighted_loss_sum += eval_res.loss * eval_res.num_examples
                total_eval_examples += eval_res.num_examples
                valid_results_count +=1
            else:
                 log(WARNING, f"Client {client_proxy.cid}: EvaluateRes issue. Status: {eval_res.status}, Loss: {eval_res.loss}, NumEx: {eval_res.num_examples}")

        if valid_results_count == 0 or total_eval_examples == 0 :
            log(WARNING, "FedRLStrategy: No valid results for federated evaluation aggregation.")
            return None, {}

        aggregated_loss = weighted_loss_sum / total_eval_examples
        log(INFO, f"FedRLStrategy: Aggregated federated evaluation loss: {aggregated_loss:.4f} from {total_eval_examples} examples across {valid_results_count} clients.")

        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results if res.status.code == fl.common.Code.OK and res.metrics]
            if eval_metrics:
                metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        
        log(INFO, f"FedRLStrategy: Federated evaluate aggregation complete for round {server_round}. Metrics: {metrics_aggregated}")
        return aggregated_loss, metrics_aggregated

 
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
 
        if PPOAgent is None or self.central_eval_env_name is None:
            log(WARNING, f"FedRLStrategy: Centralized server-side 'evaluate' skipped for round {server_round} (PPOAgent or env_name not available).")
            return None 
        
        log(INFO, f"FedRLStrategy: Starting centralized server-side evaluation for round {server_round} on env '{self.central_eval_env_name}'.")

        try:

            ndarrays = parameters_to_ndarrays(parameters)
            eval_agent_id = f"global_server_eval_agent_round_{server_round}"

            default_ppo_config = {"policy": "MlpPolicy", "verbose": 0} 
            ppo_config_for_eval = {**default_ppo_config, **self.central_eval_ppo_config}

            eval_agent = PPOAgent(
                env_name=self.central_eval_env_name, 
                agent_config=ppo_config_for_eval,

            )
            log(INFO, f"FedRLStrategy: Centralized evaluation agent '{eval_agent_id}' created.")

            eval_agent.set_model_parameters(ndarrays)
            log(DEBUG, "FedRLStrategy: Parameters set for centralized evaluation agent.")

    
            eval_metrics_from_agent = eval_agent.evaluate(n_eval_episodes=self.central_eval_episodes)
            log(INFO, f"FedRLStrategy: Centralized evaluation by PPOAgent completed. Raw metrics: {eval_metrics_from_agent}")

            mean_reward = eval_metrics_from_agent.get("mean_reward_eval", 0.0)
            loss = -float(mean_reward) 
            metrics_to_return: Dict[str, Scalar] = {
                "central_eval_mean_reward": float(mean_reward),
                "central_eval_episodes": self.central_eval_episodes,
            }

            for key, value in eval_metrics_from_agent.items():
                if key not in ["mean_reward_eval"] and isinstance(value, (int, float, np.floating, np.integer)):
                    metrics_to_return[f"central_eval_{key}"] = float(value)
            
            log(INFO, f"FedRLStrategy: Centralized server-side evaluation complete for round {server_round}. Loss: {loss}, Metrics: {metrics_to_return}")
            return loss, metrics_to_return

        except Exception as e:
            log(ERROR, f"FedRLStrategy: Error during centralized server-side evaluation: {e}", exc_info=True)
            return None 
