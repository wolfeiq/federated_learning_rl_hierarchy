import flwr as fl
import argparse
import sys
import os
from flwr.common.logger import log
from logging import INFO, WARNING, DEBUG, ERROR
import datetime # For using datetime.datetime.now()


import flwr as fl
from flwr.common.logger import log
from logging import INFO, ERROR # Import levels from standard logging
import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

LocalServerAsClient = None
try:
    from local_server_agent.local_server_app import LocalServerAsClient
except ImportError as e:
    
    log(ERROR, f"Failed to import LocalServerAsClient: {e}. Ensure 'local_server_agent' is a package (has __init__.py) and project root is in PYTHONPATH or handled by sys.path correctly.")
    
    sys.exit("Critical import error, cannot proceed.")


def main():
    if LocalServerAsClient is None:
        log(ERROR, "LocalServerAsClient class not imported. Exiting.")
        return

    parser = argparse.ArgumentParser(description="Flower Local Server Agent")
    parser.add_argument("--local_server_id", type=str, required=True, help="Unique ID for this local server")
    parser.add_argument("--global_server_address", type=str, default="[::]:9090", help="Address of the Global Flower server.")
    
    parser.add_argument("--num_ppo_clients", type=int, default=2, help="Number of PPO clients this local server will manage per round.")
    parser.add_argument("--ppo_env", type=str, default="CartPole-v1", help="Gym environment for PPO clients.")
    parser.add_argument("--ppo_timesteps", type=int, default=1024, help="Timesteps PPO clients train per local round.")
    parser.add_argument("--ppo_eval_episodes", type=int, default=5, help="Number of episodes PPO clients run for their local evaluation.")
    parser.add_argument("--local_rounds", type=int, default=1, help="Number of local aggregation rounds per global fit call.")


    args = parser.parse_args()

    log(INFO, f"Starting Local Server Agent: {args.local_server_id}")
    log(INFO, f"Connecting to Global Server at: {args.global_server_address}")
    log(INFO, f"Will manage {args.num_ppo_clients} PPO clients for env {args.ppo_env} ({args.ppo_timesteps} ts) over {args.local_rounds} local round(s).")

    client_instance = LocalServerAsClient(
        local_server_id=args.local_server_id,
        num_ppo_clients_per_round=args.num_ppo_clients,
        ppo_client_env_name=args.ppo_env,
        ppo_client_timesteps=args.ppo_timesteps,
        ppo_client_eval_episodes=args.ppo_eval_episodes,
        local_aggregation_rounds=args.local_rounds
    )

    fl.client.start_client(
        server_address=args.global_server_address,
        client=client_instance.to_client(),
    )
    log(INFO, f"Local Server Agent {args.local_server_id} finished.")

if __name__ == "__main__":
    main()