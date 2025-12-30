import flwr as fl
import argparse
from clients.ppo_flower_client import PPOFlowerClient
import random

def main():
    parser = argparse.ArgumentParser(description="Flower PPO Client")
    parser.add_argument(
        "--server_address",
        type=str,
        default="127.0.0.1:8080", 
        help="Address of the Flower server.",
    )
    parser.add_argument(
        "--client_id",
        type=str,
        default=f"ppo_client_{random.randint(1000, 9999)}",
        help="Unique ID for this client.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="CartPole-v1", 
        help="Name of the OpenAI Gym environment to use.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2048, 
        help="Number of timesteps for local training per round.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=5,
        help="Number of episodes for local evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for the PPO agent and environment.",
    )
    args = parser.parse_args()

    print(f"Starting PPOFlowerClient {args.client_id} for environment {args.env_name}")
    print(f"Connecting to server at {args.server_address}")

   
    ppo_config = {
        "policy": "MlpPolicy", 
        "learning_rate": 3e-4,
        "n_steps": args.timesteps, 
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "verbose": 0,
    }
    
    if args.seed is not None:
        import numpy as np
        import torch
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)


    client = PPOFlowerClient(
        client_id=args.client_id,
        env_name=args.env_name,
        ppo_agent_config=ppo_config,
        local_training_timesteps=args.timesteps,
        local_eval_episodes=args.eval_episodes,
        seed=args.seed
    )

    fl.client.start_client(
        server_address=args.server_address,
        client=client.to_client(), 
    )
    print(f"PPOFlowerClient {args.client_id} finished.")

if __name__ == "__main__":
    main()