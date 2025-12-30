# Hierarchical Federated Reinforcement Learning Working Prototype with PPO Agents

The system has a three-tier hierarchy:

1. Global Server at the top level
2. Local Servers that act as both servers and clients
3. PPO Clients which are actual RL agents

# How It Works

1. PPO Agents: agents/ppo_agent.py
   
Trains on Gymnasium environments (CartPole-v1)
Can both get and set model parameters for federated aggregation

2. PPO Flower Clients: clients/ppo_flower_client.py

Converts PPO agents into Flower federated learning clients

3. Local Servers: local_server_agent/

Acts as both a server & a client:
Receives global model, returns aggregated local updates
Manages multiple PPO clients
Aggregates PPO client metrics, such as rewards, losses & timesteps

4. Global Server: servers/global_server_h.py

Coordinates multiple local servers
Uses custom FedRLStrategy for aggregation
Performs centralized evaluation on the global model

5. Custom FL Strategy: servers/strategies/fed_rl_strategy.py
   
Supports both:
Federated evaluation: Clients evaluate locally
Centralized evaluation: Server evaluates global model
Aggregates RL-specific metrics (rewards, episode lengths)


This file servers/one_agent_rl.py is for a RL scenario in Holoocean (Unreal Engine). It doesn't work properly and I've rewritten the script ever since.
