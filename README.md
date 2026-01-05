# Hierarchical Federated Reinforcement Learning Working Prototype with PPO Agents

The system has a three-tier hierarchy:

1. Global Server at the top level
2. Local Servers that act as both servers and clients
3. PPO Clients which are actual RL agents

# How It Works

1. PPO Agents: agents/ppo_agent.py
   
- Trains on Gymnasium environments (CartPole-v1)
- Can both get and set model parameters for federated aggregation

2. PPO Flower Clients: clients/ppo_flower_client.py

- Converts PPO agents into Flower federated learning clients

3. Local Servers: local_server_agent/

- Acts as both a server & a client:
- Receives global model, returns aggregated local updates
- Manages multiple PPO clients
- Aggregates PPO client metrics, such as rewards, losses & timesteps

4. Global Server: servers/global_server_h.py

- Coordinates multiple local servers
- Uses custom FedRLStrategy for aggregation
- Performs centralized evaluation on the global model

5. Custom FL Strategy: servers/strategies/fed_rl_strategy.py
   
- Supports both:
- Federated evaluation: Clients evaluate locally
- Centralized evaluation: Server evaluates global model
- Aggregates RL-specific metrics (rewards, episode lengths)


This file servers/one_agent_rl.py is for a RL scenario in Holoocean (Unreal Engine). It doesn't work properly and I've rewritten the script ever since.


```mermaid
graph TD
    %% Styling
    classDef global fill:#2c3e50,stroke:#34495e,stroke-width:2px,color:white;
    classDef local fill:#2980b9,stroke:#3498db,stroke-width:2px,color:white;
    classDef client fill:#27ae60,stroke:#2ecc71,stroke-width:2px,color:white;
    classDef env fill:#e67e22,stroke:#d35400,stroke-width:2px,color:white;
    classDef strategy fill:#8e44ad,stroke:#9b59b6,stroke-width:2px,color:white,stroke-dasharray: 5 5;

    subgraph Tier_1_Global_Level [Tier 1: Global Orchestration]
        direction TB
        GS[Global Server<br/>servers/global_server_h.py]:::global
        Strat[FedRLStrategy<br/>servers/strategies/fed_rl_strategy.py]:::strategy
        
        GS -- Uses for Aggregation<br/>& Centralized Eval --> Strat
    end

    subgraph Tier_2_Local_Level [Tier 2: Intermediate Aggregation]
        LS1[Local Server 1<br/>local_server_agent/]:::local
        LS2[Local Server N...]:::local
        
        noteLS[<b>Dual Role:</b><br/>Acts as Client to Global<br/>Acts as Server to PPO Clients]
        
        LS1 -.- noteLS
    end

    subgraph Tier_3_Client_Level [Tier 3: PPO RL Agents]
        direction TB
        
        subgraph PPO_Instance [PPO Client Instance]
            FC[PPO Flower Client<br/>clients/ppo_flower_client.py]:::client
            Agent[PPO Agent<br/>agents/ppo_agent.py]:::client
            Gym((Gymnasium Env<br/>CartPole-v1)):::env
            
            FC -- Wraps --> Agent
            Agent -- Actions --> Gym
            Gym -- Rewards/Obs --> Agent
        end
    end

    %% Data Flow Connections
    GS -- Global Model Parameters --> LS1
    GS -- Global Model Parameters --> LS2
    
    LS1 -- Aggregated Local Updates<br/>& Metrics --> GS
    LS2 -- Aggregated Local Updates --> GS

    LS1 -- Local Model Parameters --> FC
    FC -- Gradients/Weights<br/>& RL Metrics --> LS1

    %% Legend - Fixed Line Below
    linkStyle default stroke-width:2px,fill:none,stroke:#bdc3c7;
