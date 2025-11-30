import asyncio
import os
import numpy as np
from stable_baselines3 import PPO
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from nuzlocke_gauntlet_rl.envs.battle_env import BattleEnv

import argparse

def train(steps, opponent_type, model_name):
    # Local Showdown Server Config
    from poke_env import LocalhostServerConfiguration
    # Try explicit config with None auth
    local_server_config = ServerConfiguration(
        "ws://192.168.1.122:8000/showdown/websocket",
        None
    )
    
    import uuid
    opponent_name = f"Opponent_{uuid.uuid4().hex[:8]}"
    agent_name = f"Agent_{uuid.uuid4().hex[:8]}"
    
    # Create opponent
    if opponent_type == "random":
        opponent = RandomPlayer(
            battle_format="gen8randombattle",
            server_configuration=local_server_config,
            account_configuration=AccountConfiguration(opponent_name, None),
        )
    elif opponent_type == "heuristic":
        opponent = SimpleHeuristicsPlayer(
            battle_format="gen8randombattle",
            server_configuration=local_server_config,
            account_configuration=AccountConfiguration(opponent_name, None),
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
    
    print(f"Training against {opponent_type} opponent ({opponent_name}) for {steps} steps...")
    
    # Create PettingZoo env
    pz_env = BattleEnv(
        battle_format="gen8randombattle",
        server_configuration=local_server_config,
        account_configuration1=AccountConfiguration(agent_name, None),
    )
    
    # Wrap to Gym env
    env = SingleAgentWrapper(pz_env, opponent=opponent)
    
    # Create log dir
    log_dir = "tmp/battler/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize agent
    # Check if model exists to continue training
    model_path = f"models/{model_name}"
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env, verbose=1, tensorboard_log=log_dir)
    else:
        print("Creating new PPO model")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # Train
    try:
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        print("Training finished!")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Battle Agent")
    parser.add_argument("--steps", type=int, default=100000, help="Total timesteps to train")
    parser.add_argument("--opponent", type=str, default="heuristic", choices=["random", "heuristic"], help="Opponent type")
    parser.add_argument("--model", type=str, default="ppo_risk_agent", help="Model name to save/load")
    
    args = parser.parse_args()
    
    train(args.steps, args.opponent, args.model)
