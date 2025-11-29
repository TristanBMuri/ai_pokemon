import asyncio
import os
import numpy as np
from stable_baselines3 import PPO
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from nuzlocke_gauntlet_rl.envs.battle_env import BattleEnv

def train():
    # Local Showdown Server Config
    from poke_env import LocalhostServerConfiguration
    # local_server_config = LocalhostServerConfiguration
    # Try explicit config with None auth
    local_server_config = ServerConfiguration(
        "ws://192.168.1.122:8000/showdown/websocket",
        None
    )
    
    import uuid
    opponent_name = f"Opponent_{uuid.uuid4().hex[:8]}"
    agent_name = f"Agent_{uuid.uuid4().hex[:8]}"
    # opponent_name = "Opponent"
    # agent_name = "Agent"
    
    # Create opponent
    opponent = RandomPlayer(
        battle_format="gen8randombattle",
        server_configuration=local_server_config,
        account_configuration=AccountConfiguration(opponent_name, None),
    )
    
    # Create PettingZoo env
    pz_env = BattleEnv(
        battle_format="gen8randombattle",
        server_configuration=local_server_config,
        account_configuration1=AccountConfiguration(agent_name, None),
    )
    
    # Wrap to Gym env
    env = SingleAgentWrapper(pz_env, opponent=opponent)
    print(f"Observation Space Type: {type(env.observation_space)}")
    print(f"Observation Space: {env.observation_space}")
    
    # Create log dir
    log_dir = "tmp/battler/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize agent
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # Train
    print("Starting training...")
    try:
        model.learn(total_timesteps=4096)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        print("Training finished!")
        
        # Save model
        os.makedirs("models", exist_ok=True)
        model.save("models/ppo_risk_agent")
        print("Model saved to models/ppo_risk_agent")
        
        env.close()

if __name__ == "__main__":
    train()
