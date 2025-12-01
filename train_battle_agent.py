import argparse
import os
import numpy as np
import uuid
from stable_baselines3 import PPO
from poke_env.player import RandomPlayer
from poke_env import ServerConfiguration, AccountConfiguration
from nuzlocke_gauntlet_rl.envs.battle_env import BattleEnv
from nuzlocke_gauntlet_rl.wrappers.single_agent_battle_wrapper import MySingleAgentWrapper

def train_battle_agent(steps: int, model_name: str, server_url: str = "ws://192.168.1.122:8000/showdown/websocket"):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    server_config = ServerConfiguration(server_url, None)
    
    # Create unique IDs
    agent_name = f"TrainAgent_{uuid.uuid4().hex[:8]}"
    opponent_name = f"TrainOpp_{uuid.uuid4().hex[:8]}"
    
    print(f"Initializing Battle Agent Training (Agent: {agent_name}, Opponent: {opponent_name})...", flush=True)
    
    # Initialize Opponent
    # Initialize Opponent
    from poke_env.player import SimpleHeuristicsPlayer
    opponent = SimpleHeuristicsPlayer(
        battle_format="gen9customgame",
        server_configuration=server_config,
        account_configuration=AccountConfiguration(opponent_name, None),
    )
    
    # Initialize BattleEnv
    pz_env = BattleEnv(
        battle_format="gen9customgame",
        server_configuration=server_config,
        account_configuration1=AccountConfiguration(agent_name, None),
    )
    
    # Wrap for Gym
    env = MySingleAgentWrapper(pz_env, opponent=opponent)
    
    # Challenge Trigger Thread
    import threading
    import time
    import asyncio
    
    def challenge_loop():
        print("Challenge loop started.", flush=True)
        # Wait for login
        time.sleep(5)
        
        while True:
            # Check if we need to trigger a challenge
            # We can just try to challenge periodically.
            # If a battle is active, it might be ignored or queued.
            # But we need to be careful not to spam too much.
            
            # Ideally we check if pz_env is reset and waiting.
            # But we don't have easy access to that state from here safely.
            
            # Just try to challenge every 10 seconds?
            # Or better: check if opponent is battling.
            
            try:
                # We need a new loop for the thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def trigger():
                    if not opponent.ps_client.logged_in.is_set():
                        await opponent.ps_client.logged_in.wait()
                        
                    # Check if opponent has active battles
                    # This is tricky to check from outside.
                    # But we can just send a challenge.
                    # print("Sending challenge...", flush=True)
                    await opponent.battle_against(pz_env.agent1, n_battles=1)
                    
                loop.run_until_complete(trigger())
                loop.close()
                
            except Exception as e:
                print(f"Challenge trigger error: {e}", flush=True)
                
            # Wait a bit before next check/challenge
            # A battle takes some time.
            time.sleep(5)
            
    t = threading.Thread(target=challenge_loop, daemon=True)
    t.start()
    
    # Initialize Agent
    print(f"Initializing RecurrentPPO Battle Agent...", flush=True)
    try:
        from sb3_contrib import RecurrentPPO
    except ImportError:
        print("Error: sb3-contrib not installed. Please install it to use RecurrentPPO.")
        return

    model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log="./tmp/battle_agent/")
    
    # Check if model exists to resume
    model_path = f"models/{model_name}"
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing battle model from {model_path}...", flush=True)
        model = RecurrentPPO.load(model_path, env=env)
    
    print(f"Starting training for {steps} steps...", flush=True)
    try:
        model.learn(total_timesteps=steps, progress_bar=True)
        print("Training complete.", flush=True)
    except KeyboardInterrupt:
        print("Training interrupted.", flush=True)
    finally:
        print(f"Saving battle model to {model_path}...", flush=True)
        model.save(model_path)
        print("Model saved.", flush=True)
        
        # Close environment
        # env.close() # Player doesn't have close

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Nuzlocke Battle Agent")
    parser.add_argument("--steps", type=int, default=10000, help="Number of training steps")
    parser.add_argument("--model", type=str, default="ppo_risk_agent_lstm_v1", help="Name of the battle agent model")
    
    args = parser.parse_args()
    
    train_battle_agent(args.steps, args.model)
