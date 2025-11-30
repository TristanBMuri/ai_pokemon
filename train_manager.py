
import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
from nuzlocke_gauntlet_rl.envs.real_battle_simulator import RealBattleSimulator
from nuzlocke_gauntlet_rl.envs.mock_battle_simulator import MockBattleSimulator

def train_manager(steps: int, model_name: str, battle_model_path: str, use_mock: bool = False):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    if use_mock:
        print("Initializing MockBattleSimulator...", flush=True)
        simulator = MockBattleSimulator()
    else:
        print(f"Initializing RealBattleSimulator with model: {battle_model_path}...", flush=True)
        simulator = RealBattleSimulator(model_path=battle_model_path)
    
    print("Initializing NuzlockeGauntletEnv...", flush=True)
    env = NuzlockeGauntletEnv(simulator=simulator)
    
    # Initialize Manager Agent
    # We use PPO because the action space is MultiDiscrete
    print(f"Initializing PPO Manager Agent...", flush=True)
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tmp/manager/")
    
    # Check if model exists to resume
    model_path = f"models/{model_name}"
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing manager model from {model_path}...", flush=True)
        model = PPO.load(model_path, env=env)
    
    print(f"Starting training for {steps} steps...", flush=True)
    try:
        model.learn(total_timesteps=steps, progress_bar=True)
        print("Training complete.", flush=True)
    except KeyboardInterrupt:
        print("Training interrupted.", flush=True)
    finally:
        print(f"Saving manager model to {model_path}...", flush=True)
        model.save(model_path)
        print("Model saved.", flush=True)
        
        # Close environment
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Nuzlocke Manager Agent")
    parser.add_argument("--steps", type=int, default=1000, help="Number of training steps")
    parser.add_argument("--model", type=str, default="ppo_manager_v1", help="Name of the manager model")
    parser.add_argument("--battle_model", type=str, default="models/ppo_risk_agent_v1", help="Path to the trained battle agent model")
    parser.add_argument("--mock", action="store_true", help="Use MockBattleSimulator instead of RealBattleSimulator")
    
    args = parser.parse_args()
    
    train_manager(args.steps, args.model, args.battle_model, args.mock)
