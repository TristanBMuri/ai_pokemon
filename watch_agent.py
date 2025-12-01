import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
from nuzlocke_gauntlet_rl.envs.real_battle_simulator import RealBattleSimulator

def watch_agent(model_name: str, battle_model_path: str, gauntlet_name: str = "kanto_leaders"):
    # Ensure models directory exists
    if not os.path.exists(f"models/{model_name}.zip"):
        print(f"Error: Model models/{model_name}.zip not found.")
        return

    print(f"Initializing RealBattleSimulator with model: {battle_model_path}...", flush=True)
    simulator = RealBattleSimulator(model_path=battle_model_path)
    
    print(f"Initializing NuzlockeGauntletEnv with gauntlet: {gauntlet_name} (Watch Mode)...", flush=True)
    # Enable Watch Mode
    env = NuzlockeGauntletEnv(simulator=simulator, gauntlet_name=gauntlet_name, watch_mode=True)
    
    # Load Model
    print(f"Loading manager model from models/{model_name}...", flush=True)
    model = PPO.load(f"models/{model_name}", env=env)
    
    print(f"\n{'='*60}")
    print(f"WATCH MODE STARTED")
    print(f"The agent will play 1 episode.")
    print(f"When a battle starts, a URL will appear below.")
    print(f"Click the URL to watch the battle in your browser.")
    print(f"{'='*60}\n", flush=True)
    
    obs, info = env.reset()
    done = False
    truncated = False
    episode_reward = 0
    
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
    # Episode finished
    progress = env.current_trainer_idx
    max_trainers = len(env.gauntlet_template.trainers)
    win = progress >= max_trainers
    survivors = len([m for m in env.roster if m.alive])
    
    print(f"\n{'='*60}")
    print(f"Episode Finished!")
    print(f"Result: {'VICTORY' if win else 'DEFEAT'}")
    print(f"Progress: Trainer {progress}/{max_trainers}")
    print(f"Survivors: {survivors}")
    print(f"Total Reward: {episode_reward:.2f}")
    print(f"{'='*60}\n", flush=True)
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch the Nuzlocke Manager Agent Live")
    parser.add_argument("--model_name", type=str, default="ppo_manager_v4", help="Name of the manager model to load")
    parser.add_argument("--battle_model", type=str, default="models/ppo_risk_agent_lstm_v1", help="Path to the trained battle agent model")
    parser.add_argument("--gauntlet", type=str, default="extended", help="Name of the gauntlet")
    
    args = parser.parse_args()
    
    watch_agent(args.model_name, args.battle_model, args.gauntlet)
