import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
from nuzlocke_gauntlet_rl.envs.real_battle_simulator import RealBattleSimulator
from nuzlocke_gauntlet_rl.envs.mock_battle_simulator import MockBattleSimulator

def evaluate_manager(episodes: int, model_name: str, battle_model_path: str, use_mock: bool = False, gauntlet_name: str = "kanto_leaders"):
    # Ensure models directory exists
    if not os.path.exists(f"models/{model_name}.zip"):
        print(f"Error: Model models/{model_name}.zip not found.")
        return

    if use_mock:
        print("Initializing MockBattleSimulator...", flush=True)
        simulator = MockBattleSimulator()
    else:
        print(f"Initializing RealBattleSimulator with model: {battle_model_path}...", flush=True)
        simulator = RealBattleSimulator(model_path=battle_model_path)
    
    print(f"Initializing NuzlockeGauntletEnv with gauntlet: {gauntlet_name}...", flush=True)
    env = NuzlockeGauntletEnv(simulator=simulator, gauntlet_name=gauntlet_name)
    
    # Load Model
    print(f"Loading manager model from models/{model_name}...", flush=True)
    model = PPO.load(f"models/{model_name}", env=env)
    
    print(f"Starting evaluation for {episodes} episodes...", flush=True)
    
    wins = 0
    total_progress = 0
    total_survivors = 0
    max_trainers = len(env.gauntlet_template.trainers)
    
    for ep in range(episodes):
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
        win = progress >= max_trainers
        survivors = len([m for m in env.roster if m.alive])
        
        if win:
            wins += 1
            
        total_progress += progress
        total_survivors += survivors
        
        print(f"Episode {ep+1}/{episodes}: Win={win}, Progress={progress}/{max_trainers}, Survivors={survivors}, Reward={episode_reward:.2f}", flush=True)
        
    # Report
    win_rate = (wins / episodes) * 100
    avg_progress = total_progress / episodes
    avg_survivors = total_survivors / episodes
    
    print("\nEvaluation Results:")
    print(f"Gauntlet: {gauntlet_name}")
    print(f"Episodes: {episodes}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Avg Progress: {avg_progress:.1f} / {max_trainers}")
    print(f"Avg Survivors: {avg_survivors:.1f}")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Nuzlocke Manager Agent")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--model", type=str, default="ppo_manager_v2", help="Name of the manager model")
    parser.add_argument("--battle_model", type=str, default="models/ppo_risk_agent_v3", help="Path to the trained battle agent model")
    parser.add_argument("--mock", action="store_true", help="Use MockBattleSimulator instead of RealBattleSimulator")
    parser.add_argument("--gauntlet", type=str, default="kanto_leaders", help="Name of the gauntlet (kanto_leaders, indigo_league, team_rocket)")
    
    args = parser.parse_args()
    
    evaluate_manager(args.episodes, args.model, args.battle_model, args.mock, args.gauntlet)
