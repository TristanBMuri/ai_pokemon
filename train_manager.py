
import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
from nuzlocke_gauntlet_rl.envs.real_battle_simulator import RealBattleSimulator
from nuzlocke_gauntlet_rl.envs.mock_battle_simulator import MockBattleSimulator

def make_env(rank: int, seed: int, battle_model_path: str, use_mock: bool, gauntlet_name: str):
    """
    Utility function for multiprocessed env.
    
    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    :param battle_model_path: path to the battle agent model
    :param use_mock: whether to use mock simulator
    :param gauntlet_name: name of the gauntlet
    """
    def _init():
        if use_mock:
            # print(f"Rank {rank}: Initializing MockBattleSimulator...", flush=True)
            simulator = MockBattleSimulator()
        else:
            # print(f"Rank {rank}: Initializing RealBattleSimulator...", flush=True)
            # Force CPU for battle agent to avoid CUDA conflicts in subprocesses
            simulator = RealBattleSimulator(model_path=battle_model_path, device="cpu")
            
        env = NuzlockeGauntletEnv(simulator=simulator, gauntlet_name=gauntlet_name)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train_manager(steps: int, model_name: str, battle_model_path: str, use_mock: bool = False, gauntlet_name: str = "kanto_leaders", n_envs: int = 1, n_steps_per_update: int = 2048):
    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    
    print(f"Initializing {n_envs} environments (Mock={use_mock}, Gauntlet={gauntlet_name})...", flush=True)
    
    # Create vectorized environment
    env_fns = [make_env(i, 42, battle_model_path, use_mock, gauntlet_name) for i in range(n_envs)]
    
    if n_envs > 1:
        # Use spawn to avoid fork issues with poke-env/asyncio
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    
    # Initialize Manager Agent
    # We use PPO because the action space is MultiDiscrete
    print(f"Initializing PPO Manager Agent (n_steps={n_steps_per_update})...", flush=True)
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./tmp/manager/", n_steps=n_steps_per_update)
    
    # Check if model exists to resume
    model_path = f"models/{model_name}"
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing manager model from {model_path}...", flush=True)
        # Note: Loading model with different env might require custom handling if env changed
        # But PPO.load handles it if env is passed
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
    parser.add_argument("--model_name", type=str, default="ppo_manager_v4", help="Name of the model to save/load")
    parser.add_argument("--battle_model", type=str, default="models/ppo_risk_agent_v3", help="Path to the trained battle agent model")
    parser.add_argument("--mock", action="store_true", help="Use MockBattleSimulator instead of RealBattleSimulator")
    parser.add_argument("--gauntlet", type=str, default="extended", help="Gauntlet to train on")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--n_steps_per_update", type=int, default=2048, help="PPO n_steps (buffer size per env)")
    
    args = parser.parse_args()
    
    train_manager(args.steps, args.model_name, args.battle_model, args.mock, args.gauntlet, args.n_envs, args.n_steps_per_update)
