
import argparse
import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from nuzlocke_gauntlet_rl.callbacks.metrics_callback import MetricsCallback
from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
from nuzlocke_gauntlet_rl.envs.real_battle_simulator import RealBattleSimulator
from nuzlocke_gauntlet_rl.envs.mock_battle_simulator import MockBattleSimulator

def make_env(rank: int, seed: int, battle_model_path: str, use_mock: bool, gauntlet_name: str, n_servers: int = 4):
    def _init():
        # Distribute across n_servers (ports 8000 to 8000+n-1)
        port = 8000 + (rank % n_servers)
        ws_url = f"ws://localhost:{port}/showdown/websocket"
        
        env = NuzlockeGauntletEnv(
            gauntlet_name=gauntlet_name,
            model_path=battle_model_path,
            simulator_url=ws_url
            # max_roster_size=400,
            # watch_mode=False
        )
        
        # Wrap with ActionMasker for MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        env = ActionMasker(env, lambda env: env.valid_action_mask())
        
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

def train_manager(steps: int, model_name: str, battle_model_path: str, use_mock: bool = False, gauntlet_name: str = "kanto_leaders", n_envs: int = 1, n_steps_per_update: int = 2048, learning_rate: float = 3e-4, batch_size: int = 64, ent_coef: float = 0.0):
    os.makedirs("models", exist_ok=True)
    
    print(f"Initializing {n_envs} environments (Gauntlet={gauntlet_name}, MaskablePPO)...", flush=True)
    
    # Check dependencies
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        print("Error: sb3-contrib is required for Action Masking. Please install it.")
        sys.exit(1)
        
    env_fns = [make_env(i, 42, battle_model_path, use_mock, gauntlet_name) for i in range(n_envs)]
    
    if n_envs > 1:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    
    print(f"Initializing MaskablePPO Manager Agent (lr={learning_rate}, batch={batch_size}, ent={ent_coef})...", flush=True)
    model = MaskablePPO(
        "MultiInputPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./tmp/manager/", 
        n_steps=n_steps_per_update,
        learning_rate=learning_rate,
        batch_size=batch_size,
        ent_coef=ent_coef
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=f"./models/{model_name}_checkpoints", name_prefix=model_name)
    metrics_callback = MetricsCallback()
    
    from nuzlocke_gauntlet_rl.callbacks.rich_dashboard import RichDashboardCallback
    dashboard_callback = RichDashboardCallback()
    
    callbacks = [checkpoint_callback, metrics_callback, dashboard_callback]
    
    model_path = f"models/{model_name}"
    if os.path.exists(f"{model_path}.zip"):
        print(f"Loading existing manager model from {model_path}...", flush=True)
        model = MaskablePPO.load(model_path, env=env)
    
    print(f"Starting training for {steps} steps...", flush=True)
    try:
        model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False) # False bar because using Rich Dashboard
        print("Training complete.", flush=True)
    except KeyboardInterrupt:
        print("Training interrupted.", flush=True)
    finally:
        print(f"Saving manager model to {model_path}...", flush=True)
        model.save(model_path)
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
    
    # Hyperparams
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning Rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")
    parser.add_argument("--ent_coef", type=float, default=0.0, help="Entropy Coefficient")
    
    args = parser.parse_args()
    
    train_manager(args.steps, args.model_name, args.battle_model, args.mock, args.gauntlet, args.n_envs, args.n_steps_per_update, args.learning_rate, args.batch_size, args.ent_coef)
