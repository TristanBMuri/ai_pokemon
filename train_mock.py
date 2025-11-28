from stable_baselines3 import PPO
from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
from nuzlocke_gauntlet_rl.envs.mock_simulator import MockBattleSimulator
import os

def train():
    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # Create env
    sim = MockBattleSimulator()
    env = NuzlockeGauntletEnv(sim)
    
    # Initialize agent
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    print("Starting training...")
    # Train for a short duration just to verify it runs
    model.learn(total_timesteps=10000)
    print("Training finished!")
    
    # Save model
    model.save("nuzlocke_mock_ppo")
    
    # Evaluate
    obs, _ = env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: Reward {reward}, Done {done}")
        if done:
            obs, _ = env.reset()

if __name__ == "__main__":
    train()
