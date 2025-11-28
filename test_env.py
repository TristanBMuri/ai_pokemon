from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
from nuzlocke_gauntlet_rl.envs.mock_simulator import MockBattleSimulator
from stable_baselines3.common.env_checker import check_env

def test_env():
    sim = MockBattleSimulator()
    env = NuzlockeGauntletEnv(sim)
    
    print("Checking environment compliance...")
    check_env(env)
    print("Environment is compliant!")
    
    obs, _ = env.reset()
    print("Initial Obs:", obs)
    
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        print(f"Action: {action}, Reward: {reward}, Done: {done}")

if __name__ == "__main__":
    test_env()
