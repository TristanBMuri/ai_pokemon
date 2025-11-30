
import numpy as np
from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
from nuzlocke_gauntlet_rl.envs.real_battle_simulator import RealBattleSimulator

def test_integration():
    # Path to the trained model
    model_path = "models/ppo_risk_agent_v1"
    
    print("Initializing RealBattleSimulator...", flush=True)
    simulator = RealBattleSimulator(model_path=model_path)
    
    print("Initializing NuzlockeGauntletEnv...", flush=True)
    env = NuzlockeGauntletEnv(simulator=simulator)
    
    print("Resetting environment...", flush=True)
    obs, info = env.reset()
    
    done = False
    truncated = False
    total_reward = 0
    step = 0
    
    print("Starting episode...")
    while not (done or truncated):
        # Action: 
        # [0] Roster: 0 (Battle) or 1 (Swap)
        # [1] Risk: 0 (Safe), 1 (Neutral), 2 (Desperate)
        
        # Simple policy: Always battle (0), random risk
        roster_action = 0
        risk_action = np.random.randint(0, 3)
        
        action = np.array([roster_action, risk_action])
        
        print(f"Step {step}: Trainer {env.current_trainer_idx}, Action {action}")
        
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        print(f"  Result: Reward={reward:.2f}, Done={done}")
        
        # Print roster status
        alive_mons = [m.spec.species for m in env.roster if m.alive]
        dead_mons = [m.spec.species for m in env.roster if not m.alive]
        print(f"  Alive: {alive_mons}")
        print(f"  Dead: {dead_mons}")
        
    print(f"Episode finished in {step} steps. Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    test_integration()
