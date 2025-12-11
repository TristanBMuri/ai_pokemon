from nuzlocke_gauntlet_rl.envs.nuzlocke_env import NuzlockeGauntletEnv
import numpy as np

def test_flow():
    print("Initializing Env...")
    env = NuzlockeGauntletEnv(model_path="", watch_mode=False) # Mock simulator
    
    print("Resetting...")
    obs, info = env.reset()
    print(f"Initial Phase: {obs['phase']} (Expected 3=STARTER)")
    
    # Step 1: Select Starter
    print("Selecting Starter (Action 1)...")
    obs, reward, term, trunc, info = env.step(1)
    print(f"Phase after Starter: {obs['phase']} (Expected 4=STRATEGIST)")
    
    if obs['phase'] == 4:
        print("Success: Entered Strategist Phase.")
        # Step 2: Strategist Action (0 = Engage)
        print("Executing Strategist Action (0)...")
        obs, reward, term, trunc, info = env.step(0)
        print(f"Phase after Strategist: {obs['phase']} (Expected 0=DECISION)")
        
        if obs['phase'] == 0:
             print("Success: Entered Decision Phase.")
             # Check Trainer Info
             print(f"Opponent Preview Shape: {obs['opponent_preview'].shape}")
             # Check if non-zero (Trainer loaded)
             if np.sum(obs['opponent_preview']) > 0:
                 print("Success: Opponent Preview is populated.")
             else:
                 print("WARNING: Opponent Preview is empty/zeros.")
                 
    else:
        print("FAILED: Did not enter Strategist Phase.")

if __name__ == "__main__":
    test_flow()
