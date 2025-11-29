
import asyncio
import numpy as np
import os
from stable_baselines3 import PPO
from poke_env.player import RandomPlayer
from poke_env import ServerConfiguration, AccountConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from nuzlocke_gauntlet_rl.envs.battle_env import BattleEnv
import uuid

def evaluate(model_path, n_episodes=20):
    # Configuration
    local_server_config = ServerConfiguration(
        "ws://192.168.1.122:8000/showdown/websocket",
        None
    )
    
    opponent = RandomPlayer(
        battle_format="gen8randombattle",
        server_configuration=local_server_config,
        account_configuration=AccountConfiguration(f"EvalOpp_{uuid.uuid4().hex[:8]}", None),
    )
    
    pz_env = BattleEnv(
        battle_format="gen8randombattle",
        server_configuration=local_server_config,
        account_configuration1=AccountConfiguration(f"EvalAgent_{uuid.uuid4().hex[:8]}", None),
    )
    
    env = SingleAgentWrapper(pz_env, opponent=opponent)
    
    model = PPO.load(model_path)
    
    results = {}
    
    for risk_level, risk_name in [(0, "Safe"), (2, "Desperate")]:
        print(f"\nEvaluating with Risk Level: {risk_name} ({risk_level})")
        wins = 0
        total_fainted = 0
        
        for i in range(n_episodes):
            obs, info = env.reset(options={"risk_token": risk_level})
            done = False
            truncated = False
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                
            # Check result
            # We need to access the inner battle to check win/loss and fainted
            # SingleAgentWrapper hides it, but we can access via pz_env
            # Wait, pz_env.agent1.battle is the last battle object
            
            battle = pz_env.agent1.battle
            if battle and battle.won:
                wins += 1
            
            fainted = len([mon for mon in battle.team.values() if mon.fainted]) if battle else 0
            total_fainted += fainted
            
            print(f"Episode {i+1}/{n_episodes}: {'Win' if battle.won else 'Loss'}, Fainted: {fainted}")
            
        results[risk_name] = {
            "win_rate": wins / n_episodes,
            "avg_fainted": total_fainted / n_episodes
        }
        
    print("\n--- Evaluation Results ---")
    for name, metrics in results.items():
        print(f"{name}: Win Rate={metrics['win_rate']:.2f}, Avg Fainted={metrics['avg_fainted']:.2f}")
        
    env.close()

if __name__ == "__main__":
    evaluate("models/ppo_risk_agent")
