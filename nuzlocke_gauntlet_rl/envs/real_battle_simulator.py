
import asyncio
import numpy as np
from typing import List, Tuple
from stable_baselines3 import PPO
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import ServerConfiguration, AccountConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from nuzlocke_gauntlet_rl.envs.battle_env import BattleEnv
from nuzlocke_gauntlet_rl.envs.battle_simulator import BattleSimulator
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec
import uuid
import os

class RealBattleSimulator(BattleSimulator):
    """
    Runs battles using a trained RL agent and a local Showdown server.
    """
    def __init__(self, model_path: str, server_url: str = "ws://192.168.1.122:8000/showdown/websocket"):
        self.server_config = ServerConfiguration(server_url, None)
        
        # Create unique IDs
        self.agent_name = f"SimAgent_{uuid.uuid4().hex[:8]}"
        self.opponent_name = f"SimOpp_{uuid.uuid4().hex[:8]}"
        
        # Initialize Opponent (Heuristic is better for realistic testing)
        self.opponent = RandomPlayer(
            battle_format="gen8customgame", # Custom game allows any team
            server_configuration=self.server_config,
            account_configuration=AccountConfiguration(self.opponent_name, None),
        )
        
        # Initialize BattleEnv (The Agent)
        self.pz_env = BattleEnv(
            battle_format="gen8customgame",
            server_configuration=self.server_config,
            account_configuration1=AccountConfiguration(self.agent_name, None),
        )
        
        # Wrap for Gym
        self.env = SingleAgentWrapper(self.pz_env, opponent=self.opponent)
        
        # Load Model
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)
        print("Model loaded.")
        
    def simulate_battle(self, my_team: List[PokemonSpec], enemy_team: List[PokemonSpec], risk_token: int = 0) -> Tuple[bool, List[bool]]:
        """
        Runs a single battle simulation.
        """
        # Convert teams to string format
        my_team_str = self._specs_to_team_str(my_team)
        enemy_team_str = self._specs_to_team_str(enemy_team)
        
        print(f"DEBUG: My Team:\n{my_team_str}")
        print(f"DEBUG: Enemy Team:\n{enemy_team_str}")
        
        # Update teams
        # Try to use update_team method if available
        if hasattr(self.opponent, "update_team"):
            self.opponent.update_team(enemy_team_str)
        else:
            self.opponent._team = enemy_team_str
            
        # For agent (BattleEnv), it seems to be a PettingZoo env with agent1/agent2
        # If agent1 is the player object:
        if hasattr(self.pz_env, "agent1") and hasattr(self.pz_env.agent1, "_team"):
             pass
             
        # Let's try setting on pz_env._team anyway, but also check if we can set it on the internal player
        self.pz_env._team = my_team_str
        
        try:
            self.pz_env.agent1._team = my_team_str
        except:
            pass
            
        # Reset and run
        obs, info = self.env.reset(options={"risk_token": risk_token})
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)
            
        # Battle over.
        # Get result.
        # Access the battle object from the agent
        # pz_env has battle1
        battle = getattr(self.pz_env, "battle1", None)
        if not battle:
             battle = getattr(self.pz_env, "battle", None)
             
        if not battle:
            print("ERROR: Could not find battle object on pz_env")
            return False, [False]*len(my_team)
        
        win = battle.won
        
        # Survivors
        # We need to map back to the original my_team order?
        # Showdown might reorder? No, usually preserves order.
        # We need to match by species/nickname.
        
        survivors = []
        for mon_spec in my_team:
            # Construct expected name (species or nickname)
            # Showdown uses Species as name if no nickname.
            # We don't use nicknames yet.
            name = mon_spec.species
            
            # Find in battle.team
            # Keys in battle.team are usually "p1: Species" or just "Species"?
            # poke-env parses it.
            
            # Let's iterate values and match species.
            found = False
            for mon in battle.team.values():
                if mon.species == name: # or mon.species in name
                    survivors.append(not mon.fainted)
                    found = True
                    break
            if not found:
                # If not found, maybe it was never sent out? 
                # Or maybe name mismatch.
                # Assume dead if not found? Or alive?
                # In a full battle, all mons should be in the team dict.
                survivors.append(False) # Fallback
                
        return win, survivors

    def _specs_to_team_str(self, specs: List[PokemonSpec]) -> str:
        return "\n\n".join([s.to_showdown_format() for s in specs])
