import asyncio
import numpy as np
import threading
import time
from typing import List, Tuple
from stable_baselines3 import PPO
from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import ServerConfiguration, AccountConfiguration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.teambuilder import Teambuilder, ConstantTeambuilder
from nuzlocke_gauntlet_rl.envs.battle_env import BattleEnv
from nuzlocke_gauntlet_rl.envs.battle_simulator import BattleSimulator
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec
import uuid
import os

# Dummy Teambuilder for parsing
class ParsingTeambuilder(Teambuilder):
    def yield_team(self):
        return None

class RealBattleSimulator(BattleSimulator):
    """
    Runs battles using a trained RL agent and a local Showdown server.
    """
    def __init__(self, model_path: str, server_url: str = "ws://192.168.1.122:8000/showdown/websocket", device: str = "auto"):
        self.server_config = ServerConfiguration(server_url, None)
        
        # Create unique IDs
        self.agent_name = f"SimAgent_{uuid.uuid4().hex[:8]}"
        self.opponent_name = f"SimOpp_{uuid.uuid4().hex[:8]}"
        
        # Initialize Opponent (Heuristic is better for realistic testing)
        # Initialize Opponent (Heuristic is better for realistic testing)
        self.opponent = SimpleHeuristicsPlayer(
            battle_format="gen9customgame", # Custom game allows any team
            server_configuration=self.server_config,
            account_configuration=AccountConfiguration(self.opponent_name, None),
        )
        
        # Initialize BattleEnv (The Agent)
        self.pz_env = BattleEnv(
            battle_format="gen9customgame",
            server_configuration=self.server_config,
            account_configuration1=AccountConfiguration(self.agent_name, None),
        )
        
        # Wrap for Gym
        from nuzlocke_gauntlet_rl.wrappers.single_agent_battle_wrapper import MySingleAgentWrapper
        self.env = MySingleAgentWrapper(self.pz_env, opponent=self.opponent)
        
        # Load Model
        print(f"Loading model from {model_path} (device={device})...")
        self.model = PPO.load(model_path, device=device)
        print("Model loaded.")
        
        self.teambuilder = ParsingTeambuilder()
        
    def _pack_team(self, team_str: str) -> str:
        return self.teambuilder.join_team(self.teambuilder.parse_showdown_team(team_str))

    def simulate_battle(self, my_team: List[PokemonSpec], enemy_team: List[PokemonSpec], risk_token: int = 0) -> Tuple[bool, List[bool]]:
        """
        Runs a single battle simulation.
        """
        # Convert teams to string format
        my_team_str = self._specs_to_team_str(my_team)
        enemy_team_str = self._specs_to_team_str(enemy_team)
        
        # Pack teams
        my_packed = self._pack_team(my_team_str)
        enemy_packed = self._pack_team(enemy_team_str)
        
        print(f"DEBUG: My Team (Packed): {my_packed[:50]}...")
        
        # Update teams using ConstantTeambuilder
        self.opponent._team = ConstantTeambuilder(enemy_packed)
        
        # Update agent team
        if hasattr(self.pz_env, "agent1"):
             self.pz_env.agent1._team = ConstantTeambuilder(my_packed)
        else:
             print("WARNING: Could not find agent1 on pz_env to set team.")
             self.pz_env._team = ConstantTeambuilder(my_packed)

        # Start background thread to trigger challenge
        def challenge_trigger():
            # print("Challenge trigger thread started. Waiting 1s...", flush=True)
            time.sleep(1) # Wait for reset to start
            # print("Triggering challenge now...", flush=True)
            # Find target
            target = self.pz_env
            if hasattr(self.pz_env, "agent1"):
                target = self.pz_env.agent1
            
            # Run challenge in new loop
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def run_challenge():
                     await self.opponent.ps_client.logged_in.wait()
                     await self.opponent.battle_against(target, n_battles=1)
                     
                loop.run_until_complete(run_challenge())
                loop.close()
            except Exception as e:
                print(f"Challenge trigger failed: {e}")

        t = threading.Thread(target=challenge_trigger, daemon=True)
        t.start()
             
        # Monitor removed

        # Reset and run
        print("Calling self.env.reset()...", flush=True)
        obs, info = self.env.reset(options={"risk_token": risk_token})
        print("self.env.reset() returned.", flush=True)
        done = False
        truncated = False
        
        while not (done or truncated):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = self.env.step(action)
            
        # Battle over.
        # Get result.
        battle = getattr(self.pz_env, "battle1", None)
        if not battle:
             battle = getattr(self.pz_env, "battle", None)
             
        if not battle:
            print("ERROR: Could not find battle object on pz_env")
            return False, [False]*len(my_team)
        
        win = battle.won
        
        # Survivors
        survivors = []
        for mon_spec in my_team:
            name = mon_spec.species
            found = False
            for mon in battle.team.values():
                if mon.species == name: 
                    survivors.append(not mon.fainted)
                    found = True
                    break
            if not found:
                survivors.append(False) # Fallback
                
        return win, survivors

    def _specs_to_team_str(self, specs: List[PokemonSpec]) -> str:
        return "\n\n".join([s.to_showdown_format() for s in specs])
