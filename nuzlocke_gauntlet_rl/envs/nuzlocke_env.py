import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, List
from nuzlocke_gauntlet_rl.utils.specs import GauntletSpec, MonInstance, PokemonSpec
from nuzlocke_gauntlet_rl.envs.battle_simulator import BattleSimulator
from nuzlocke_gauntlet_rl.data.parsers import load_kanto_leaders
import uuid

class NuzlockeGauntletEnv(gym.Env):
    """
    Gymnasium environment for the Nuzlocke Gauntlet.
    
    Phases:
    0: PRE_BATTLE - Select party from box.
    1: DONE - Game over.
    """
    
    def __init__(self, simulator: BattleSimulator):
        super().__init__()
        self.simulator = simulator
        
        # Load gauntlet (hardcoded for now)
        self.gauntlet_template = load_kanto_leaders()
        
        # State
        self.roster: List[MonInstance] = []
        self.current_trainer_idx = 0
        
        # Action Space:
        # For now, simplified: Select top N mons from roster to form a party.
        # We'll just use Discrete(1) for "Go with current top 6" to start, 
        # but really we want to select a subset.
        # Let's make it Discrete(2): 
        # 0: Fight with current party (first 6 alive)
        # 1: Swap first box mon with last party mon (naive box management)
        self.action_space = spaces.Discrete(2)
        
        # Observation Space:
        # Simplified for prototype:
        # - Trainer Index (Discrete)
        # - Roster Count (Discrete)
        # - Top 6 Mons Levels (Box)
        self.observation_space = spaces.Dict({
            "trainer_idx": spaces.Discrete(20), # Max 20 trainers
            "roster_count": spaces.Discrete(100),
            "party_levels": spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32)
        })
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_trainer_idx = 0
        
        # Initialize starter roster (e.g. 3 random mons level 50)
        self.roster = [
            MonInstance(id=str(uuid.uuid4()), spec=PokemonSpec(species="Starter1", level=50, moves=[]), current_hp=100, in_party=True),
            MonInstance(id=str(uuid.uuid4()), spec=PokemonSpec(species="Starter2", level=50, moves=[]), current_hp=100, in_party=True),
            MonInstance(id=str(uuid.uuid4()), spec=PokemonSpec(species="Starter3", level=50, moves=[]), current_hp=100, in_party=True),
            MonInstance(id=str(uuid.uuid4()), spec=PokemonSpec(species="Box1", level=45, moves=[]), current_hp=100, in_party=False)
        ]
        
        return self._get_obs(), {}
        
    def step(self, action):
        # Simple logic
        if action == 1:
            # Swap logic (placeholder)
            pass
            
        # Proceed to battle (always battle for now if action 0)
        current_trainer = self.gauntlet_template.trainers[self.current_trainer_idx]
        
        # Get party (alive and in_party)
        party = [m for m in self.roster if m.alive and m.in_party][:6]
        party_specs = [m.spec for m in party]
        
        win, survivors = self.simulator.simulate_battle(party_specs, current_trainer.team)
        
        # Apply deaths
        deaths = 0
        for i, survived in enumerate(survivors):
            if not survived:
                party[i].alive = False
                deaths += 1
                
        reward = 0
        terminated = False
        truncated = False
        
        if win:
            reward += 1.0
            self.current_trainer_idx += 1
            if self.current_trainer_idx >= len(self.gauntlet_template.trainers):
                reward += 10.0 # Gauntlet clear bonus
                terminated = True
        else:
            reward -= 1.0 # Wipe penalty
            terminated = True # Game over
            
        # Penalty for deaths
        reward -= (deaths * 0.1)
        
        # Check if we have any alive mons left
        alive_count = sum(1 for m in self.roster if m.alive)
        if alive_count == 0:
            terminated = True
            reward -= 5.0
            
        return self._get_obs(), reward, terminated, truncated, {}
        
    def _get_obs(self):
        party = [m for m in self.roster if m.alive and m.in_party][:6]
        levels = np.zeros(6, dtype=np.float32)
        for i, m in enumerate(party):
            levels[i] = m.spec.level
            
        return {
            "trainer_idx": self.current_trainer_idx,
            "roster_count": len([m for m in self.roster if m.alive]),
            "party_levels": levels
        }
