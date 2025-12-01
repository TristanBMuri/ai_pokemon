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
    
    def __init__(self, simulator: BattleSimulator, gauntlet_name: str = "kanto_leaders"):
        super().__init__()
        self.simulator = simulator
        
        # Load gauntlet
        from nuzlocke_gauntlet_rl.data.parsers import load_kanto_leaders, load_indigo_league, load_team_rocket
        
        if gauntlet_name == "kanto_leaders":
            self.gauntlet_template = load_kanto_leaders()
        elif gauntlet_name == "indigo_league":
            self.gauntlet_template = load_indigo_league()
        elif gauntlet_name == "team_rocket":
            self.gauntlet_template = load_team_rocket()
        elif gauntlet_name == "extended":
            from nuzlocke_gauntlet_rl.data.parsers import load_extended_gauntlet
            self.gauntlet_template = load_extended_gauntlet()
        else:
            raise ValueError(f"Unknown gauntlet: {gauntlet_name}")
        
        # Initialize MovesetGenerator
        from nuzlocke_gauntlet_rl.utils.moveset_generator import MovesetGenerator
        self.moveset_generator = MovesetGenerator()
        
        # State
        self.roster: List[MonInstance] = []
        self.current_trainer_idx = 0
        
        # Action Space:
        # [0-5]: Roster Indices (Select 6 mons from max 30)
        # [6-11]: Build Indices (Select build 0-2 for each mon)
        # [12]: Risk Token (0=Safe, 1=Neutral, 2=Desperate)
        self.max_roster_size = 30
        self.n_builds = 3
        
        dims = [self.max_roster_size] * 6 + [self.n_builds] * 6 + [3]
        self.action_space = spaces.MultiDiscrete(dims)
        
        # Observation Space:
        # - Trainer Index (Discrete)
        # - Roster Count (Discrete)
        # - Top 6 Mons Levels (Box)
        # - Opponent Team Preview (Box): 6 mons * 2 types (encoded)
        #   We'll encode types as integers 0-18.
        self.observation_space = spaces.Dict({
            "trainer_idx": spaces.Discrete(100), # Max 100 trainers
            "roster_count": spaces.Discrete(100),
            "party_levels": spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32),
            "opponent_preview": spaces.Box(low=0, high=18, shape=(6, 2), dtype=np.int32)
        })
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_trainer_idx = 0
        
        # Initialize starter roster (Full team of 6)
        # Initialize large roster (Box System)
        # We want ~50 fully evolved mons.
        # For now, we'll use a curated list of strong mons to ensure viability.
        species_pool = [
            "Charizard", "Blastoise", "Venusaur", "Pikachu", "Snorlax", "Gengar",
            "Dragonite", "Tyranitar", "Metagross", "Salamence", "Garchomp", "Hydreigon",
            "Volcarona", "Aegislash", "Dragapult", "Gyarados", "Milotic", "Swampert",
            "Infernape", "Empoleon", "Torterra", "Lucario", "Gardevoir", "Gallade",
            "Scizor", "Heracross", "Weavile", "Gliscor", "Mamoswine", "Rotom-Wash",
            "Ferrothorn", "Excadrill", "Conkeldurr", "Chandelure", "Haxorus", "Darmanitan",
            "Alakazam", "Machamp", "Golem", "Slowbro", "Starmie", "Jolteon", "Vaporeon",
            "Flareon", "Espeon", "Umbreon", "Sylveon", "Leafeon", "Glaceon", "Togekiss"
        ]
        
        self.roster = []
        for species in species_pool:
            # Randomize ability/nature slightly? For now standard.
            ability = self.moveset_generator.get_ability(species)
            self.roster.append(
                MonInstance(
                    id=str(uuid.uuid4()), 
                    spec=PokemonSpec(species=species, level=50, moves=[], ability=ability), 
                    current_hp=100, 
                    in_party=False
                )
            )
            
        # Set first 6 as initial party
        for i in range(6):
            if i < len(self.roster):
                self.roster[i].in_party = True
        
        return self._get_obs(), {}
        
    def step(self, action):
        # Unpack action
        roster_indices = action[0:6]
        build_indices = action[6:12]
        risk_token = action[12]
        
        # Construct Party
        party = []
        party_specs = []
        
        # Filter alive roster
        alive_roster = [m for m in self.roster if m.alive]
        
        for i, roster_idx in enumerate(roster_indices):
            if roster_idx < len(alive_roster):
                mon = alive_roster[roster_idx]
                
                # Check if already selected (unique mons only)
                if mon in party:
                    continue
                    
                # Generate Moveset
                build_idx = build_indices[i]
                movesets = self.moveset_generator.generate_builds(mon.spec, n_builds=self.n_builds)
                
                # Select build (fallback to 0 if invalid index, though MultiDiscrete handles range)
                selected_moves = movesets[build_idx] if build_idx < len(movesets) else movesets[0]
                if not selected_moves and movesets: selected_moves = movesets[0] # Fallback
                
                # Update spec with selected moves
                # We create a copy to avoid mutating the original spec permanently?
                # Actually, spec is static, but moves change per battle?
                # For now, just update the spec.
                mon.spec.moves = selected_moves
                
                party.append(mon)
                party_specs.append(mon.spec)
                mon.in_party = True
            
        # Ensure at least 1 mon, and try to fill up to 6
        if not party and alive_roster:
            # If completely empty, force add first one
            mon = alive_roster[0]
            movesets = self.moveset_generator.generate_builds(mon.spec, n_builds=self.n_builds)
            mon.spec.moves = movesets[0] if movesets else []
            party.append(mon)
            party_specs.append(mon.spec)
            mon.in_party = True
            
        # Auto-fill if < 6 and we have more alive mons
        # This helps the agent learn by giving it a full team even if it selects few
        if len(party) < 6 and len(party) < len(alive_roster):
            for mon in alive_roster:
                if len(party) >= 6: break
                if mon not in party:
                    # Default to build 0
                    movesets = self.moveset_generator.generate_builds(mon.spec, n_builds=self.n_builds)
                    mon.spec.moves = movesets[0] if movesets else []
                    party.append(mon)
                    party_specs.append(mon.spec)
                    mon.in_party = True
            
        # Update in_party status for others
        for m in self.roster:
            if m not in party:
                m.in_party = False
            
        # Proceed to battle
        current_trainer = self.gauntlet_template.trainers[self.current_trainer_idx]
        
        # Simulate Battle
        win, survivors = self.simulator.simulate_battle(party_specs, current_trainer.team, risk_token=risk_token)
        
        # Apply deaths
        deaths = 0
        for i, survived in enumerate(survivors):
            if i < len(party): # Safety check
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
            
        # Opponent Preview
        opponent_preview = np.zeros((6, 2), dtype=np.int32)
        
        # Type Mapping
        type_map = {
            "Normal": 1, "Fire": 2, "Water": 3, "Electric": 4, "Grass": 5, "Ice": 6,
            "Fighting": 7, "Poison": 8, "Ground": 9, "Flying": 10, "Psychic": 11,
            "Bug": 12, "Rock": 13, "Ghost": 14, "Dragon": 15, "Steel": 16, "Dark": 17, "Fairy": 18
        }
        
        if self.current_trainer_idx < len(self.gauntlet_template.trainers):
            trainer = self.gauntlet_template.trainers[self.current_trainer_idx]
            for i, mon in enumerate(trainer.team[:6]):
                types = self.moveset_generator.get_types(mon.species)
                for j, t in enumerate(types[:2]):
                    opponent_preview[i, j] = type_map.get(t, 0)
                
        return {
            "trainer_idx": self.current_trainer_idx,
            "roster_count": len([m for m in self.roster if m.alive]),
            "party_levels": levels,
            "opponent_preview": opponent_preview
        }
