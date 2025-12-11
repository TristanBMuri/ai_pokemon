import gymnasium as gym
from gymnasium import spaces
import numpy as np
import uuid
import uuid
from typing import List, Dict, Optional, Tuple, Any

from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec, TrainerSpec, GauntletSpec, MonInstance
from nuzlocke_gauntlet_rl.data.parsers import (
    load_kanto_leaders, 
    load_indigo_league, 
    load_kanto_leaders, 
    load_indigo_league, 
    load_team_rocket, 
    load_complete_gauntlet,  # Ensure this is available
    build_gauntlet_graph
)
from nuzlocke_gauntlet_rl.mechanics.map_core import GauntletMap, GauntletNode
from nuzlocke_gauntlet_rl.envs.real_battle_simulator import RealBattleSimulator
# For testing/mocking, you might conditionally import MockBattleSimulator

class NuzlockeGauntletEnv(gym.Env):
    """
    Sequential Nuzlocke Gauntlet environment.
    
    Phases:
    1. DECISION: Choose to [0] Fight (Keep Team) or [1] Edit Team (-0.1 Reward).
    2. BUILD_MEMBER: Choose Species from Roster (Index 0..MaxRoster). Repeated 6 times.
    3. BUILD_MOVES: Choose Move from Learnset (Index 0..MaxMoveID) + [0] Stop. Repeated 4 times per member.
    4. BATTLE: Executed automatically after Decision=Fight or Team Build Complete.
    """
    
    metadata = {"render_modes": ["human"]}
    
    PHASE_DECISION = 0
    PHASE_SELECT_MEMBER = 1
    PHASE_SELECT_MOVE = 2
    PHASE_SELECT_STARTER = 3
    PHASE_STRATEGIST = 4 
    PHASE_BUILD_SPECIES = 5 
    PHASE_BUILD_MOVE = 6
    PHASE_BUILD_ITEM = 7
    
    def __init__(
        self,
        gauntlet_name: str = "complete",
        model_path: str = "models/battle_agent_v1", # Low-level policy
        max_roster_size: int = 400,
        simulator_url: str = "ws://localhost:8000/showdown/websocket",
        watch_mode: bool = False
    ):
        super().__init__()
        
        # Load Gauntlet
        if gauntlet_name == "complete":
            self.gauntlet_template = load_complete_gauntlet()
        elif gauntlet_name == "kanto_leaders":
            self.gauntlet_template = load_kanto_leaders()
        elif gauntlet_name == "indigo_league":
            self.gauntlet_template = load_indigo_league()
        elif gauntlet_name == "team_rocket":
            self.gauntlet_template = load_team_rocket()
        else:
            try:
                # Try relative import for extended if needed
                from nuzlocke_gauntlet_rl.data.parsers import load_extended_gauntlet
                if gauntlet_name == "extended":
                     self.gauntlet_template = load_extended_gauntlet()
                else:
                     raise ValueError
            except:
                raise ValueError(f"Unknown gauntlet: {gauntlet_name}")
                
        # Initialize MovesetGenerator
        from nuzlocke_gauntlet_rl.utils.moveset_generator import MovesetGenerator
        self.moveset_generator = MovesetGenerator()
        
        # [NEW] Build Map
        self.gauntlet_map = build_gauntlet_graph(self.gauntlet_template)
        self.current_node_id = self.gauntlet_map.start_node_id
        
        # Determine Limits
        self.max_roster_size = max_roster_size
        self.max_move_id = self.moveset_generator.max_move_id # e.g. ~950
        
        # Action Space: Single Discrete Value
        # Max of (Decisions=2, RosterSize=400, MoveIDs=1000)
        # We assume 0 is NO-OP or Stop in some contexts.
        self.action_space_size = max(self.max_roster_size, self.max_move_id + 1)
        self.action_space = spaces.Discrete(self.action_space_size)
        
        # Observation Space (Dict)
        # We need to expose enough state for the agent to know WHAT to pick.
        # This is tricky for PPO without variable sizing.
        # Minimal viable obs:
        # - Current Phase (Enum)
        # - Current Slot Index (0-5 for Member, 0-3 for Move)
        # - Current Trainer Index (0-60)
        # - Current Party Summary (Level, Type1/2 encoded?) - Or rely on LSTM memory?
        # - Opponent Preview (Same as before)
        # - Mask (Available actions) - Handled via `valid_action_mask`.
        
        # For simplicity, we reuse the old structure but add phase info.
        self.observation_space = spaces.Dict({
            "phase": spaces.Discrete(8), # Updated to include new phases
            "slot_idx": spaces.Discrete(7), 
            "node_idx": spaces.Discrete(200), # Approximate node count
            "roster_count": spaces.Discrete(self.max_roster_size + 1),
            "party_levels": spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32), 
            "opponent_preview": spaces.Box(low=0, high=1000, shape=(6, 14), dtype=np.int32),
            "map_state": spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32),
            "risk_vector": spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32) # Struct for risk per party slot?
        })
        
        # Simulator
        if not model_path:
             from nuzlocke_gauntlet_rl.envs.mock_battle_simulator import MockBattleSimulator
             self.simulator = MockBattleSimulator()
        else:
             self.simulator = RealBattleSimulator(model_path=model_path, server_url=simulator_url)
        
        self.watch_mode = watch_mode
        
        # Initialize Mechanics
        from nuzlocke_gauntlet_rl.data.parsers import load_encounters
        from nuzlocke_gauntlet_rl.mechanics.nuzlocke_mechanics import NuzlockeMechanics, trainer_unlocks
        
        self.encounters_map = load_encounters() # Load CSV
        self.mechanics = NuzlockeMechanics(self.encounters_map, self.moveset_generator)
        self.trainer_unlocks = trainer_unlocks
        
        # Internal State
        self.roster: List[MonInstance] = []
        # self.current_trainer_idx = 0 # DEPRECATED
        self.current_node_id = self.gauntlet_map.start_node_id
        self.visited_routes: Set[str] = set()
        self.party: List[MonInstance] = [] # Current ACTIVE party
        self.rebuild_count = 0 # Track rebuilds for current trainer
        
        # Building State
        self.build_party_specs: List[PokemonSpec] = []
        self.build_current_mon: Optional[MonInstance] = None
        self.build_current_moves: List[str] = []
        self.current_party_slot = 0 # 0-5
        self.current_move_slot = 0 # 0-3
        
        self.current_phase = self.PHASE_DECISION
        self.current_slot = 0 # Exposed in Obs, derived from above

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # self.current_trainer_idx = 0 # DEPRECATED
        self.current_node_id = self.gauntlet_map.start_node_id
        self.rebuild_count = 0
        
        # 1. Initialize Roster with Starter
        self.roster = [] # Empty Roster
        
        # 2. Process Initial Unlocks (None)
        # User requested 2nd encounter ONLY after first fight.
        # We also do NOT give starter here. We let the Agent pick.
        self.visited_routes = set()
        initial_routes = ["PALLET TOWN"] 
        for r in initial_routes:
             self.visited_routes.add(r)
            
        # Clear Party
        self.party = []
        
        # Start at STARTER selection, then transition to STRATEGIST
        self.current_phase = self.PHASE_SELECT_STARTER
        
        return self._get_obs(), {}

    def step(self, action: int):
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if self.current_phase == self.PHASE_SELECT_STARTER:
            choice_idx = int(action)
            # Support 0-26 (Gen 1-9 starters)
            starter = self.mechanics.get_starter_choice(choice_idx)
            if starter:
                self.roster.append(starter)
            else:
                self.roster.append(self.mechanics.get_starter_choice(1)) # Check your bounds!
                
            self.current_phase = self.PHASE_STRATEGIST
            return self._get_obs(), 0, False, False, {"metrics": {}}

        # PHASE: STRATEGIST [NEW]
        elif self.current_phase == self.PHASE_STRATEGIST:
            # Action: Choose Next Node (0 to N-1)
            # Currently we are AT a node. The Strategist decides to ENGAGE this node or (if optional) SKIP.
            # But our map graph structure implies we are at a node.
            # Let's say we are "Looking at" self.current_node_id.
            # Action 0: Enter/Engage.
            # Action 1: Skip (if optional).
            
            # Since our current graph is linear Gyms (Mandatory), we just auto-transition or accept 0.
            
            # Future: If we have branches, self.gauntlet_map.get_successors(current)
            
            # For now, just move to DECISION (Builder/Fight choice)
            self.current_phase = self.PHASE_DECISION
            return self._get_obs(), 0, False, False, {}

        # PHASE: DECISION
        elif self.current_phase == self.PHASE_DECISION:
            if action == 1: # REBUILD
                 # Economy: Cost 5 if > 0
                 cost = 0.0
                 if self.rebuild_count > 0:
                     cost = 5.0 # Reflecting specs
                     # self.current_budget -= 5 # If we tracked budget? prompt says "Progress + Items - Cost of deaths".
                     # Just applying negative reward?
                     reward = -0.1 * cost # Scale down to RL reward? Or just -5?
                     # Let's use -0.1 as base penalty to avoid huge swings, or strict -5 if budget
                     reward = -0.5 # Penalty
                 
                 self.rebuild_count += 1
                 
                 # Initialize Build
                 self.current_phase = self.PHASE_BUILD_SPECIES
                 self.current_party_slot = 0
                 self.party = [] # Reset Party
                 
            else: # FIGHT (0)
                 # Executed when agent is confident
                 outcome = self._run_battle()
                 return outcome # _run_battle returns full step tuple

        # PHASE: BUILD SPECIES (Slot 1-6)
        elif self.current_phase == self.PHASE_BUILD_SPECIES:
             # Action is Roster Index
             if 0 <= action < len(self.roster):
                 mon = self.roster[action]
                 # Valid pick? Alive and not already in party?
                 # Note: self.party is being built.
                 if mon.alive and mon not in self.party:
                     self.build_current_mon = mon
                     self.build_current_moves = []
                     self.current_move_slot = 0
                     self.current_phase = self.PHASE_BUILD_MOVE
                 else:
                     # Invalid pick (Should be masked). Penalty?
                     reward = -0.01
             
             # If Roster empty or no choices, handled by Mask?
             # Auto-finish if no candidates?

        # PHASE: BUILD MOVES (Move 1-4)
        elif self.current_phase == self.PHASE_BUILD_MOVE:
             # Action is Move ID
             move_name = self.moveset_generator.get_move_name(action)
             
             if self.build_current_mon and move_name:
                 # Add Move
                 self.build_current_moves.append(move_name)
                 self.current_move_slot += 1
                 
                 # Transition logic
                 if self.current_move_slot >= 4:
                     self.current_phase = self.PHASE_BUILD_ITEM
             else:
                 # Invalid move?
                 pass
                 
        # PHASE: BUILD ITEM
        elif self.current_phase == self.PHASE_BUILD_ITEM:
             # Action is Item ID (Placeholder: 0=None, 1=Leftovers...)
             # For now, ignore item logic or simple mock
             item = None # TODO: self.item_map.get(action)
             
             # Finalize Member
             if self.build_current_mon:
                 self.build_current_mon.spec.moves = self.build_current_moves
                 self.build_current_mon.spec.item = item
                 self.party.append(self.build_current_mon)
                 
             self.current_party_slot += 1
             
             if self.current_party_slot >= 6:
                 # Done Building
                 self.current_phase = self.PHASE_DECISION
             else:
                 # Next Member
                 self.current_phase = self.PHASE_BUILD_SPECIES
                 
        return self._get_obs(), reward, terminated, truncated, info

    def _run_battle(self):
         current_node = self.gauntlet_map.get_node(self.current_node_id)
         if not current_node or "trainer" not in current_node.data:
             # Should not happen in GYM node
             return self._get_obs(), 0, True, False, {"error": "No trainer in node"}
             
         current_trainer = current_node.data["trainer"]
         
         # Level Scaling
         target_level = 5
         if current_trainer.level_cap:
             target_level = current_trainer.level_cap
         elif current_trainer.team:
             target_level = max(p.level for p in current_trainer.team)
             
         target_level = max(5, target_level)
         
         for m in self.party:
             # RATCHET LEVELING: Only scale UP. Never down.
             m.spec.level = max(m.spec.level, target_level)
         
         party_specs = [m.spec for m in self.party]
         
         win, survivors, metrics = self.simulator.simulate_battle(party_specs, current_trainer.team, print_url=self.watch_mode)
         
         # Apply Deaths
         deaths = 0
         for i, survived in enumerate(survivors):
             if i < len(self.party):
                 if not survived:
                     self.party[i].alive = False
                     deaths += 1
                     
         # Rewards
         reward = 0
         terminated = False
         
         if win:
             reward += 1.0
             # Reset Rebuild Count for next trainer since we won
             self.rebuild_count = 0 
             
             # UNLOCK ROUTES
             unlocked = self.trainer_unlocks.get(self.current_trainer_idx, [])
             for r in unlocked:
                 if r not in self.visited_routes:
                     self.visited_routes.add(r)
                     mon = self.mechanics.roll_encounter(r, self.roster)
                     if mon: 
                         # print(f"Captured {mon.spec.species} on {r}!")
                         self.roster.append(mon)
             
             self.current_trainer_idx += 1
             if self.current_trainer_idx >= len(self.gauntlet_template.trainers):
                 reward += 10.0
                 terminated = True
         else:
             reward -= 1.0 # Wipe
             terminated = True # Loss ends run
             
         reward -= (deaths * 0.1)
         
         # Check Roster Wipe (Redundant if Loss=Terminated, but safe)
         if not any(m.alive for m in self.roster):
             terminated = True
             reward -= 5.0
             
         # Reset Phase if not terminated
         if not terminated:
             # Return to DECISION for next fight
             self.current_phase = self.PHASE_DECISION
             
         # Update Metrics with Env Context
         # metrics["trainer_idx"] = self.current_trainer_idx # Removed index tracking for now
         
         # Move to Next Node
         if win:
             successors = self.gauntlet_map.get_successors(self.current_node_id)
             if successors:
                 self.current_node_id = successors[0].node_id
                 self.current_phase = self.PHASE_STRATEGIST
             else:
                 # Victory!
                 reward += 10.0
                 terminated = True
         else:
             terminated = True # Wipe = Game Over
             
         metrics["pokemon_fainted"] = deaths
         
         info = {
             "metrics": metrics,
             "episode": {"r": reward, "l": 1} # Dummy
         }
         return self._get_obs(), reward, terminated, False, info

    def valid_action_mask(self):
        mask = np.zeros(self.action_space_size, dtype=bool)
        
        if self.current_phase == self.PHASE_SELECT_STARTER:
            mask[:27] = True # Gen 1-9 Starters (0-26)
            return mask
            
        elif self.current_phase == self.PHASE_DECISION:
             # 0=Fight, 1=Rebuild
             if self.party: mask[0] = True # Can fight if party exists (Wait, Party is cleared on Rebuild. So only if Prev Party?)
             # Issue: self.party is current ACTIVE party. On Reset it is empty.
             # So on First Turn, self.party is empty -> Must Rebuild.
             
             # But if we just finished a battle, self.party has survivors.
             
             if not self.party:
                 mask[0] = False # Must rebuild
             else:
                 mask[0] = True
                 
             mask[1] = True # Always can rebuild
             
        elif self.current_phase == self.PHASE_BUILD_SPECIES:
             # Mask valid roster indices
             any_valid = False
             for i, m in enumerate(self.roster):
                 if m.alive and m not in self.party:
                     mask[i] = True
                     any_valid = True
             
             # If no valid choices (all dead or party full?), we should have auto-exited.
             # But if not, allow 0 (NO-OP/Stop)?
             if not any_valid:
                 mask[0] = True
                 
        elif self.current_phase == self.PHASE_BUILD_MOVE:
             if self.build_current_mon:
                 lvl = self.build_current_mon.spec.level
                 learnable_ids = self.moveset_generator.get_learnable_moves_ids_at_level(self.build_current_mon.spec.species, lvl)
                 picked_names = set(self.build_current_moves)
                 
                 for mid in learnable_ids:
                     mname = self.moveset_generator.get_move_name(mid)
                     if mname not in picked_names:
                         mask[mid] = True
             
             # Always allow some fallback if empty?
             if not np.any(mask):
                  mask[0] = True # Prevent crash
                  
        elif self.current_phase == self.PHASE_BUILD_ITEM:
             mask[:] = True # All items allowed (or masked by inventory)
                        
        return mask

    def _get_obs(self):
        # 1. Opponent Preview
        opp_preview = np.zeros((6, 14), dtype=np.int32)
        if self.current_node_id:
             node = self.gauntlet_map.get_node(self.current_node_id)
             if node and "trainer" in node.data:
                 trainer = node.data["trainer"]
             else:
                 trainer = None
        else:
            trainer = None
            
        if trainer:
            for i, mon in enumerate(trainer.team[:6]):
                # Encode Types
                types = self.moveset_generator.get_types(mon.species)
                # Encode Types (Simple Hash)
                t1 = abs(hash(types[0].lower())) % 20 if types else 0
                t2 = abs(hash(types[1].lower())) % 20 if len(types) > 1 else 0
                
                # Base Stats
                stats = self.moveset_generator.get_base_stats(mon.species)
                
                # Features: [Type1, Type2, HP, Atk, Def, SpA, SpD, Spe, Level, AbilityID, Move1, Move2, Move3, Move4]
                opp_preview[i] = [
                    t1, t2,
                    stats[0], stats[1], stats[2], stats[3], stats[4], stats[5],
                    mon.level,
                    self.moveset_generator.encode_ability(mon.ability),
                    self.moveset_generator.encode_move(mon.moves[0] if len(mon.moves)>0 else ""),
                    self.moveset_generator.encode_move(mon.moves[1] if len(mon.moves)>1 else ""),
                    self.moveset_generator.encode_move(mon.moves[2] if len(mon.moves)>2 else ""),
                    self.moveset_generator.encode_move(mon.moves[3] if len(mon.moves)>3 else "")
                ]

        # 2. Party Levels (and/or Species IDs if we want them)
        party_levels = np.zeros(6, dtype=np.float32)
        for i, m in enumerate(self.party):
            party_levels[i] = m.spec.level

        # Determine slot_idx to show based on phase
        obs_slot = 0
        if self.current_phase == self.PHASE_BUILD_SPECIES:
            obs_slot = self.current_party_slot
        elif self.current_phase == self.PHASE_BUILD_MOVE:
            obs_slot = self.current_move_slot
            
        return {
            "phase": self.current_phase,
            "slot_idx": obs_slot,
            "node_idx": 0, # Placeholder
            "map_state": np.zeros(10, dtype=np.float32),
            "risk_vector": np.zeros(6, dtype=np.float32),
            "roster_count": sum(1 for m in self.roster if m.alive),
            "party_levels": party_levels,
            "opponent_preview": opp_preview
        }
