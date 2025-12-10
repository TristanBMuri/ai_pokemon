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
    load_team_rocket, 
    load_complete_gauntlet  # Ensure this is available
)
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
    
    # State Constants
    PHASE_DECISION = 0
    PHASE_SELECT_MEMBER = 1
    PHASE_SELECT_MOVE = 2
    PHASE_SELECT_STARTER = 3
    
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
            "phase": spaces.Discrete(5), # 0=Decision, 1=Mem, 2=Move, 3=Starter
            "slot_idx": spaces.Discrete(7), # Which member 0-5, or move 0-3
            "trainer_idx": spaces.Discrete(100),
            "roster_count": spaces.Discrete(self.max_roster_size + 1),
            "party_levels": spaces.Box(low=0, high=100, shape=(6,), dtype=np.float32), # Current team state?
            "opponent_preview": spaces.Box(low=0, high=1000, shape=(6, 14), dtype=np.int32)
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
        self.current_trainer_idx = 0
        self.visited_routes: Set[str] = set()
        self.party: List[MonInstance] = [] # Current ACTIVE party
        self.rebuild_count = 0 # Track rebuilds for current trainer
        
        # Building State
        self.build_party_specs: List[PokemonSpec] = []
        self.build_current_mon: Optional[MonInstance] = None
        self.build_current_moves: List[str] = []
        
        self.current_phase = self.PHASE_DECISION
        self.current_slot = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_trainer_idx = 0
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
        
        # Start at DECISION. If no party, user MUST rebuild (masking handles this?).
        self.current_phase = self.PHASE_SELECT_STARTER # Start here if roster empty
        
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
                
            self.current_phase = self.PHASE_DECISION
            return self._get_obs(), 0, False, False, {"metrics": {}}

        # PHASE: DECISION
        elif self.current_phase == self.PHASE_DECISION:
            if action == 1: # REBUILD
                 # Reward penalty only if this is not the first rebuild (Chicken Out)
                 if self.rebuild_count > 0:
                     reward = -0.1 # Punishment for indecision/chickening out
                 
                 self.rebuild_count += 1
                 self.current_phase = self.PHASE_SELECT_MEMBER
                 self.current_slot = 0
                 self.build_party_specs = []
                 self.party = [] # Clear old party
                 
            else: # FIGHT (0)
                 # Executed when agent is confident
                 outcome = self._run_battle()
                 return outcome # _run_battle returns full step tuple

        # PHASE: SELECT MEMBER
        elif self.current_phase == self.PHASE_SELECT_MEMBER:
             # Action is Roster Index
             if 0 <= action < len(self.roster):
                 mon = self.roster[action] # Get instance
                 if mon.alive and mon not in self.party: 
                     # Start building this Mon
                     self.build_current_mon = mon
                     self.build_current_moves = []
                     
                     # Switch to MOVES
                     self.current_phase = self.PHASE_SELECT_MOVE
                     self.current_slot = 0
             
        # PHASE: SELECT MOVES
        elif self.current_phase == self.PHASE_SELECT_MOVE:
             # Action is Move ID
             move_name = self.moveset_generator.get_move_name(action)
             
             # Re-validate against learnset (Safety Check)
             # Even if masked, sometimes PPO can be weird or if mask calc differed.
             if self.build_current_mon and move_name:
                 lvl = self.build_current_mon.spec.level
                 valid_ids = self.moveset_generator.get_learnable_moves_ids_at_level(self.build_current_mon.spec.species, lvl)
                 if action in valid_ids:
                     self.build_current_moves.append(move_name)
                 # Else: Invalid move picked (despite mask). Ignore.
                 
             self.current_slot += 1
             
             # Check if done with moves
             if self.current_slot >= 4:
                 # Finalize Mon
                 if not self.build_current_moves:
                     # Force at least one move to prevent Showdown auto-fill chaos
                     lvl = self.build_current_mon.spec.level
                     avail = self.moveset_generator.get_learnable_moves_at_level(self.build_current_mon.spec.species, lvl)
                     if avail:
                         self.build_current_moves.append(avail[0])
                     else:
                         self.build_current_moves.append("struggle")
                         
                 self.build_current_mon.spec.moves = self.build_current_moves
                 self.party.append(self.build_current_mon)
                 
                 # Check if team full OR no more roster candidates
                 available = [m for m in self.roster if m not in self.party and m.alive]
                 
                 if len(self.party) >= 6 or not available:
                     # Done building. Return to DECISION
                     self.current_phase = self.PHASE_DECISION
                 else:
                     # Next Member
                     self.current_phase = self.PHASE_SELECT_MEMBER
                     self.current_slot = len(self.party)
        
        return self._get_obs(), reward, terminated, truncated, info

    def _run_battle(self):
         current_trainer = self.gauntlet_template.trainers[self.current_trainer_idx]
         
         # Level Scaling
         if current_trainer.team:
            target_level = max(p.level for p in current_trainer.team)
            target_level = max(5, target_level)
            for m in self.party:
                m.spec.level = target_level
         
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
         metrics["trainer_idx"] = self.current_trainer_idx - 1 if win else self.current_trainer_idx # If won, we incremented. Revert for log.
         if win: metrics["trainer_idx"] = self.current_trainer_idx - 1 # We incremented above
         else: metrics["trainer_idx"] = self.current_trainer_idx
         
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
             if self.party: mask[0] = True # Can fight if party exists
             mask[1] = True # Always can rebuild
             
        elif self.current_phase == self.PHASE_SELECT_MEMBER:
             # Mask valid roster indices
             for i, m in enumerate(self.roster):
                 if m.alive and m not in self.party:
                     mask[i] = True
                     
        elif self.current_phase == self.PHASE_SELECT_MOVE:
             # Mask valid move IDs for current mon
             if self.build_current_mon:
                 # Use Cap or Current Level? Nuzlocke usually caps at next gym leader.
                 # Let's use current_level of the Mon (which is scaled to Cap)
                 lvl = self.build_current_mon.spec.level
                 learnable_ids = self.moveset_generator.get_learnable_moves_ids_at_level(self.build_current_mon.spec.species, lvl)
                 # Also exclude already picked moves?
                 picked_names = set(self.build_current_moves)
                 
                 for mid in learnable_ids:
                     mname = self.moveset_generator.get_move_name(mid)
                     if mname not in picked_names:
                         mask[mid] = True
                         
        return mask

    def _get_obs(self):
        # 1. Opponent Preview
        opp_preview = np.zeros((6, 14), dtype=np.int32)
        if self.current_trainer_idx < len(self.gauntlet_template.trainers):
            trainer = self.gauntlet_template.trainers[self.current_trainer_idx]
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

        return {
            "phase": self.current_phase,
            "slot_idx": self.current_slot,
            "trainer_idx": self.current_trainer_idx,
            "roster_count": sum(1 for m in self.roster if m.alive),
            "party_levels": party_levels,
            "opponent_preview": opp_preview
        }
