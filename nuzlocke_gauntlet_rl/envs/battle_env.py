import numpy as np
from gymnasium import spaces
from poke_env.environment.singles_env import SinglesEnv
from poke_env.battle import AbstractBattle

class BattleEnv(SinglesEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_token = 0 # 0: Safe, 1: Balanced, 2: Desperate
        
        # Workaround for SingleAgentWrapper if observation_spaces is missing
        if not hasattr(self, "observation_spaces") or not self.observation_spaces:
            low, high, shape, dtype = self.describe_embedding()
            space = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
            self.observation_spaces = {agent: space for agent in self.possible_agents}
            
        if not hasattr(self, "action_spaces") or not self.action_spaces:
            # Assuming standard Gen 8 action space size of 22
            # 4 moves + 6 switches + 4 moves (mega) + 4 moves (z-move) + 4 moves (dynamax)
            # Actually poke-env usually uses 22.
            # But let's try to get it from the class if possible, or hardcode.
            space = spaces.Discrete(22) 
            self.action_spaces = {agent: space for agent in self.possible_agents}
        
    def reset(self, seed=None, options=None):
        import sys
        
        self._last_fainted = {} # Reset fainted tracking
                    
        if options and "risk_token" in options:
            self.risk_token = options["risk_token"]
        else:
            # Default or random if not specified
            self.risk_token = np.random.randint(0, 3)
            
        # Bypass super().reset() because it forces agent1 vs agent2 challenge
        # We want to wait for an external challenge (or one triggered by us externally)
        
        if seed is not None:
            from gymnasium.utils import seeding
            self._np_random, seed = seeding.np_random(seed)
            
        # Reset agent1
        # This will clear _current_battle and wait for a new one
        self.agent1.reset_battles()
        
        # Start accepting challenges
        from poke_env.concurrency import POKE_LOOP
        import asyncio
        asyncio.run_coroutine_threadsafe(
            self.agent1.accept_challenges(None, 1), POKE_LOOP
        )
        
        # Wait for battle to be ready (request received)
        self.battle1 = self.agent1.battle_queue.get()
        
        # Ensure agents list is populated (needed for wrapper)
        self.agents = [self.agent1.username]
        
        # We don't care about agent2 or battle2 in this mode
        
        obs = {self.agent1.username: self.embed_battle(self.battle1)}
        return obs, self.get_additional_info()

    def calc_reward(self, battle: AbstractBattle) -> float:
        reward = 0.0
        
        # Win/Loss
        if battle.finished:
            if battle.won:
                reward += 1.0
            else:
                reward -= 1.0
                
        # Death penalty based on risk
        # Count fainted mons in current battle
        current_fainted = len([mon for mon in battle.team.values() if mon.fainted])
        
        # Get last fainted count for this specific battle instance
        last_fainted = self._last_fainted.get(battle, 0)
        
        new_deaths = current_fainted - last_fainted
        
        if new_deaths > 0:
            base_penalty = 0.1
            if self.risk_token == 0: # Safe
                penalty = 0.5
            elif self.risk_token == 1: # Balanced
                penalty = 0.1
            else: # Desperate
                penalty = 0.0
            
            reward -= penalty * new_deaths
            
        # Update last fainted count
        self._last_fainted[battle] = current_fainted
            
        # HP preservation reward/penalty could also be added here
        
        return reward

    def embed_battle(self, battle: AbstractBattle):
        # Features:
        # 1. Active Mon: HP (1), Status (7), Boosts (7) -> 15
        # 2. Opponent Active Mon: HP (1), Status (7), Boosts (7) -> 15
        # 3. Field: Weather (9), Terrain (5) -> 14
        # 4. Risk Token (3)
        # 5. Moves (4 * 4): Power, Acc, Eff, STAB -> 16
        # 6. Team HP (6 + 6) -> 12
        # Total: 47 + 16 + 12 = 75
        
        # Helper to encode status
        def encode_status(status):
            # Status: None, BRN, FRZ, PAR, PSN, SLP, TOX
            encoding = np.zeros(7, dtype=np.float32)
            if status is None:
                encoding[0] = 1.0
            else:
                try:
                    status_map = {'BRN': 1, 'FRZ': 2, 'PAR': 3, 'PSN': 4, 'SLP': 5, 'TOX': 6}
                    idx = status_map.get(status.name, 0)
                    encoding[idx] = 1.0
                except:
                    encoding[0] = 1.0
            return encoding

        # Helper to encode boosts
        def encode_boosts(boosts):
            encoding = np.zeros(7, dtype=np.float32)
            keys = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
            for i, k in enumerate(keys):
                val = boosts.get(k, 0)
                encoding[i] = (val + 6) / 12.0
            return encoding
            
        # Helper to encode weather
        def encode_weather(weather):
            encoding = np.zeros(9, dtype=np.float32)
            if weather is None:
                encoding[0] = 1.0
            else:
                w_map = {
                    'SUNNYDAY': 1, 'RAINDANCE': 2, 'SANDSTORM': 3, 'HAIL': 4, 'SNOW': 5,
                    'DESOLATELAND': 6, 'PRIMORDIALSEA': 7, 'DELTASTREAM': 8
                }
                name = weather.name if hasattr(weather, 'name') else str(weather).upper()
                idx = w_map.get(name, 0)
                encoding[idx] = 1.0
            return encoding

        # Helper to encode terrain
        def encode_terrain(terrain):
            encoding = np.zeros(5, dtype=np.float32)
            if terrain is None:
                encoding[0] = 1.0
            else:
                t_map = {'ELECTRIC': 1, 'GRASSY': 2, 'MISTY': 3, 'PSYCHIC': 4}
                name = terrain.name if hasattr(terrain, 'name') else str(terrain).upper()
                name = name.replace('_TERRAIN', '')
                idx = t_map.get(name, 0)
                encoding[idx] = 1.0
            return encoding

        # Active Mon
        active = battle.active_pokemon
        if active:
            my_hp = np.array([active.current_hp_fraction], dtype=np.float32)
            my_status = encode_status(active.status)
            my_boosts = encode_boosts(active.boosts)
        else:
            my_hp = np.array([0.0], dtype=np.float32)
            my_status = np.zeros(7, dtype=np.float32); my_status[0] = 1.0
            my_boosts = np.full(7, 0.5, dtype=np.float32)

        # Opponent Mon
        op = battle.opponent_active_pokemon
        if op:
            op_hp = np.array([op.current_hp_fraction], dtype=np.float32)
            op_status = encode_status(op.status)
            op_boosts = encode_boosts(op.boosts)
        else:
            op_hp = np.array([0.0], dtype=np.float32)
            op_status = np.zeros(7, dtype=np.float32); op_status[0] = 1.0
            op_boosts = np.full(7, 0.5, dtype=np.float32)

        # Field
        weather_enc = encode_weather(battle.weather)
        terrain_enc = np.zeros(5, dtype=np.float32)
        terrain_enc[0] = 1.0
        for field in battle.fields:
            f_str = str(field).upper()
            if 'ELECTRIC' in f_str: terrain_enc[1] = 1.0; terrain_enc[0] = 0.0
            elif 'GRASSY' in f_str: terrain_enc[2] = 1.0; terrain_enc[0] = 0.0
            elif 'MISTY' in f_str: terrain_enc[3] = 1.0; terrain_enc[0] = 0.0
            elif 'PSYCHIC' in f_str: terrain_enc[4] = 1.0; terrain_enc[0] = 0.0
            
        # Risk
        risk_encoding = np.zeros(3, dtype=np.float32)
        risk_encoding[self.risk_token] = 1.0
        
        # Moves
        moves_enc = np.zeros(16, dtype=np.float32)
        if active:
            # Get moves. battle.available_moves are the ones we can click.
            # But we want to encode all 4 moves if possible, or just available ones?
            # Let's encode the 4 moves in the move slots.
            # active.moves is a dict.
            move_list = list(active.moves.values())
            for i, move in enumerate(move_list[:4]):
                # Power
                power = move.base_power / 100.0
                # Accuracy
                acc = move.accuracy
                if acc is True: acc = 1.0
                else: acc = acc / 100.0
                
                # Effectiveness
                eff = 1.0
                if op:
                    eff = op.damage_multiplier(move.type)
                
                # STAB
                stab = 1.0
                if move.type in active.types:
                    stab = 1.5
                    
                base_idx = i * 4
                moves_enc[base_idx] = power
                moves_enc[base_idx+1] = acc
                moves_enc[base_idx+2] = eff
                moves_enc[base_idx+3] = stab
                
        # Team HP
        my_team_hp = np.zeros(6, dtype=np.float32)
        op_team_hp = np.zeros(6, dtype=np.float32)
        
        # My Team
        # battle.team is a dict of mon_ident -> Pokemon
        for i, mon in enumerate(battle.team.values()):
            if i < 6:
                my_team_hp[i] = mon.current_hp_fraction
                
        # Opponent Team
        # battle.opponent_team is a dict
        for i, mon in enumerate(battle.opponent_team.values()):
            if i < 6:
                op_team_hp[i] = mon.current_hp_fraction
        
        return np.concatenate([
            my_hp, my_status, my_boosts,
            op_hp, op_status, op_boosts,
            weather_enc, terrain_enc,
            risk_encoding,
            moves_enc,
            my_team_hp, op_team_hp
        ])

    def describe_embedding(self):
        # Total dims: 47 + 16 + 12 = 75
        dims = 75
        return (
            np.zeros(dims, dtype=np.float32),
            np.ones(dims, dtype=np.float32),
            (dims,),
            np.float32
        )

    def step(self, actions):
        # Only handle agent1
        agent_username = self.agent1.username
        if agent_username in actions:
            action = actions[agent_username]
            order = self.action_to_order(action, self.battle1)
            self.agent1.order_queue.put(order)
            
        # Wait for result
        self.battle1 = self.agent1.battle_queue.get()
        
        # Calc reward, term, trunc
        reward = self.calc_reward(self.battle1)
        term, trunc = self.calc_term_trunc(self.battle1)
        
        obs = {agent_username: self.embed_battle(self.battle1)}
        rewards = {agent_username: reward}
        terminated = {agent_username: term}
        truncated = {agent_username: trunc}
        infos = {agent_username: {}}
        
        return obs, rewards, terminated, truncated, infos

    def action_to_order(self, action, battle, fake=False, strict=True):
        from poke_env.player.battle_order import BattleOrder
        try:
            return super().action_to_order(action, battle, fake=fake, strict=strict)
        except (AssertionError, ValueError, IndexError) as e:
            # Fallback to a random valid move
            # print(f"DEBUG: Invalid action {action} for battle {battle.battle_tag}: {e}. Picking random move.")
            return self.choose_random_move(battle)

    def choose_random_move(self, battle):
        from poke_env.player.battle_order import SingleBattleOrder, ForfeitBattleOrder
        import random
        
        if battle.available_moves:
            return SingleBattleOrder(random.choice(battle.available_moves))
            
        if battle.available_switches:
            return SingleBattleOrder(random.choice(battle.available_switches))
            
        return ForfeitBattleOrder()
        
    def close(self):
        # Override close to avoid checking battle2
        # self.agent1.close() # Player does not have close() method!
        # self.agent2.close() # agent2 is not used
        
        # We just need to close agent1.
        # But Player doesn't have close.
        # Maybe we should close ps_client?
        # self.agent1.ps_client.stop_listening() # If it exists
        pass


