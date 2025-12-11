import numpy as np
from poke_env.battle import AbstractBattle

class BattleEmbedder:
    """
    Standalone class to convert poke-env Battle objects into numpy observations.
    Refactored from BattleEnv.
    """
    def __init__(self):
        self.risk_token = 0 # Default

    def describe_embedding(self):
        # Total dims: 47 + 16 + 12 = 75
        dims = 75
        return (
            np.zeros(dims, dtype=np.float32),
            np.ones(dims, dtype=np.float32),
            (dims,),
            np.float32
        )

    def embed_battle(self, battle: AbstractBattle, risk_token=0):
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
        if 0 <= risk_token < 3:
            risk_encoding[risk_token] = 1.0
        
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
