import numpy as np
import zlib
from poke_env.battle import AbstractBattle

class BattleEmbedder:
    """
    Standalone class to convert poke-env Battle objects into numpy observations.
    Refactored from BattleEnv.
    """
    def __init__(self):
        self.risk_token = 0 # Default

    def _hash_to_bin(self, value, n_bits):
        """Hashes a string/value to a binary vector of size n_bits."""
        enc = np.zeros(n_bits, dtype=np.float32)
        if not value: return enc
        
        s = str(value).lower().encode('utf-8')
        h = zlib.crc32(s)
        
        for i in range(n_bits):
            if (h >> i) & 1:
                enc[i] = 1.0
        return enc

    def describe_embedding(self):
        # Total dims detailed calculation:
        # Base (75) + Global (40) = 115
        # Teammates (6 * 152) = 912
        # where 152 = 77 (Old) + 6 (Stats) + 11 (Species) + 9 (Item) + 9 (Abi) + 40 (MoveIDs)
        # Active Perfect Info:
        # Op Actions (Moves 16 + Items 5 + Stats 5 + Ability 20) -> 46
        # + Op Global (40) + Op Active IDs (69) + My Active IDs (69)
        # Grand Total:
        # Team (912) + Basics (75) + PerfectInfo (46) + IDs (138) + Global (37) ~= 1208
        dims = 1208
        return (
            np.zeros(dims, dtype=np.float32),
            np.ones(dims, dtype=np.float32),
            (dims,),
            np.float32
        )

    # ... [Keep Helpers _get_active_move_pp_fraction etc from previous steps] ...

    def encode_teammate(self, mon, active_op):
        # Base: 16 (Moves) + 5 (Items) + 20 (Ability) + 36 (Types) = 77
        # Stats: 6
        # IDs: 11 (Species) + 9 (Item) + 9 (Ability) + 40 (Moves) = 69
        # Total: 77 + 6 + 69 = 152
        enc = np.zeros(152, dtype=np.float32)
        
        # Moves (0-15) & Move IDs (From 112-151)
        move_list = list(mon.moves.values())
        for i, move in enumerate(move_list[:4]):
            power = move.base_power / 100.0
            acc = (move.accuracy / 100.0) if (move.accuracy is not True) else 1.0
            eff = active_op.damage_multiplier(move.type) if active_op else 1.0
            stab = 1.5 if move.type in mon.types else 1.0
            
            base_idx = i * 4
            enc[base_idx] = power
            enc[base_idx+1] = acc
            enc[base_idx+2] = eff
            enc[base_idx+3] = stab
            
            # ID (10 bits)
            id_enc = self._hash_to_bin(move.id, 10)
            id_start = 112 + (i * 10)
            enc[id_start : id_start+10] = id_enc
        
        # Item (16-20) & Item ID (92-100)
        item = mon.item
        if item:
            if 'choice' in item or ' scarf' in item or ' specs' in item or ' band' in item: enc[16] = 1.0
            elif 'life' in item and 'orb' in item: enc[17] = 1.0
            elif 'focus' in item and 'sash' in item: enc[18] = 1.0
            elif 'leftovers' in item: enc[19] = 1.0
            else: enc[20] = 1.0
            
            enc[92:101] = self._hash_to_bin(item, 9)
        
        # Ability (21-40) & Ability ID (101-109)
        abi_enc = self.encode_ability(mon.ability)
        enc[21:41] = abi_enc
        enc[101:110] = self._hash_to_bin(mon.ability, 9)
        
        # Types (41-76) -> Type 1 (41-58) + Type 2 (59-76)
        if mon.types:
            t1_enc = self.encode_type(mon.types[0])
            enc[41:59] = t1_enc
            if len(mon.types) > 1:
                t2_enc = self.encode_type(mon.types[1])
                enc[59:77] = t2_enc
                
        # Base Stats (77-82)
        enc[77] = mon.base_stats['hp'] / 255.0
        enc[78] = mon.base_stats['atk'] / 255.0
        enc[79] = mon.base_stats['def'] / 255.0
        enc[80] = mon.base_stats['spa'] / 255.0
        enc[81] = mon.base_stats['spd'] / 255.0
        enc[82] = mon.base_stats['spe'] / 255.0
        
        # Species ID (83-93) - 11 bits
        enc[83:94] = self._hash_to_bin(mon.species, 11)
        
        return enc


    def encode_type(self, type_obj):
        # 18 Standard Types
        encoding = np.zeros(18, dtype=np.float32)
        if not type_obj: return encoding
        type_map = {
            'BUG': 0, 'DARK': 1, 'DRAGON': 2, 'ELECTRIC': 3, 'FAIRY': 4, 'FIGHTING': 5, 
            'FIRE': 6, 'FLYING': 7, 'GHOST': 8, 'GRASS': 9, 'GROUND': 10, 'ICE': 11, 
            'NORMAL': 12, 'POISON': 13, 'PSYCHIC': 14, 'ROCK': 15, 'STEEL': 16, 'WATER': 17
        }
        name = type_obj.name.upper() if hasattr(type_obj, 'name') else str(type_obj).upper()
        idx = type_map.get(name, -1)
        if idx >= 0: encoding[idx] = 1.0
        return encoding

    def encode_status(self, status):
        enc = np.zeros(7, dtype=np.float32)
        if status:
            if status.name == 'SLP': enc[1] = 1.0
            elif status.name == 'PSN': enc[2] = 1.0
            elif status.name == 'BRN': enc[3] = 1.0
            elif status.name == 'FRZ': enc[4] = 1.0
            elif status.name == 'PAR': enc[5] = 1.0
            elif status.name == 'TOX': enc[6] = 1.0
        else: enc[0] = 1.0 # No status
        return enc

    def encode_boosts(self, boosts):
        enc = np.full(7, 0.5, dtype=np.float32) # Default to 0.5 (no change)
        if boosts:
            enc[0] = (boosts.get('atk', 0) + 6) / 12.0
            enc[1] = (boosts.get('def', 0) + 6) / 12.0
            enc[2] = (boosts.get('spa', 0) + 6) / 12.0
            enc[3] = (boosts.get('spd', 0) + 6) / 12.0
            enc[4] = (boosts.get('spe', 0) + 6) / 12.0
            enc[5] = (boosts.get('accuracy', 0) + 6) / 12.0
            enc[6] = (boosts.get('evasion', 0) + 6) / 12.0
        return enc

    def encode_weather(self, weather):
        enc = np.zeros(6, dtype=np.float32)
        if weather:
            w_str = str(weather).upper()
            if 'SUN' in w_str: enc[1] = 1.0
            elif 'RAIN' in w_str: enc[2] = 1.0
            elif 'SAND' in w_str: enc[3] = 1.0
            elif 'HAIL' in w_str: enc[4] = 1.0
            elif 'FOG' in w_str: enc[5] = 1.0
        else: enc[0] = 1.0 # No weather
        return enc

    def encode_ability(self, ability):
        # 20 most common abilities, plus 1 for 'other'
        # This is a simplified example, a real implementation would need a comprehensive list
        ability_map = {
            'levitate': 0, 'intimidate': 1, 'regenerator': 2, 'prankster': 3,
            'protean': 4, 'libero': 5, 'drought': 6, 'drizzle': 7,
            'sandstream': 8, 'snowwarning': 9, 'unburden': 10, 'speedboost': 11,
            'magicbounce': 12, 'shadowtag': 13, 'arenatrap': 14, 'magnetpull': 15,
            'toughclaws': 16, 'adaptability': 17, 'sheerforce': 18, 'multiscale': 19
        }
        enc = np.zeros(20, dtype=np.float32)
        if ability:
            name = str(ability).lower()
            idx = ability_map.get(name, -1)
            if idx >= 0: enc[idx] = 1.0
        return enc

    def _get_side_conditions(self, conditions):
        # 10 dims
        enc = np.zeros(10, dtype=np.float32)
        if not conditions: return enc
        for cond in conditions:
            name = str(cond).upper()
            if 'STEALTHROCK' in name: enc[0] = 1.0
            elif 'SPIKES' in name: enc[1] = 0.33 if '1' in name else (0.66 if '2' in name else 1.0)
            elif 'TOXICSPIKES' in name: enc[2] = 1.0
            elif 'STICKYWEB' in name: enc[3] = 1.0
            elif 'LIGHTSCREEN' in name: enc[4] = 1.0
            elif 'REFLECT' in name: enc[5] = 1.0
            elif 'AURORAVEIL' in name: enc[6] = 1.0
            elif 'TAILWIND' in name: enc[7] = 1.0
            elif 'MIST' in name: enc[8] = 1.0
            elif 'SAFEGUARD' in name: enc[9] = 1.0
        return enc

    def _get_volatile_status(self, mon):
        # 6 dims
        enc = np.zeros(6, dtype=np.float32)
        if not mon: return enc
        for v in mon.effects:
            name = str(v).upper()
            if 'CONFUSION' in name: enc[0] = 1.0
            elif 'TAUNT' in name: enc[1] = 1.0
            elif 'LEECH' in name: enc[2] = 1.0
            elif 'SUBSTITUTE' in name: enc[3] = 1.0
            elif 'ENCORE' in name: enc[4] = 1.0
            elif 'SALT' in name: enc[5] = 1.0
        return enc

    def _get_timers(self, battle):
        # 4 dims: Weather, Terrain, MySide, OpSide (approximate or raw count if available)
        # Poke-env wraps Showdown, which sometimes hides exact timers.
        # We will return 0.0 for now but keep the slot.
        return np.zeros(4, dtype=np.float32)

    def _get_active_move_pp_fraction(self, battle):
         # 4 dims
         enc = np.zeros(4, dtype=np.float32)
         active = battle.active_pokemon
         if active:
             for i, move in enumerate(list(active.moves.values())[:4]):
                 if move.max_pp > 0:
                    enc[i] = move.current_pp / move.max_pp
         return enc

    def embed_battle(self, battle: AbstractBattle, risk_token=0, opponent_team=None):
        # Helper functions moved to methods
        
        # ... [Embed Logic Matches] ...
        # Active Mon
        active = battle.active_pokemon
        if active:
            my_hp = np.array([active.current_hp_fraction], dtype=np.float32)
            my_status = self.encode_status(active.status)
            my_boosts = self.encode_boosts(active.boosts)
        else:
            my_hp = np.array([0.0], dtype=np.float32)
            my_status = np.zeros(7, dtype=np.float32); my_status[0] = 1.0
            my_boosts = np.full(7, 0.5, dtype=np.float32)

        # Opponent Mon
        op = battle.opponent_active_pokemon
        if op:
            op_hp = np.array([op.current_hp_fraction], dtype=np.float32)
            op_status = self.encode_status(op.status)
            op_boosts = self.encode_boosts(op.boosts)
        else:
            op_hp = np.array([0.0], dtype=np.float32)
            op_status = np.zeros(7, dtype=np.float32); op_status[0] = 1.0
            op_boosts = np.full(7, 0.5, dtype=np.float32)

        # Field
        weather_enc = self.encode_weather(battle.weather)
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
        if 0 <= risk_token < 3: risk_encoding[risk_token] = 1.0
        
        # My Moves
        moves_enc = np.zeros(16, dtype=np.float32)
        if active:
            move_list = list(active.moves.values())
            for i, move in enumerate(move_list[:4]):
                power = move.base_power / 100.0
                acc = (move.accuracy / 100.0) if (move.accuracy is not True) else 1.0
                eff = op.damage_multiplier(move.type) if op else 1.0
                stab = 1.5 if move.type in active.types else 1.0
                base_idx = i * 4
                moves_enc[base_idx] = power
                moves_enc[base_idx+1] = acc
                moves_enc[base_idx+2] = eff
                moves_enc[base_idx+3] = stab
                
        # Team HP
        my_team_hp = np.zeros(6, dtype=np.float32)
        op_team_hp = np.zeros(6, dtype=np.float32)
        
        for i, mon in enumerate(battle.team.values()):
             if i < 6: my_team_hp[i] = mon.current_hp_fraction
        for i, mon in enumerate(battle.opponent_team.values()):
             if i < 6: op_team_hp[i] = mon.current_hp_fraction
             
        # --- PERFECT INFO ---
        op_moves_enc = np.zeros(16, dtype=np.float32)
        op_items_enc = np.zeros(5, dtype=np.float32)
        rel_stats_enc = np.zeros(5, dtype=np.float32)
        op_ability_enc = np.zeros(20, dtype=np.float32) # NEW
        
        true_op = None
        if opponent_team and op:
             for mon in opponent_team.values():
                 if mon.species == op.species:
                     true_op = mon
                     break
        
        if true_op:
            # 1. Op Moves
            move_list = list(true_op.moves.values())
            for i, move in enumerate(move_list[:4]):
                power = move.base_power / 100.0
                acc = (move.accuracy / 100.0) if (move.accuracy is not True) else 1.0
                eff = active.damage_multiplier(move.type) if active else 1.0
                stab = 1.5 if move.type in true_op.types else 1.0
                base_idx = i * 4
                op_moves_enc[base_idx] = power
                op_moves_enc[base_idx+1] = acc
                op_moves_enc[base_idx+2] = eff
                op_moves_enc[base_idx+3] = stab

            # 2. Op Items
            item = true_op.item
            if item:
                if 'choice' in item or ' scarf' in item or ' specs' in item or ' band' in item: op_items_enc[0] = 1.0
                elif 'life' in item and 'orb' in item: op_items_enc[1] = 1.0
                elif 'focus' in item and 'sash' in item: op_items_enc[2] = 1.0
                elif 'leftovers' in item: op_items_enc[3] = 1.0
                else: op_items_enc[4] = 1.0
            
            # 3. Relative Stats
            if active:
                my_stats = active.stats
                op_stats = true_op.stats
                s_ratio = my_stats['spe'] / (op_stats['spe'] + 0.1)
                rel_stats_enc[0] = np.clip(s_ratio, 0, 3.0)
                ad_ratio = my_stats['atk'] / (op_stats['def'] + 0.1)
                rel_stats_enc[1] = np.clip(ad_ratio, 0, 3.0)
                sd_ratio = my_stats['spa'] / (op_stats['spd'] + 0.1)
                rel_stats_enc[2] = np.clip(sd_ratio, 0, 3.0)
                da_ratio = my_stats['def'] / (op_stats['atk'] + 0.1)
                rel_stats_enc[3] = np.clip(da_ratio, 0, 3.0)
                ds_ratio = my_stats['spd'] / (op_stats['spa'] + 0.1)
                rel_stats_enc[4] = np.clip(ds_ratio, 0, 3.0)

            # 4. Op Ability
            if true_op.ability:
                 op_ability_enc = self.encode_ability(true_op.ability)
            elif op.ability: # Fallback to known ability if perfect info missing?
                 op_ability_enc = self.encode_ability(op.ability)

        # --- TEAMMATE DETAILS ---
        # 6 teammates * 152 dims = 912
        my_team_details = np.zeros(912, dtype=np.float32)
        team_list = list(battle.team.values())
        op = battle.opponent_active_pokemon
        for i in range(6):
            if i < len(team_list):
                 mon = team_list[i]
                 enc = self.encode_teammate(mon, op)
                 start = i * 152
                 end = start + 152
                 my_team_details[start:end] = enc

        # --- GLOBAL BATTLE STATE (40 dims) ---
        my_side_cond = self._get_side_conditions(battle.side_conditions)
        op_side_cond = self._get_side_conditions(battle.opponent_side_conditions)
        my_volatile = self._get_volatile_status(active)
        op_volatile = self._get_volatile_status(op)
        timers = self._get_timers(battle)
        pp_frac = self._get_active_move_pp_fraction(battle)
        
        global_state = np.concatenate([
            my_side_cond, op_side_cond,
            my_volatile, op_volatile,
            timers, pp_frac
        ])

        # --- ACTIVE DISTINCT IDs (138 dims) ---
        # My Active (Species 11 + Item 9 + Abi 9 + Moves 40 = 69)
        my_active_ids = np.zeros(69, dtype=np.float32)
        if active:
             my_active_ids[0:11] = self._hash_to_bin(active.species, 11)
             my_active_ids[11:20] = self._hash_to_bin(active.item, 9)
             my_active_ids[20:29] = self._hash_to_bin(active.ability, 9)
             moves = list(active.moves.values())
             for i, m in enumerate(moves[:4]):
                  start_id = 29 + (i * 10)
                  my_active_ids[start_id : start_id+10] = self._hash_to_bin(m.id, 10)

        # Op Active (Species 11 + Item 9 + Abi 9 + Moves 40 = 69)
        op_active_ids = np.zeros(69, dtype=np.float32)
        if op:
             op_active_ids[0:11] = self._hash_to_bin(op.species, 11)
             op_active_ids[11:20] = self._hash_to_bin(op.item, 9)
             op_active_ids[20:29] = self._hash_to_bin(op.ability, 9)
             
             # For moves, we use what we know (op_moves_enc was computed above, but that's functional)
             # Here we want IDs if we know them.
             # We can try to get them from true_op if available, else standard move tracking
             known_moves = []
             if true_op:
                  known_moves = list(true_op.moves.values())
             else:
                  known_moves = list(op.moves.values()) # Revealed moves
                  
             for i, m in enumerate(known_moves[:4]):
                  start_id = 29 + (i * 10)
                  op_active_ids[start_id : start_id+10] = self._hash_to_bin(m.id, 10)

        # Order: [Teammates (912)] + [Global Context (~374)]
        # This makes slicing obs[:, :912] easy for the Transformer
        return np.concatenate([
            my_team_details,    # 0 -> 912
            # --- GLOBAL/ACTIVE CONTEXT ---
            my_hp, my_status, my_boosts,
            op_hp, op_status, op_boosts,
            weather_enc, terrain_enc,
            risk_encoding,
            moves_enc,
            my_team_hp, op_team_hp,
            op_moves_enc, op_items_enc, rel_stats_enc,
            op_ability_enc,
            global_state,
            my_active_ids,
            op_active_ids
        ])
