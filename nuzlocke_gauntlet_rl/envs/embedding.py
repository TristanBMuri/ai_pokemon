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
        # Total dims: 
        # Base: 47 + 16 (MyMoves) + 12 (TeamHP) = 75
        # Perfect Info:
        # + 16 (OpMoves) 
        # + 5 (OpItems) 
        # + 5 (RelativeStats)
        # + 20 (OpAbility) <--- NEW
        # Total: 75 + 16 + 5 + 5 + 20 = 121
        dims = 583
        return (
            np.zeros(583, dtype=np.float32),
            np.ones(583, dtype=np.float32),
            (583,),
            np.float32
        )

    def embed_battle(self, battle: AbstractBattle, risk_token=0, opponent_team=None):
        # 1. My Active (HP, Status, Boosts) -> 1 + 7 + 7 = 15
        # 2. Op Active (HP, Status, Boosts) -> 1 + 7 + 7 = 15
        # 3. Field (Weather, Terrain) -> 9 + 5 = 14
        # 4. Risk Token (3) -> 3
        # 5. My Moves (Power, Acc, Eff, STAB) -> 16
        # 6. Team HP (My 6 + Op 6) -> 12
        # --- Base Total: 75 ---
        # 7. Op Moves (16) -> 16
        # 8. Op Items (5) -> 5
        # 9. Relative Stats (5) -> 5
        # 10. Op Ability (20) -> 20
        # --- Perfect Info Total: 121 ---
        # 11. My Team Details (6 * (16 Moves + 5 Items + 20 Ability + 36 Types)) -> 6 * 77 = 462
        # --- Grand Total: 121 + 462 = 583 ---
        
        # ... [Previous Helpers] ...
        def encode_status(status):
            encoding = np.zeros(7, dtype=np.float32)
            if status is None: encoding[0] = 1.0
            else:
                status_map = {'BRN': 1, 'FRZ': 2, 'PAR': 3, 'PSN': 4, 'SLP': 5, 'TOX': 6}
                try: idx = status_map.get(status.name, 0)
                except: idx = 0
                encoding[idx] = 1.0
            return encoding

        def encode_boosts(boosts):
            encoding = np.zeros(7, dtype=np.float32)
            keys = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']
            for i, k in enumerate(keys):
                encoding[i] = (boosts.get(k, 0) + 6) / 12.0
            return encoding
            
        def encode_weather(weather):
            encoding = np.zeros(9, dtype=np.float32)
            if weather is None: encoding[0] = 1.0
            else:
                w_map = {'SUNNYDAY': 1, 'RAINDANCE': 2, 'SANDSTORM': 3, 'HAIL': 4, 'SNOW': 5, 'DESOLATELAND': 6, 'PRIMORDIALSEA': 7, 'DELTASTREAM': 8}
                name = weather.name if hasattr(weather, 'name') else str(weather).upper()
                encoding[w_map.get(name, 0)] = 1.0
            return encoding

        def encode_terrain(terrain):
            encoding = np.zeros(5, dtype=np.float32)
            if terrain is None: encoding[0] = 1.0
            else:
                t_map = {'ELECTRIC': 1, 'GRASSY': 2, 'MISTY': 3, 'PSYCHIC': 4}
                name = terrain.name if hasattr(terrain, 'name') else str(terrain).upper().replace('_TERRAIN', '')
                encoding[t_map.get(name, 0)] = 1.0
            return encoding

        def encode_ability(ability_name):
            # Functional Categories (20 dims)
            # 0: Levitate/Earth Eater (Ground Immunity)
            # 1: Intimidate (Atk Drop)
            # 2: Regenerator (Heal on Switch)
            # 3: Speed Boost/Unburden (Speed)
            # 4: Huge/Pure Power/Gorilla (Atk Boost)
            # 5: Prankster/Gale/Triage (Priority)
            # 6: Libero/Protean (Type Change)
            # 7: Mold Breaker/Tera/Turbo (Ignore Ability)
            # 8: Weather Setter (Drizzle/Drought/Stream/Snow)
            # 9: Terrain Setter (Surge)
            # 10: Absorb/Storm (Type Immunity + Boost)
            # 11: Multiscale/Shadow (Damage Reduction)
            # 12: Wonder Guard (Immunity)
            # 13: Magic Bounce (Reflect Status)
            # 14: Beast Boost/Moxie (Snowball)
            # 15: Technician (Weak Move Boost)
            # 16: Guts/Toxic/Flare (Status Boost)
            # 17: Sheer Force (Sec Effect Boost)
            # 18: Unaware (Ignore Boosts)
            # 19: Other/None
            
            enc = np.zeros(20, dtype=np.float32)
            if not ability_name:
                enc[19] = 1.0
                return enc
                
            a = ability_name.lower().replace(" ", "")
            
            if a in ['levitate', 'eartheater']: enc[0] = 1.0
            elif a in ['intimidate']: enc[1] = 1.0
            elif a in ['regenerator', 'naturalcure']: enc[2] = 1.0
            elif a in ['speedboost', 'unburden', 'motordrive', 'steamengine']: enc[3] = 1.0
            elif a in ['hugepower', 'purepower', 'gorillatactics']: enc[4] = 1.0
            elif a in ['prankster', 'galewings', 'triage']: enc[5] = 1.0
            elif a in ['libero', 'protean', 'colorchange']: enc[6] = 1.0
            elif a in ['moldbreaker', 'teravolt', 'turboblaze']: enc[7] = 1.0
            elif a in ['drizzle', 'drought', 'sandstream', 'snowwarning', 'desolateland', 'primordialsea']: enc[8] = 1.0
            elif 'surge' in a: enc[9] = 1.0
            elif a in ['voltabsorb', 'lightningrod', 'flashfire', 'sapsipper', 'stormdrain', 'dryskin', 'waterabsorb']: enc[10] = 1.0
            elif a in ['multiscale', 'shadowshield', 'filter', 'solidrock', 'prismarmor']: enc[11] = 1.0
            elif a in ['wonderguard']: enc[12] = 1.0
            elif a in ['magicbounce', 'magicguard']: enc[13] = 1.0
            elif a in ['beastboost', 'moxie', 'grimneigh', 'chillingneigh', 'soulheart']: enc[14] = 1.0
            elif a in ['technician', 'toughclaws', 'steelyspirit', 'punkrock', 'strongjaw', 'sharpness']: enc[15] = 1.0
            elif a in ['guts', 'toxicboost', 'flareboost', 'marvelscale', 'quickfeet']: enc[16] = 1.0
            elif a in ['sheerforce']: enc[17] = 1.0
            elif a in ['unaware']: enc[18] = 1.0
            else: enc[19] = 1.0
            
            return enc

        def encode_type(type_obj):
            # 18 Standard Types
            # BUG, DARK, DRAGON, ELECTRIC, FAIRY, FIGHTING, FIRE, FLYING, GHOST, GRASS, GROUND, ICE, NORMAL, POISON, PSYCHIC, ROCK, STEEL, WATER
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

        def encode_teammate(mon, active_op):
            # 16 (Moves) + 5 (Items) + 20 (Ability) + 36 (Types) = 77
            enc = np.zeros(77, dtype=np.float32)
            
            # Moves (0-15)
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
            
            # Item (16-20)
            item = mon.item
            if item:
                if 'choice' in item or ' scarf' in item or ' specs' in item or ' band' in item: enc[16] = 1.0
                elif 'life' in item and 'orb' in item: enc[17] = 1.0
                elif 'focus' in item and 'sash' in item: enc[18] = 1.0
                elif 'leftovers' in item: enc[19] = 1.0
                else: enc[20] = 1.0
            
            # Ability (21-40)
            abi_enc = encode_ability(mon.ability)
            enc[21:41] = abi_enc
            
            # Types (41-76) -> Type 1 (41-58) + Type 2 (59-76)
            # mon.types is a tuple/list of Type objects
            if mon.types:
                t1_enc = encode_type(mon.types[0])
                enc[41:59] = t1_enc
                if len(mon.types) > 1:
                    t2_enc = encode_type(mon.types[1])
                    enc[59:77] = t2_enc
            
            return enc

        # ... [Embed Logic Matches] ...
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
                 op_ability_enc = encode_ability(true_op.ability)
            elif op.ability: # Fallback to known ability if perfect info missing?
                 op_ability_enc = encode_ability(op.ability)

        # --- TEAMMATE DETAILS ---
        # 6 teammates * 77 dims = 462
        my_team_details = np.zeros(462, dtype=np.float32)
        team_list = list(battle.team.values())
        for i in range(6):
            if i < len(team_list):
                 mon = team_list[i]
                 # We use 'op' (Opponent Active) to check effectiveness
                 # If op is None, effectiveness 1.0 (handled in helper)
                 enc = encode_teammate(mon, op)
                 start = i * 77
                 end = start + 77
                 my_team_details[start:end] = enc

        return np.concatenate([
            my_hp, my_status, my_boosts,
            op_hp, op_status, op_boosts,
            weather_enc, terrain_enc,
            risk_encoding,
            moves_enc,
            my_team_hp, op_team_hp,
            op_moves_enc, op_items_enc, rel_stats_enc,
            op_ability_enc,
            my_team_details
        ])
