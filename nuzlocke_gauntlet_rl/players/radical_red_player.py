from poke_env.player import Player
from poke_env.battle import Pokemon, Move, AbstractBattle
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
import random

class RadicalRedPlayer(Player):
    """
    A custom player that mimics the AI heuristics of Pokemon Radical Red (Hardcore Mode).
    Based on research into CFRU (Complete Fire Red Upgrade) AI logic.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anti_abuse_counter = 0
        self.last_opponent_mon = None
        
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # Update Anti-Abuse Counter (CFRU Logic: +3 on switch, -1 per turn)
        if battle.opponent_active_pokemon:
            if self.last_opponent_mon and self.last_opponent_mon != battle.opponent_active_pokemon:
                self.anti_abuse_counter += 3
            self.last_opponent_mon = battle.opponent_active_pokemon
            
        self.anti_abuse_counter = max(0, self.anti_abuse_counter - 1)
        
        # 1. Switch Logic
        # Try to switch if we are dead to rights or can absorb a predicted move
        best_switch = self._choose_best_switch(battle)
        if best_switch and self._should_switch(battle):
             return best_switch

        # 2. Check for Revenge Kill (if active is fainted or simply forced)
        if battle.active_pokemon.fainted:
             # _choose_best_switch handles fainted cases too usually, but let's be safe
             return best_switch if best_switch else self.choose_random_move(battle)

        # 3. Move Scoring System
        best_move_order = self._choose_best_move(battle)
        if best_move_order:
            return best_move_order
            
        return self.choose_random_move(battle)

    def _should_switch(self, battle: AbstractBattle) -> bool:
        """
        Radical Red Logic:
        - Switch if immune to predicted move (Hard to predict perfectly here without engine access)
        - Switch if slower and gonna die (OHKO)
        """
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        if not opponent or active.fainted:
            return True # Must switch if fainted
            
        # Estimating Speed
        my_speed = active.base_stats["spe"]
        op_speed = opponent.base_stats["spe"]
        
        # If we are slower and likely to be OHKO'd
        if my_speed < op_speed:
            # Check for 4x weakness or strong STAB
            # Very rough estimate
            weakness = active.damage_multiplier(opponent.type_1)
            if opponent.type_2:
                weakness = max(weakness, active.damage_multiplier(opponent.type_2))
                
            if weakness >= 4.0:
                return True # Almost certainly dead
            if weakness >= 2.0 and active.current_hp_fraction < 0.6:
                return True # Probably dead
                
        return False

    def _choose_best_switch(self, battle: AbstractBattle) -> BattleOrder:
        available = [m for m in battle.available_switches]
        if not available: return None
        
        opponent = battle.opponent_active_pokemon
        if not opponent: 
            # Opponent fainted? Just pick highest level/BST
            return SingleBattleOrder(max(available, key=lambda m: m.base_stats["bst"]))
        
        best_mon = None
        best_score = -9999
        
        for mon in available:
            score = 0
            
            # 1. Resistance Score (Defense)
            # Avoid weaknesses
            def_mult = mon.damage_multiplier(opponent.type_1)
            if opponent.type_2:
                def_mult *= mon.damage_multiplier(opponent.type_2)
            
            # Lower multiplier is better. 
            # 0.25 -> +40, 0.5 -> +20, 1.0 -> 0, 2.0 -> -20, 4.0 -> -40
            if def_mult <= 0.25: score += 40
            elif def_mult <= 0.5: score += 20
            elif def_mult >= 4.0: score -= 40
            elif def_mult >= 2.0: score -= 20
            
            # 2. Offense Score (We can hit them)
            # Do we have a super effective STAB move?
            off_mult1 = opponent.damage_multiplier(mon.type_1)
            off_mult2 = opponent.damage_multiplier(mon.type_2) if mon.type_2 else 0
            
            if off_mult1 >= 2.0 or off_mult2 >= 2.0:
                score += 15
                
            # 3. Speed Score
            if mon.base_stats["spe"] > opponent.base_stats["spe"]:
                score += 10
            
            if score > best_score:
                best_score = score
                best_mon = mon
                
        if best_mon:
            return SingleBattleOrder(best_mon)
        return None

    def _choose_best_move(self, battle: AbstractBattle) -> BattleOrder:
        available_moves = battle.available_moves
        if not available_moves: return None
        
        opponent = battle.opponent_active_pokemon
        active = battle.active_pokemon
        
        # Scoring System
        # Key: Move -> Score
        scores = {move: 0 for move in available_moves}
        
        for move in available_moves:
            # --- DAMAGE CALCULATION ---
            if move.category != MoveCategory.STATUS:
                damage = self._estimate_damage(move, active, opponent)
                
                # Base Score from Damage %
                # If dmg is 50% of opp HP, score += 50
                if opponent.max_hp:
                     # Estimate hp fraction
                     # damage is absolute? no estimate is usually absolute but we assume standard 100 level stats roughly
                     # normalize by opponent HP estimate
                     # Let's use % damage estimate
                     percent_dmg = damage
                     # Cap at 100
                     if percent_dmg > 100: percent_dmg = 100
                     scores[move] += percent_dmg
                else:
                     scores[move] += damage / 2.0 # Fallback
                
                # KO Bonus (CFRU: +4 points equivalent)
                # If predicted damage >= opponent current HP
                # We need opponent current HP %
                if percent_dmg >= (opponent.current_hp_fraction * 100):
                     scores[move] += 40 # Huge bonus for KO
                     
                # Priority Bonus if KO is close
                if move.priority > 0 and (opponent.current_hp_fraction * 100) < 30:
                     scores[move] += 15
                     
            else:
                # --- STATUS / SETUP ---
                # Sleep
                if move.status == "sleep" or move.status == "freeze":
                    if not opponent.status:
                         scores[move] += 25
                         
                # Burn (if Physical attacker)
                elif move.status == "burn":
                    if not opponent.status:
                         # Check if physical
                         if opponent.base_stats["atk"] > opponent.base_stats["spa"]:
                             scores[move] += 20
                         else:
                             scores[move] += 5
                             
                # Setup (Swords Dance etc)
                if move.boosts:
                     # Only setup if healthy
                     if active.current_hp_fraction > 0.8:
                         scores[move] += 15
                     elif active.current_hp_fraction < 0.4:
                         scores[move] -= 10 # Don't setup if dying
                         
                # Hazards
                if move.side_condition:
                     # Setup rocks if early game
                     scores[move] += 10

        # Pick best move with some randomness if scores are close?
        # CFRU is deterministic usually unless scores equal.
        # Pick max.
        if not scores: return None
        
        best_move = max(scores, key=scores.get)
        return SingleBattleOrder(best_move)

    def _estimate_damage(self, move: Move, attacker: Pokemon, defender: Pokemon) -> float:
        """
        Returns estimated % HP damage (0-100+).
        """
        if move.category == MoveCategory.STATUS: return 0
        
        # 1. Stats
        if move.category == MoveCategory.PHYSICAL:
            a = attacker.stats["atk"]
            d = defender.base_stats["def"] * 2 + 31 + 63 # Rough estimate of EV/IV/Level
            # Wait, defender stats are hidden usually in current poke-env?
            # base_stats are available. stats (current) might be estimated.
            # Using base stats is safer proxy if exact unknown.
            # Let's use simple base stat ratio adjusted for level 50 vs 100?
            # Standard formula: ((2 * L / 5 + 2) * Power * A / D) / 50 + 2
            a = attacker.base_stats["atk"] # use base as proxy
            d = defender.base_stats["def"]
        else:
            a = attacker.base_stats["spa"]
            d = defender.base_stats["spd"]
            
        # 2. Power
        power = move.base_power
        
        # 3. STAB
        stab = 1.5 if move.type in attacker.types else 1.0
        
        # 4. Effectiveness
        eff = defender.damage_multiplier(move.type)
        
        # 5. Rough Damage Formula (simplified level 100)
        # Damage ~= ( (2*100/5 + 2) * Power * A/D ) / 50 + 2
        # ~= (42 * Power * A/D) / 50
        # ~= 0.84 * Power * A/D
        
        base_dmg = 0.84 * power * (a / d)
        final_dmg = base_dmg * stab * eff
        
        # Convert to % of HP?
        # HP ~= Base * 2 + 110 + 63 (at level 100) ~= Base * 2 + 200
        hp_est = defender.base_stats["hp"] * 2 + 200
        
        percent = (final_dmg / hp_est) * 100
        return percent
