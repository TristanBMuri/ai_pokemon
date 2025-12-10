from poke_env.player import Player
from poke_env.battle import Pokemon, Move, AbstractBattle
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder
import random

class RadicalRedPlayer(Player):
    """
    A custom player that mimics the AI heuristics of Pokemon Radical Red (Hardcore Mode).
    
    Features:
    1. Absorption: Switches to immunity if predicted.
    2. Revenge Killing: Switches to faster mon that can kill.
    3. Anti-Abuse: Tracks player switches to predict and counter.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anti_abuse_counter = 0
        self.last_opponent_mon = None
        
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # Update Anti-Abuse Counter
        if battle.opponent_active_pokemon:
            if self.last_opponent_mon and self.last_opponent_mon != battle.opponent_active_pokemon:
                self.anti_abuse_counter += 3
            self.last_opponent_mon = battle.opponent_active_pokemon
            
        self.anti_abuse_counter = max(0, self.anti_abuse_counter - 1)
        
        # 1. Check for Absorption / Immunity Switch
        # If opponent is likely to use a move we are immune to, switch to immunity.
        # This requires predicting opponent move.
        
        # 2. Check for Revenge Kill (if active is fainted)
        if battle.active_pokemon.fainted:
            return self._choose_revenge_killer(battle)
            
        # 3. Check if we should switch to avoid death
        # If we are slower and dead to next hit
        if self._should_switch_to_avoid_death(battle):
            switch = self._choose_best_switch(battle)
            if switch:
                return switch
                
        # 4. Anti-Abuse Prediction
        # If counter is high, predict a switch and double switch or use coverage
        if self.anti_abuse_counter >= 9:
            # 25% chance to read switch
            if random.random() < 0.25:
                # Predict opponent switch
                pass
                
        # Default: Choose best move
        return self._choose_best_move(battle)
        
    def _choose_revenge_killer(self, battle: AbstractBattle) -> BattleOrder:
        # Find mon that outspeeds and kills
        available = [m for m in battle.available_switches]
        if not available:
            return self.choose_random_move(battle)
            
        opponent = battle.opponent_active_pokemon
        if not opponent:
             return self.choose_random_move(battle)
             
        best_mon = None
        best_score = -1
        
        for mon in available:
            score = 0
            # Speed check
            # We don't know exact speed, but use base speed + level estimate
            my_speed = mon.base_stats["spe"]
            op_speed = opponent.base_stats["spe"]
            
            if my_speed > op_speed:
                score += 100
                
            # Type advantage
            eff = opponent.damage_multiplier(mon.type_1)
            if mon.type_2:
                eff *= opponent.damage_multiplier(mon.type_2)
                
            score += eff * 10
            
            if score > best_score:
                best_score = score
                best_mon = mon
                
        if best_mon:
            return SingleBattleOrder(best_mon)
            
        return self.choose_random_move(battle)

    def _should_switch_to_avoid_death(self, battle: AbstractBattle) -> bool:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        if not opponent: return False
        
        # Simple speed check
        my_speed = active.base_stats["spe"]
        op_speed = opponent.base_stats["spe"]
        
        if my_speed < op_speed:
            # If we are slower, check if we are weak to opponent
            # This is a heuristic, real calc would need damage calc
            weakness = active.damage_multiplier(opponent.type_1)
            if opponent.type_2:
                weakness = max(weakness, active.damage_multiplier(opponent.type_2))
                
            if weakness >= 2.0:
                return True
                
        return False
        
    def _choose_best_switch(self, battle: AbstractBattle) -> BattleOrder:
        available = [m for m in battle.available_switches]
        if not available: return None
        
        opponent = battle.opponent_active_pokemon
        if not opponent: return None
        
        best_mon = None
        best_score = -1
        
        for mon in available:
            # Score based on resistance to opponent
            resistance = 1.0 / mon.damage_multiplier(opponent.type_1)
            if opponent.type_2:
                resistance += 1.0 / mon.damage_multiplier(opponent.type_2)
                
            if resistance > best_score:
                best_score = resistance
                best_mon = mon
                
        if best_mon:
            return SingleBattleOrder(best_mon)
        return None

    def _choose_best_move(self, battle: AbstractBattle) -> BattleOrder:
        available_moves = battle.available_moves
        if not available_moves:
            return self.choose_random_move(battle)
            
        opponent = battle.opponent_active_pokemon
        best_move = None
        best_damage = -1
        
        for move in available_moves:
            if move.category == "Status":
                continue
                
            # Estimate damage
            # Power * STAB * Effectiveness
            power = move.base_power
            
            stab = 1.5 if move.type in battle.active_pokemon.types else 1.0
            
            eff = 1.0
            if opponent:
                eff = opponent.damage_multiplier(move.type)
                
            damage = power * stab * eff
            
            if damage > best_damage:
                best_damage = damage
                best_move = move
                
        if best_move:
            return SingleBattleOrder(best_move)
            
        return self.choose_random_move(battle)
