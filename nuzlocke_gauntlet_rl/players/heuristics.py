import random
from typing import List, Optional, Union
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.side_condition import SideCondition
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder, DoubleBattleOrder
from poke_env.battle.move_category import MoveCategory
from poke_env.player.player import Player # Only for static helper methods

class BaseHeuristic:
    """Base class for heuristic logic without Player overhead."""
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        raise NotImplementedError

    def choose_random_move(self, battle: AbstractBattle) -> BattleOrder:
        return Player.choose_random_move(battle)
        
    def create_order(self, order: BattleOrder, **kwargs) -> BattleOrder:
        if isinstance(order, BattleOrder):
             return order 
        return order


class SimpleHeuristicsLogic(BaseHeuristic):
    """
    Lightweight version of Poke-Env's SimpleHeuristicsPlayer.
    """
    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealthrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }
    ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    SWITCH_OUT_MATCHUP_THRESHOLD = -2

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if isinstance(battle, DoubleBattle):
             return self.choose_random_move(battle)

        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon

        if active is None or opponent is None:
            return self.choose_random_move(battle)

        # 1. Check for switch out
        if self._should_switch_out(battle):
             best_switch = max(battle.available_switches, key=lambda s: self._estimate_matchup(s, opponent)) if battle.available_switches else None
             if best_switch:
                 return SingleBattleOrder(best_switch)

        # 2. Choose Best Move
        available_moves = battle.available_moves
        if not available_moves:
             return self.choose_random_move(battle)
             
        best_move = max(
            available_moves,
            key=lambda m: m.base_power 
            * (1.5 if m.type in active.types else 1) 
            * opponent.damage_multiplier(m.type)
            * m.accuracy
        )
        
        return SingleBattleOrder(best_move)

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max([mon.damage_multiplier(t) for t in opponent.types if t is not None])
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT
        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        return score

    def _should_switch_out(self, battle: AbstractBattle):
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not battle.available_switches: return False
        
        if [m for m in battle.available_switches if self._estimate_matchup(m, opponent) > 0]:
             if self._estimate_matchup(active, opponent) < self.SWITCH_OUT_MATCHUP_THRESHOLD:
                 return True
             if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3: return True
             if active.boosts["atk"] <= -3 and active.base_stats["atk"] >= active.base_stats["spa"]: return True
             if active.boosts["spa"] <= -3 and active.base_stats["spa"] >= active.base_stats["atk"]: return True
             
        return False
        
    def _stat_estimation(self, mon, stat):
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost


class RadicalRedLogic(BaseHeuristic):
    """
    Logic extracted from RadicalRedPlayer.
    """
    def __init__(self):
        self.anti_abuse_counter = 0
        self.last_opponent_mon = None
        
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # Update Anti-Abuse Counter
        if battle.opponent_active_pokemon:
            if self.last_opponent_mon and self.last_opponent_mon != battle.opponent_active_pokemon:
                self.anti_abuse_counter += 3
            self.last_opponent_mon = battle.opponent_active_pokemon
            
        self.anti_abuse_counter = max(0, self.anti_abuse_counter - 1)
        
        # 1. Switch Logic
        best_switch = self._choose_best_switch(battle)
        if best_switch and self._should_switch(battle):
             return best_switch

        # 2. Check for Revenge Kill
        if battle.active_pokemon.fainted:
             return best_switch if best_switch else self.choose_random_move(battle)

        # 3. Move Scoring
        best_move_order = self._choose_best_move(battle)
        if best_move_order:
            return best_move_order
            
        return self.choose_random_move(battle)

    def _should_switch(self, battle: AbstractBattle) -> bool:
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        if not opponent or active.fainted: return True
        
        my_speed = active.base_stats["spe"]
        op_speed = opponent.base_stats["spe"]
        
        if my_speed < op_speed:
            weakness = active.damage_multiplier(opponent.type_1)
            if opponent.type_2:
                weakness = max(weakness, active.damage_multiplier(opponent.type_2))
            if weakness >= 4.0: return True
            if weakness >= 2.0 and active.current_hp_fraction < 0.6: return True
        return False

    def _choose_best_switch(self, battle: AbstractBattle):
        available = [m for m in battle.available_switches]
        if not available: return None
        opponent = battle.opponent_active_pokemon
        if not opponent: 
            return SingleBattleOrder(max(available, key=lambda m: m.base_stats["bst"]))
            
        best_mon = None
        best_score = -9999
        for mon in available:
            score = 0
            # Resistance
            def_mult = mon.damage_multiplier(opponent.type_1)
            if opponent.type_2: def_mult *= mon.damage_multiplier(opponent.type_2)
            if def_mult <= 0.25: score += 40
            elif def_mult <= 0.5: score += 20
            elif def_mult >= 4.0: score -= 40
            elif def_mult >= 2.0: score -= 20
            
            # Offense
            off_mult1 = opponent.damage_multiplier(mon.type_1)
            off_mult2 = opponent.damage_multiplier(mon.type_2) if mon.type_2 else 0
            if off_mult1 >= 2.0 or off_mult2 >= 2.0: score += 15
            
            if mon.base_stats["spe"] > opponent.base_stats["spe"]: score += 10
            
            if score > best_score:
                best_score = score
                best_mon = mon
        if best_mon: return SingleBattleOrder(best_mon)
        return None

    def _choose_best_move(self, battle: AbstractBattle):
        available_moves = battle.available_moves
        if not available_moves: return None
        opponent = battle.opponent_active_pokemon
        active = battle.active_pokemon
        
        scores = {move: 0 for move in available_moves}
        for move in available_moves:
            if move.category != MoveCategory.STATUS:
                damage = self._estimate_damage(move, active, opponent)
                scores[move] += min(damage, 100)
                if damage >= (opponent.current_hp_fraction * 100): scores[move] += 40
                if move.priority > 0 and (opponent.current_hp_fraction * 100) < 30: scores[move] += 15
            else:
                if move.status in ["sleep", "freeze"] and not opponent.status: scores[move] += 25
                elif move.status == "burn" and not opponent.status:
                     scores[move] += 20 if opponent.base_stats["atk"] > opponent.base_stats["spa"] else 5
                if move.boosts:
                     if active.current_hp_fraction > 0.8: scores[move] += 15
                     elif active.current_hp_fraction < 0.4: scores[move] -= 10
                if move.side_condition: scores[move] += 10
        if not scores: return None
        return SingleBattleOrder(max(scores, key=scores.get))

    def _estimate_damage(self, move, attacker, defender):
        if move.category == MoveCategory.STATUS: return 0
        if move.category == MoveCategory.PHYSICAL:
            a = attacker.base_stats["atk"]
            d = defender.base_stats["def"]
        else:
            a = attacker.base_stats["spa"]
            d = defender.base_stats["spd"]
        power = move.base_power
        stab = 1.5 if move.type in attacker.types else 1.0
        eff = defender.damage_multiplier(move.type)
        idx = max(1, d)
        base_dmg = 0.84 * power * (a / idx)
        final_dmg = base_dmg * stab * eff
        hp_est = defender.base_stats["hp"] * 2 + 200
        return (final_dmg / hp_est) * 100
