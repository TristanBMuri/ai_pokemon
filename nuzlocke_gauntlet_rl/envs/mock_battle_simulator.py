
from typing import List, Tuple
import numpy as np
from nuzlocke_gauntlet_rl.envs.battle_simulator import BattleSimulator
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec

class MockBattleSimulator(BattleSimulator):
    """
    A mock simulator that determines battle outcome based on simple heuristics.
    Used for testing the Manager agent without Showdown overhead.
    """
    def simulate_battle(self, my_team: List[PokemonSpec], enemy_team: List[PokemonSpec], risk_token: int = 0) -> Tuple[bool, List[bool]]:
        # Simple heuristic:
        # Calculate total power of my team vs enemy team
        # Power = Level * 10
        
        my_power = sum([p.level * 10 for p in my_team])
        enemy_power = sum([p.level * 10 for p in enemy_team])
        
        # Risk token influence:
        # 0 (Safe): Lower variance, slightly lower win rate if underpowered
        # 1 (Neutral): Standard
        # 2 (Desperate): High variance, chance to win even if underpowered
        
        win_prob = 0.5
        if my_power > enemy_power:
            win_prob = 0.8
        elif my_power < enemy_power:
            win_prob = 0.2
            
        if risk_token == 0: # Safe
            # If stronger, win more often. If weaker, lose more often.
            if win_prob > 0.5: win_prob += 0.1
            else: win_prob -= 0.1
        elif risk_token == 2: # Desperate
            # Normalize towards 0.5 (chaos)
            win_prob = (win_prob + 0.5) / 2
            
        win = np.random.random() < win_prob
        
        # Survivors
        # If win, more survivors.
        survivors = []
        for _ in my_team:
            if win:
                survivors.append(np.random.random() > 0.2) # 80% survival chance
            else:
                survivors.append(np.random.random() > 0.8) # 20% survival chance
                
        return win, survivors
