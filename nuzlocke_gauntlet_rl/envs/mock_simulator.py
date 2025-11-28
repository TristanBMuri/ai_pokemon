import random
from typing import List, Tuple
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec
from nuzlocke_gauntlet_rl.envs.battle_simulator import BattleSimulator

class MockBattleSimulator(BattleSimulator):
    """
    A simple mock simulator that decides the winner based on level and team size.
    Used for training the run-level policy without the overhead of Showdown.
    """
    
    def simulate_battle(self, my_team: List[PokemonSpec], enemy_team: List[PokemonSpec]) -> Tuple[bool, List[bool]]:
        if not my_team:
            return False, []
            
        # Simple heuristic: Sum of levels
        my_power = sum(p.level for p in my_team)
        enemy_power = sum(p.level for p in enemy_team)
        
        # Add some randomness
        # If my_power > enemy_power, high chance to win
        win_prob = my_power / (my_power + enemy_power) if (my_power + enemy_power) > 0 else 0
        
        win = random.random() < win_prob
        
        survivors = []
        if win:
            # If we win, most survive, maybe 1 dies if close
            for _ in my_team:
                # 90% survival rate per mon if win
                survivors.append(random.random() < 0.9)
        else:
            # If we lose, everyone dies (or maybe 1 survives to retreat, but Nuzlocke usually means wipe)
            survivors = [False] * len(my_team)
            
        return win, survivors
