from abc import ABC, abstractmethod
from typing import List, Tuple
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec

class BattleSimulator(ABC):
    """Abstract interface for running battles."""
    
    @abstractmethod
    def simulate_battle(self, my_team: List[PokemonSpec], enemy_team: List[PokemonSpec], risk_token: int = 0) -> Tuple[bool, List[bool], dict]:
        """
        Simulates a battle.
        
        Args:
            my_team: List of user's Pokemon specs.
            enemy_team: List of opponent's Pokemon specs.
            
        Returns:
            win: True if user won.
            survivors: List of booleans matching my_team, True if that mon survived.
            metrics: Dictionary of battle metrics (turns, opponent_fainted, etc.)
        """
        pass
