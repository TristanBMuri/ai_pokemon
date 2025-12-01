import random
from typing import List, Dict, Optional
from poke_env.data import GenData
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec

class MovesetGenerator:
    def __init__(self, gen: int = 9):
        self.gen_data = GenData.from_gen(gen)
        self.learnset = self.gen_data.learnset
        self.pokedex = self.gen_data.pokedex
        self.moves = self.gen_data.moves
        
    def get_learnable_moves(self, species: str) -> List[str]:
        """Returns a list of all learnable moves for a species."""
        species_id = species.lower().replace(" ", "").replace("-", "").replace(".", "")
        # Handle special cases if needed (e.g. forms)
        # poke-env usually handles forms by name (e.g. charizardmega)
        
        if species_id not in self.learnset:
            # Try to find base species if not found?
            # For now, just return empty list or log warning
            print(f"WARNING: No learnset found for {species_id}")
            return []
            
        data = self.learnset[species_id]
        if "learnset" in data:
            return list(data["learnset"].keys())
        return list(data.keys()) # Fallback if structure is different

    def generate_builds(self, spec: PokemonSpec, n_builds: int = 3) -> List[List[str]]:
        """
        Generates n_builds movesets for the given PokemonSpec.
        Returns a list of lists of move names.
        """
        learnable_moves = self.get_learnable_moves(spec.species)
        if not learnable_moves:
            return [[] for _ in range(n_builds)]
            
        # Filter valid moves (exist in moves data)
        valid_moves = [m for m in learnable_moves if m in self.moves]
        
        # Get Pokemon types
        species_id = spec.species.lower().replace(" ", "").replace("-", "").replace(".", "")
        types = self.pokedex.get(species_id, {}).get("types", [])
        
        builds = []
        
        # Build 0: Aggressive (High Power STAB + Coverage)
        builds.append(self._generate_aggressive_build(valid_moves, types))
        
        # Build 1: Balanced (STAB + Status/Utility)
        if n_builds > 1:
            builds.append(self._generate_balanced_build(valid_moves, types))
            
        # Build 2+: Random
        for _ in range(n_builds - len(builds)):
            builds.append(self._generate_random_build(valid_moves))
            
        return builds

    def _generate_aggressive_build(self, moves: List[str], types: List[str]) -> List[str]:
        """Prioritizes high power STAB moves and coverage."""
        stab_moves = []
        coverage_moves = []
        
        for m in moves:
            move_data = self.moves[m]
            if move_data["category"] == "Status":
                continue
                
            power = move_data.get("basePower", 0)
            if power < 40: continue # Skip weak moves
            
            if move_data["type"] in types:
                stab_moves.append((m, power))
            else:
                coverage_moves.append((m, power))
                
        # Sort by power desc
        stab_moves.sort(key=lambda x: x[1], reverse=True)
        coverage_moves.sort(key=lambda x: x[1], reverse=True)
        
        build = []
        # Take top 2 STAB
        for m, _ in stab_moves[:2]:
            build.append(m)
            
        # Fill rest with Coverage
        for m, _ in coverage_moves:
            if len(build) >= 4: break
            if m not in build:
                build.append(m)
                
        # If still not full, fill with more STAB
        for m, _ in stab_moves:
            if len(build) >= 4: break
            if m not in build:
                build.append(m)
                
        return build

    def _generate_balanced_build(self, moves: List[str], types: List[str]) -> List[str]:
        """Mix of STAB and Status moves."""
        stab_moves = []
        status_moves = []
        
        for m in moves:
            move_data = self.moves[m]
            if move_data["category"] == "Status":
                status_moves.append(m)
            else:
                power = move_data.get("basePower", 0)
                if power < 40: continue
                if move_data["type"] in types:
                    stab_moves.append((m, power))
                    
        stab_moves.sort(key=lambda x: x[1], reverse=True)
        # Randomize status moves to get variety
        random.shuffle(status_moves)
        
        build = []
        # 2 STAB
        for m, _ in stab_moves[:2]:
            build.append(m)
            
        # 2 Status
        for m in status_moves[:2]:
            build.append(m)
            
        # Fill if needed
        remaining = [m for m in moves if m not in build]
        random.shuffle(remaining)
        while len(build) < 4 and remaining:
            build.append(remaining.pop())
            
        return build

    def _generate_random_build(self, moves: List[str]) -> List[str]:
        """Random valid moves."""
        if len(moves) <= 4:
            return moves
        return random.sample(moves, 4)
