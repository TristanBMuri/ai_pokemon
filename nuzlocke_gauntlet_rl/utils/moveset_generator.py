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
        
        # Global Move ID Mapping
        # Sort keys for determinism
        self.all_moves_list = sorted(list(self.moves.keys()))
        self.move_to_id = {m: i+1 for i, m in enumerate(self.all_moves_list)} # 0 is RESERVED/EMPTY
        self.id_to_move_map = {i+1: m for i, m in enumerate(self.all_moves_list)}
        self.max_move_id = len(self.all_moves_list)
        
    def _to_id(self, text: str) -> str:
        """Converts text to Showdown ID format (lowercase, alphanumeric only)."""
        return "".join(c for c in text.lower() if c.isalnum())
        
    def get_move_id(self, move_name: str) -> int:
        clean_name = self._to_id(move_name)
        return self.move_to_id.get(clean_name, 0)
        
    def get_move_name(self, move_id: int) -> str:
        return self.id_to_move_map.get(move_id, None)

    def get_learnable_moves_ids(self, species: str) -> List[int]:
        names = self.get_learnable_moves(species)
        return [self.get_move_id(n) for n in names if n in self.move_to_id]

    def get_learnable_moves_ids_at_level(self, species: str, level: int) -> List[int]:
        names = self.get_learnable_moves_at_level(species, level)
        return [self.get_move_id(n) for n in names if n in self.move_to_id]

    def get_types(self, species: str) -> List[str]:
        """Returns the types of a species."""
        species_id = self._to_id(species)
        if species_id in self.pokedex:
            return self.pokedex[species_id].get("types", [])
        return []

    def get_base_stats(self, species: str) -> List[int]:
        """Returns [HP, Atk, Def, SpA, SpD, Spe]."""
        species_id = self._to_id(species)
        if species_id in self.pokedex:
            stats = self.pokedex[species_id].get("baseStats", {})
            return [
                stats.get("hp", 0),
                stats.get("atk", 0),
                stats.get("def", 0),
                stats.get("spa", 0),
                stats.get("spd", 0),
                stats.get("spe", 0)
            ]
        return [0, 0, 0, 0, 0, 0]

    def encode_ability(self, ability: str) -> int:
        """Encodes ability name to int ID (hash mod 1000)."""
        if not ability: return 0
        return abs(hash(ability.lower())) % 1000

    def encode_move(self, move: str) -> int:
        """Encodes move name to int ID (hash mod 1000)."""
        if not move: return 0
        return abs(hash(move.lower())) % 1000


    def get_ability(self, species: str) -> str:
        """Returns the first ability of a species."""
        species_id = self._to_id(species)
        if species_id in self.pokedex:
            abilities = self.pokedex[species_id].get("abilities", {})
            # Return first available ability (0 or H or S)
            for k in ["0", "1", "H", "S"]:
                if k in abilities:
                    return abilities[k]
            # Fallback: return any value
            if abilities:
                return list(abilities.values())[0]
        return "No Ability"

    def get_learnable_moves(self, species: str) -> List[str]:
        """Returns a list of all learnable moves for a species."""
        species_id = self._to_id(species)
        
        # Try exact match first
        if species_id in self.learnset:
            data = self.learnset[species_id]
            if "learnset" in data:
                return list(data["learnset"].keys())
            return list(data.keys())

        # Fallback logic for forms
        # Common suffixes to strip: -mega, -megax, -megay, -primal, -alola, -galar, -hisui, -paldea
        # Also Ogerpon masks: -wellspring, -hearthflame, -cornerstone
        # And others like -therian, -incarnate, etc.
        
        # Simple heuristic: try to find the base name by splitting on '-'
        # But species_id has removed hyphens.
        # So we work with the input species string.
        
        base_species = species.split("-")[0]
        base_id = self._to_id(base_species)
        
        if base_id in self.learnset:
            # print(f"DEBUG: Using base species {base_species} for {species}")
            data = self.learnset[base_id]
            if "learnset" in data:
                return list(data["learnset"].keys())
            return list(data.keys())
            
        print(f"WARNING: No learnset found for {species} (ID: {species_id}, Base: {base_id})")
        return []

    def get_learnable_moves_at_level(self, species: str, level: int, gen: int = 9) -> List[str]:
        """Returns moves learnable by level up up to specific level."""
        species_id = self._to_id(species)
        learnset_data = None
        
        # 1. Try exact
        if species_id in self.learnset:
            learnset_data = self.learnset[species_id]
        else:
            # 2. Try base
            base_species = species.split("-")[0]
            base_id = self._to_id(base_species)
            if base_id in self.learnset:
                learnset_data = self.learnset[base_id]
        
        if not learnset_data: return []
        
        # Extract 'learnset' dict
        data = learnset_data.get("learnset", learnset_data)
        
        valid_moves = []
        prefix = f"{gen}L"
        
        for move_id, sources in data.items():
            for src in sources:
                # Check for Level Up in current Gen (e.g. 9L15)
                # Also support older gens? No, Radical Red is Gen 9 based.
                if src.startswith(prefix):
                    try:
                        req_lvl = int(src[len(prefix):])
                        if req_lvl <= level:
                            valid_moves.append(move_id)
                            break # Found valid source
                    except: pass
                
                # Check for Level 1 moves from older gens if current gen missing?
                # Or "Start" moves. Poke-env data usually uses "9L1" for start moves.
                # But sometimes just "8L1" if unchanged?
                # Let's check for ANY "L1" or "L0" from Gen 9, 8, 7.
                if (f"{gen}L1" in src) or (f"{gen-1}L1" in src) or (f"{gen-2}L1" in src):
                     valid_moves.append(move_id)
                     break
                     
        return valid_moves

    def generate_builds(self, spec: PokemonSpec, n_builds: int = 3) -> List[List[str]]:
        """
        Generates n_builds movesets for the given PokemonSpec.
        Returns a list of lists of move names.
        """
        learnable_moves = self.get_learnable_moves(spec.species)
        if not learnable_moves:
            return [[] for _ in range(n_builds)]
            
        # Filter valid moves (exist in moves data)
        # Also apply global blacklist here
        bad_moves = {
            "ceaseedge", "ceaselessedge", "hyperbeam", "gigaimpact", "blastburn", 
            "frenzyplant", "hydrocannon", "roaroftime", "eternabeam", "meteorassault", 
            "prismaticlaser", "focuspunch", "lastresort", "belch", "synchronoise", 
            "dreameater", "explosion", "selfdestruct", "finalgambit", "memento"
        }
        
        valid_moves = [
            m for m in learnable_moves 
            if m in self.moves and m.lower().replace(" ", "").replace("-", "") not in bad_moves
        ]
        
        # Get Pokemon types
        species_id = self._to_id(spec.species)
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
        """Prioritizes high power STAB moves and coverage, avoiding recharge/bad moves."""
        stab_moves = []
        coverage_moves = []
        
        # Moves to avoid for simple agents
        bad_moves = {
            "hyperbeam", "gigaimpact", "blastburn", "frenzyplant", "hydrocannon", 
            "roaroftime", "eternabeam", "meteorassault", "prismaticlaser",
            "focuspunch", "lastresort", "belch", "synchronoise", "dreameater",
            "explosion", "selfdestruct", "finalgambit", "memento",
            "ceaseedge", "ceaselessedge"
        }
        
        for m in moves:
            move_id = m.lower().replace(" ", "").replace("-", "")
            if move_id in bad_moves:
                continue
                
            move_data = self.moves[m]
            if move_data["category"] == "Status":
                continue
                
            power = move_data.get("basePower", 0)
            acc = move_data.get("accuracy", 100)
            if acc is True: acc = 100
            
            # Skip weak moves or very inaccurate ones
            if power < 60: continue 
            if acc < 70: continue
            
            # Score = Power * (Acc/100)
            score = power * (acc / 100.0)
            
            if move_data["type"] in types:
                stab_moves.append((m, score))
            else:
                coverage_moves.append((m, score))
                
        # Sort by score desc
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
        
        bad_moves = {
            "hyperbeam", "gigaimpact", "blastburn", "frenzyplant", "hydrocannon", 
            "roaroftime", "eternabeam", "meteorassault", "prismaticlaser",
            "focuspunch", "lastresort", "belch", "synchronoise", "dreameater",
            "explosion", "selfdestruct", "finalgambit", "memento",
            "ceaseedge", "ceaselessedge"
        }
        
        for m in moves:
            move_id = m.lower().replace(" ", "").replace("-", "")
            if move_id in bad_moves:
                continue
                
            move_data = self.moves[m]
            if move_data["category"] == "Status":
                # Prioritize good status moves
                if move_id in ["protect", "thunderwave", "willowisp", "toxic", "roost", "recover", "softboiled", "synthesis", "moonlight", "morningsun", "slackoff", "milkdrink", "shoreup", "strengthsap", "swordsdance", "nastyplot", "calmmind", "bulkup", "dragondance", "shellsmash", "quiverdance", "stealthrock", "spikes", "defog", "rapidspin"]:
                    status_moves.append((m, 100))
                else:
                    status_moves.append((m, 50))
            else:
                power = move_data.get("basePower", 0)
                acc = move_data.get("accuracy", 100)
                if acc is True: acc = 100
                
                if power < 60: continue
                if acc < 70: continue
                
                score = power * (acc / 100.0)
                
                if move_data["type"] in types:
                    stab_moves.append((m, score))
                    
        stab_moves.sort(key=lambda x: x[1], reverse=True)
        status_moves.sort(key=lambda x: x[1], reverse=True)
        
        build = []
        # 2 STAB
        for m, _ in stab_moves[:2]:
            build.append(m)
            
        # 2 Status
        for m, _ in status_moves[:2]:
            build.append(m)
            
        # Fill if needed
        remaining = [m for m in moves if m not in build and m.lower().replace(" ", "").replace("-", "") not in bad_moves]
        random.shuffle(remaining)
        while len(build) < 4 and remaining:
            build.append(remaining.pop())
            
        return build

    def _generate_random_build(self, moves: List[str]) -> List[str]:
        """Random valid moves."""
        if len(moves) <= 4:
            return moves
        return random.sample(moves, 4)
