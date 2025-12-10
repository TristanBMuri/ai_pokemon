import random
from typing import List, Dict, Optional, Set
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec, MonInstance
import uuid

# Map Trainer Index (0-based from Complete Gauntlet) to Unlocked Routes
# 0: Rival Lab
# 1: Rival Route 22
# 2: Brendan Viridian Forest
# 0: Rival Lab
# 1: Rival Route 22
# 2: Brendan Viridian Forest
trainer_unlocks = {
    # 0: Early Game (Rival Lab) -> Open World Start
    0: ["ROUTE 1", "ROUTE 22", "ROUTE 2", "VIRIDIAN FOREST"],
    
    # 4: Falkner (Pewter) -> Pewter Area
    4: ["PEWTER CITY"],
    
    # 5: Brock (Gym 1) -> Path to Mt Moon
    5: ["ROUTE 3", "MT MOON 1F"],
    
    # 7: Archer (Mt Moon) -> Exit Moon & Route 4
    7: ["MT MOON B1F", "MT MOON B2F", "ROUTE 4"],
    
    # 15: Bugsy (Route 25) -> Nugget Bridge Area
    15: ["ROUTE 24", "ROUTE 25"],
    
    # 16: Misty (Gym 2) -> Path South
    16: ["ROUTE 5", "ROUTE 6", "DIGLETT CAVE", "DIGLETT CAVE B1F"],
    
    # 20: Lt. Surge (Gym 3) -> East Kanto
    20: ["ROUTE 9", "ROUTE 10", "ROCK TUNNEL 1F", "ROCK TUNNEL B1F", "ROUTE 11"],
    
    # 23: Erika (Gym 4) -> Cycling Road
    23: ["ROUTE 16", "ROUTE 17", "ROUTE 18"],
    
    # 27: Giovanni (Rocket Hideout) -> Pokemon Tower & Lavender
    27: ["PKMN TOWER 3&5F", "PKMN TOWER 4F", "PKMN TOWER 6F", "PKMN TOWER 7F"],
    
    # 30: Sabrina (Gym 6) -> Silence Bridge Path
    30: ["ROUTE 12", "ROUTE 13", "ROUTE 14", "ROUTE 15"],
    
    # 48: Koga (Gym 5 in RR?) -> Sea Routes
    48: ["SEAFOAM 1F", "SEAFOAM B1F", "SEAFOAM B2F", "SEAFOAM B3F", "SEAFOAM B4F", "ROUTE 21A"],
    
    # 50: Blaine (Gym 7) -> Cinnabar Mansion & Power Plant (Backtrack)
    50: ["MANSION 1F", "MANSION 2F", "MANSION 3F", "MANSION B1F", "POWER PLANT"],
    
    # 52: Clair (Gym 8) -> Victory Road
    52: ["ROUTE 23", "VICTORY ROAD 1F", "VICTORY ROAD 2F", "VICTORY ROAD 3F", "CERULEAN CAVE 1F"],
}

class NuzlockeMechanics:
    def __init__(self, encounters_map: Dict[str, List[Dict]], moveset_gen):
        self.encounters = encounters_map
        self.moveset_gen = moveset_gen
        
    def get_starter(self) -> MonInstance:
        """Legacy random starter."""
        return self.get_starter_choice(random.randint(0, 2))

    def get_starter_choice(self, choice_idx: int) -> MonInstance:
        """
        Returns starter based on index (0-26).
        Gen 1: 0-2 (Grass, Fire, Water)
        ...
        Gen 9: 24-26
        """
        starters = [
            # Gen 1
            PokemonSpec(species="Bulbasaur", level=5, moves=["tackle", "growl"], ability="overgrow"),
            PokemonSpec(species="Charmander", level=5, moves=["scratch", "growl"], ability="blaze"),
            PokemonSpec(species="Squirtle", level=5, moves=["tackle", "tailwhip"], ability="torrent"),
            # Gen 2
            PokemonSpec(species="Chikorita", level=5, moves=["tackle", "growl"], ability="overgrow"),
            PokemonSpec(species="Cyndaquil", level=5, moves=["tackle", "leer"], ability="blaze"),
            PokemonSpec(species="Totodile", level=5, moves=["scratch", "leer"], ability="torrent"),
            # Gen 3
            PokemonSpec(species="Treecko", level=5, moves=["pound", "leer"], ability="overgrow"),
            PokemonSpec(species="Torchic", level=5, moves=["scratch", "growl"], ability="blaze"),
            PokemonSpec(species="Mudkip", level=5, moves=["tackle", "growl"], ability="torrent"),
            # Gen 4
            PokemonSpec(species="Turtwig", level=5, moves=["tackle", "withdraw"], ability="overgrow"),
            PokemonSpec(species="Chimchar", level=5, moves=["scratch", "leer"], ability="blaze"),
            PokemonSpec(species="Piplup", level=5, moves=["pound", "growl"], ability="torrent"),
            # Gen 5
            PokemonSpec(species="Snivy", level=5, moves=["tackle", "leer"], ability="overgrow"),
            PokemonSpec(species="Tepig", level=5, moves=["tackle", "tailwhip"], ability="blaze"),
            PokemonSpec(species="Oshawott", level=5, moves=["tackle", "tailwhip"], ability="torrent"),
            # Gen 6
            PokemonSpec(species="Chespin", level=5, moves=["tackle", "growl"], ability="overgrow"),
            PokemonSpec(species="Fennekin", level=5, moves=["scratch", "tailwhip"], ability="blaze"),
            PokemonSpec(species="Froakie", level=5, moves=["pound", "growl"], ability="torrent"),
            # Gen 7
            PokemonSpec(species="Rowlet", level=5, moves=["tackle", "growl"], ability="overgrow"),
            PokemonSpec(species="Litten", level=5, moves=["scratch", "growl"], ability="blaze"),
            PokemonSpec(species="Popplio", level=5, moves=["pound", "growl"], ability="torrent"),
            # Gen 8
            PokemonSpec(species="Grookey", level=5, moves=["scratch", "growl"], ability="overgrow"),
            PokemonSpec(species="Scorbunny", level=5, moves=["tackle", "growl"], ability="blaze"),
            PokemonSpec(species="Sobble", level=5, moves=["pound", "growl"], ability="torrent"),
            # Gen 9
            PokemonSpec(species="Sprigatito", level=5, moves=["scratch", "tailwhip"], ability="overgrow"),
            PokemonSpec(species="Fuecoco", level=5, moves=["tackle", "leer"], ability="blaze"),
            PokemonSpec(species="Quaxly", level=5, moves=["pound", "growl"], ability="torrent"),
        ]
        
        if 0 <= choice_idx < len(starters):
            spec = starters[choice_idx]
        else:
            spec = starters[1] # Default Charmander
            
        return MonInstance(id=str(uuid.uuid4()), spec=spec, current_hp=100, in_party=False)
        
    def roll_encounter(self, route: str, current_roster: List[MonInstance]) -> Optional[MonInstance]:
        if route not in self.encounters:
            return None
            
        options = self.encounters[route]
        if not options: return None
        
        # Dupes Clause: Try 5 times to find non-dupe
        existing_species = {m.spec.species for m in current_roster}
        
        for _ in range(10):
            # Weighted Random?
            # Rates sum to ?? check csv. Usually 100% per time of day.
            # We treat list as uniform or use 'rate' field.
            
            # Simple weighted selection
            total_rate = sum(o['rate'] for o in options)
            if total_rate <= 0:
                 pick = random.choice(options)
            else:
                 r = random.uniform(0, total_rate)
                 upto = 0
                 pick = options[0]
                 for o in options:
                     if results := o: # logic fix
                         pass
                     if upto + o['rate'] >= r:
                         pick = o
                         break
                     upto += o['rate']
            
            species = pick['species']
            
            # Check Dupe (Evolution Family?)
            # Strict species check for now
            if species in existing_species:
                continue
                
            # Found unique
            # Parse Level
            lvl_str = pick['level'] # "2-4"
            try:
                if "-" in lvl_str:
                    min_l, max_l = map(int, lvl_str.split("-"))
                    level = random.randint(min_l, max_l)
                else:
                    level = int(lvl_str)
            except:
                level = 5
                
            return self._create_mon(species, level)
            
        return None # Failed to find non-dupe
        
    def _create_mon(self, species: str, level: int) -> MonInstance:
        ability = self.moveset_gen.get_ability(species)
        spec = PokemonSpec(species=species, level=level, moves=[], ability=ability)
        return MonInstance(id=str(uuid.uuid4()), spec=spec, current_hp=100, in_party=False)
