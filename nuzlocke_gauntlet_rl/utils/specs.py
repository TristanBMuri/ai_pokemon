from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class PokemonSpec(BaseModel):
    """Defines a static Pokémon configuration (species, moves, etc.)."""
    species: str
    level: int
    moves: List[str]
    ability: Optional[str] = None
    item: Optional[str] = None
    nature: Optional[str] = None
    evs: dict[str, int] = Field(default_factory=dict)
    ivs: dict[str, int] = Field(default_factory=dict)

    def to_showdown_format(self) -> str:
        """Converts the spec to a Showdown-compatible string."""
        lines = []
        
        # Item mapping
        item = self.item
        if item == "Well. Mask": item = "Wellspring Mask"
        if item == "Hear. Mask": item = "Hearthflame Mask"
        if item == "Corn. Mask": item = "Cornerstone Mask"
        if item == "terrainextend": item = "Terrain Extender"
        
        item_str = f" @ {item}" if item else ""
        
        # Normalize species for Showdown
        species = self.species
        if species == "Kyogre-P":
            species = "Kyogre-Primal"
        elif species == "Groudon-P":
            species = "Groudon-Primal"
        elif species == "Ogerpon-W":
            species = "Ogerpon-Wellspring"
        elif species == "Ogerpon-H":
            species = "Ogerpon-Hearthflame"
        elif species == "Ogerpon-C":
            species = "Ogerpon-Cornerstone"
        elif species.endswith("-A"):
            species = species[:-2] + "-Alola"
        elif species.endswith("-G"):
            species = species[:-2] + "-Galar"
        elif species.endswith("-H"): # Hisui
            species = species[:-2] + "-Hisui"
        elif species.endswith("-P"): # Paldea (Tauros)
            species = species[:-2] + "-Paldea"
            
        lines.append(f"{species}{item_str}")
        lines.append(f"Level: {self.level}")
        
        if self.ability:
            # Sanitize ability (take first line if multiple)
            ability = self.ability.split('\n')[0].strip()
            if ability == "intimidateboth": ability = "Intimidate"
            lines.append(f"Ability: {ability}")
            
        if self.nature:
            lines.append(f"{self.nature} Nature")
        
        if self.evs:
            evs_str = " / ".join(f"{v} {k}" for k, v in self.evs.items())
            lines.append(f"EVs: {evs_str}")
            
        if self.ivs:
            ivs_str = " / ".join(f"{v} {k}" for k, v in self.ivs.items())
            lines.append(f"IVs: {ivs_str}")
            
        for move in self.moves:
            lines.append(f"- {move}")
            
        return "\n".join(lines)

class TrainerSpec(BaseModel):
    """Defines a trainer and their team."""
    name: str
    team: List[PokemonSpec]
    
class GauntletSpec(BaseModel):
    """Defines a sequence of trainers."""
    trainers: List[TrainerSpec]

class MonInstance(BaseModel):
    """Represents a runtime instance of a Pokémon in a run."""
    id: str
    spec: PokemonSpec
    current_hp: int
    alive: bool = True
    status: Optional[str] = None
    in_party: bool = False
