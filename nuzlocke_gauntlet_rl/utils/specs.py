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
