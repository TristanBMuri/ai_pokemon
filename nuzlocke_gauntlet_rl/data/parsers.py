import pandas as pd
import math
from typing import List, Optional
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec, TrainerSpec, GauntletSpec

def parse_boss_csv(file_path: str) -> List[TrainerSpec]:
    """
    Parses a Radical Red Boss CSV file and returns a list of TrainerSpecs.
    Assumes the grid layout observed in 'Default Mode Bosses...csv'.
    """
    # Read without header
    df = pd.read_csv(file_path, header=None)
    
    trainers = []
    
    # Iterate through rows to find trainer blocks
    # We look for the "GYM LEADER" or similar text in column 2 (index 2)
    # Actually, looking at the output, "GYM LEADER\nBROCK" is in col 2.
    
    n_rows, n_cols = df.shape
    
    # Pokemon data columns start at index 4 and increment by 5
    # 4, 9, 14, 19, 24, 29
    poke_cols = [4, 9, 14, 19, 24, 29]
    
    i = 0
    while i < n_rows:
        # Check for trainer name block
        # The cell might contain "GYM LEADER\nNAME" or just "NAME" depending on the file
        # In the inspection, it was at col 2.
        cell_val = df.iloc[i, 2]
        
        if isinstance(cell_val, str) and ("GYM LEADER" in cell_val or "BOSS" in cell_val or "RIVAL" in cell_val or "ROCKET" in cell_val or "ELITE FOUR" in cell_val or "CHAMPION" in cell_val):
            # Found a trainer block
            trainer_name = cell_val.split('\n')[-1].strip() # Take the last line as name
            
            # Sometimes the name is just in the cell if it doesn't have the prefix
            if len(cell_val.split('\n')) == 1:
                 # Fallback logic if needed, but for now assume the format
                 pass

            team = []
            
            # Parse the 6 potential pokemon slots
            for col_idx in poke_cols:
                if col_idx >= n_cols:
                    continue
                    
                species = df.iloc[i, col_idx]
                
                # Check if there is a pokemon here (not NaN)
                if not isinstance(species, str) and isinstance(species, float) and math.isnan(species):
                    continue
                
                if isinstance(species, str) and species.strip() == "":
                    continue
                    
                # Level is at row i+1
                try:
                    level_val = df.iloc[i+1, col_idx]
                    level = int(level_val) if not math.isnan(level_val) else 50 # Fallback
                except:
                    level = 50

                # Nature at i+4
                nature = df.iloc[i+4, col_idx]
                if not isinstance(nature, str): nature = None
                
                # Ability at i+5
                ability = df.iloc[i+5, col_idx]
                if not isinstance(ability, str): ability = None
                
                # Item at i+6
                item = df.iloc[i+6, col_idx]
                if not isinstance(item, str): item = None
                
                # Moves at i+7 to i+10
                moves = []
                for r in range(7, 11):
                    move = df.iloc[i+r, col_idx]
                    if isinstance(move, str) and move.strip() != "-" and move.strip() != "":
                        moves.append(move.strip())
                
                # Create Spec
                # Note: EVs/IVs are further down, skipping for now for simplicity
                # as they require parsing the "BASE STATS" block which is offset differently
                
                poke = PokemonSpec(
                    species=species.strip(),
                    level=level,
                    moves=moves,
                    ability=ability.strip() if ability else None,
                    item=item.strip() if item else None,
                    nature=nature.strip() if nature else None
                )
                team.append(poke)
            
            if team:
                trainers.append(TrainerSpec(name=trainer_name, team=team))
            
            # Skip ahead to next block
            # The blocks seem to be around 25-30 rows. 
            # We can just increment by 1 and let the loop find the next one, 
            # but to be safe let's skip at least 15 rows.
            i += 15
        else:
            i += 1
            
    return trainers

def load_kanto_leaders() -> GauntletSpec:
    """Helper to load the Kanto Leaders gauntlet."""
    path = "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Kanto Leaders.csv"
    trainers = parse_boss_csv(path)
    return GauntletSpec(trainers=trainers)
