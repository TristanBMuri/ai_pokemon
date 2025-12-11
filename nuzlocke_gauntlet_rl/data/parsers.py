import pandas as pd
import math
import re
from typing import List, Optional, Dict, Tuple
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec, TrainerSpec, GauntletSpec
from nuzlocke_gauntlet_rl.mechanics.map_core import GauntletMap, GauntletNode
import uuid

def load_trainer_order() -> List[Tuple[str, str, int]]:
    """
    Parses 'Default Mode Bosses v4.1 (with EVs) - Radical Red - Trainer Order.csv'
    Returns a list of (trainer_name, location, level_cap).
    """
    path = "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Trainer Order.csv"
    try:
        df = pd.read_csv(path, header=None)
    except FileNotFoundError:
        print(f"Warning: File {path} not found.")
        return []
        
    order = []
    # Col 3: Trainer Name
    # Col 5: Level Cap
    # Location is usually at i+1, Col 3
    
    n_rows = len(df)
    for i in range(n_rows):
        name_cell = df.iloc[i, 3]
        cap_cell = df.iloc[i, 5]
        
        if isinstance(name_cell, str) and isinstance(cap_cell, (int, float, str)):
            # Normalize name
            name = name_cell.strip()
            if "CLICK ON" in name or "OPTIONAL" in name: continue
            
            try:
                cap_str = str(cap_cell).strip()
                if not cap_str.isdigit(): continue
                cap = int(cap_str)
                
                # Get Location from next row
                loc = ""
                if i + 1 < n_rows:
                    loc_cell = df.iloc[i+1, 3]
                    if isinstance(loc_cell, str):
                        loc = loc_cell.strip()
                
                order.append((name, loc, cap))
            except ValueError:
                continue
                
    return order

def normalize_text(text: str) -> str:
    """Normalize text for fuzzy matching."""
    t = text.upper().replace(".", "").replace("MINI BOSS", "").strip()
    # Expansions
    t = t.replace("VIRID ", "VIRIDIAN ")
    t = t.replace("NUGG ", "NUGGET ")
    t = t.replace("ISL ", "ISLAND ")
    t = t.replace("MT ", "MOUNT ")
    return t

def parse_boss_csv(file_path: str, level_cap_lookup: Dict[str, int] = None) -> List[TrainerSpec]:
    """
    Parses a Radical Red Boss CSV file.
    :param level_cap_lookup: Dict mapping normalized "NAME|LOC" -> Cap.
    """
    try:
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        return []

    trainers = []
    n_rows, n_cols = df.shape
    poke_cols = [4, 9, 14, 19, 24, 29]
    
    i = 0
    while i < n_rows:
        cell_val = df.iloc[i, 2]
        
        if isinstance(cell_val, str) and any(x in cell_val for x in ["GYM LEADER", "BOSS", "RIVAL", "ROCKET", "ELITE FOUR", "CHAMPION", "TRAINER", "LASS", "CAMPER", "SAILOR", "BIKER", "CUE BALL", "NERD", "BRENDAN", "MAY", "ACE", "BEAUTY", "GENTLEMAN", "FISHERMAN", "BIRD", "TAMER", "BLACK BELT", "BURGLAR", "YOUNGSTER", "PICNICKER", "ROCKER", "GRUNT", "ARCHER", "ARIANA", "CATCHER", "SCIENTIST", "HIKER", "CRUSH", "PSYCHIC", "CHANNELER", "JUGGLER", "GAMBLER", "ENGINEER", "DOCTOR", "PAINTER", "OFFICER", "BUGSY", "WHITNEY", "MORTY", "CHUCK", "PRYCE", "JASMINE", "CLAIR", "FALKNER", "BUG CATCHER", "SWIMMER"]):
            
            parts = [p.strip() for p in cell_val.split('\n') if p.strip()]
            trainer_name_raw = " ".join(parts) # "VIRID. FOREST LASS ANNE"
            
            # Identify Name and Location from raw parts
            # Heuristic: Part 0 is usually Location if multiple parts
            # Last part is Name
            # But "SAILOR EDMOND & SAILOR TREVOR" is complex.
            
            current_cap = 50
            if level_cap_lookup:
                found_cap = None
                
                # Try permutations against lookup keys
                # Lookup keys: "RIVAL|ROUTE 22"
                
                # We construct a normalized string from raw
                norm_raw = normalize_text(trainer_name_raw)
                
                for key, cap in level_cap_lookup.items():
                    k_name, k_loc = key.split("|")
                    
                    # Check if both name and loc are present in the raw string
                    # e.g. "RIVAL" in "ROUTE 22 #1 RIVAL" AND "ROUTE 22" in "ROUTE 22 #1 RIVAL"
                    if k_name in norm_raw and k_loc in norm_raw:
                        found_cap = cap
                        break
                        
                    # Backup: If Name matches and k_loc is empty?
                    if k_name in norm_raw and not k_loc:
                        found_cap = cap
                        # continue searching for better match?
                
                # Fallback for "LASS ANNE": Key="LASS ANNE", Loc="VIRIDIAN FOREST"
                # Raw="VIRID FOREST LASS ANNE"
                if not found_cap:
                    pass

                if found_cap:
                    current_cap = found_cap

            trainer_display_name = trainer_name_raw
            
            team = []
            valid_team = False
            
            for col_idx in poke_cols:
                if col_idx >= n_cols: continue
                species = df.iloc[i, col_idx]
                if not isinstance(species, str) or species.strip() == "": continue
                if isinstance(species, float) and math.isnan(species): continue
                valid_team = True
                
                try:
                    level_val = df.iloc[i+1, col_idx]
                    if isinstance(level_val, str) and "Lv" in level_val:
                        match = re.search(r'- ?(\d+)', level_val)
                        offset = int(match.group(1)) if match else 0
                        level = max(1, current_cap - offset)
                    else:
                        level = int(level_val)
                except:
                    level = current_cap
                
                def clean_cell(r_offset):
                    val = df.iloc[i+r_offset, col_idx]
                    if not isinstance(val, str): return None
                    val = val.strip()
                    if val.lower() in ["no item", "none", "", "-"]:
                         return None
                    return val

                nature = clean_cell(4)
                ability = clean_cell(5)
                item = clean_cell(6)
                moves = []
                for r in range(7, 11):
                    m = df.iloc[i+r, col_idx]
                    if isinstance(m, str) and m.strip() not in ["", "-"]:
                        move_clean = m.strip()
                        if move_clean.lower().startswith("hp "): continue
                        if "Return" in move_clean or "Frustration" in move_clean: continue
                        if "cease" in move_clean.lower(): continue
                        moves.append(move_clean)
                
                poke = PokemonSpec(
                    species=species.strip(), level=level, moves=moves,
                    ability=ability, item=item, nature=nature
                )
                team.append(poke)
            
            if valid_team and team:
                trainers.append(TrainerSpec(name=trainer_display_name, team=team))
            
            i += 15
        else:
            i += 1
            
    return trainers

def load_all_trainers() -> List[TrainerSpec]:
    """
    Loads all trainers from all CSVs.
    """
    order = load_trainer_order()
    # Build Lookup: "NAME|LOC" -> Cap
    level_lookup = {}
    for name, loc, cap in order:
        # Key: Normalized Name | Normalized Loc
        k = f"{normalize_text(name)}|{normalize_text(loc)}"
        level_lookup[k] = cap

    files = [
        "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Kanto Leaders.csv",
        "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Indigo League.csv",
        "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Team Rocket.csv",
        "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Rivals.csv",
        "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Mini Bosses.csv",
        "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Johto Leaders.csv"
    ]
    
    all_specs = []
    for f in files:
        specs = parse_boss_csv(f, level_cap_lookup=level_lookup)
        all_specs.extend(specs)
            
    return all_specs

def load_complete_gauntlet() -> GauntletSpec:
    final_gauntlet_list = []
    
    # --- 1. Manual: Rival Lab (Lvl 5) ---
    rival_lab = TrainerSpec(
        name="Rival Lab",
        team=[PokemonSpec(species="Squirtle", level=5, moves=["Tackle", "Tail Whip"], ability="Torrent")]
    )
    final_gauntlet_list.append(rival_lab)
    
    # --- 2. Load Data ---
    order = load_trainer_order() # [(Name, Loc, Cap)]
    all_trainers = load_all_trainers() # List[TrainerSpec]
    
    # --- 3. Stitch ---
    added_names = set()
    
    for name, loc, cap in order:
        name_norm = normalize_text(name)
        loc_norm = normalize_text(loc)
        
        match = None
        candidates = []
        
        # Search all trainers for best match
        for t in all_trainers:
            t_name_norm = normalize_text(t.name)
            
            # Criteria:
            # 1. Name match (Bidirectional)
            # "LUCA" in "ROCKER LUCA" (True) or "ROCKER LUCA" in "LUCA" (False)
            if name_norm in t_name_norm or t_name_norm in name_norm:
                # 2. Location match (if provided)
                if loc_norm and (loc_norm in t_name_norm or t_name_norm in loc_norm):
                    # Strong match
                    candidates.append((t, 0)) # Match score 0 (Best)
                else:
                    # Weak Name match
                    candidates.append((t, 1))

        if candidates:
            # Sort: Priority score, then Level Diff
            # Calculate Level Diff
            def level_diff(t):
                max_lvl = max(p.level for p in t.team) if t.team else 0
                return abs(max_lvl - cap)
            
            candidates.sort(key=lambda x: (x[1], level_diff(x[0]))) # (Score, Diff)
            
            # Filter huge diffs (>5 levels) unless it's the only match?
            # Actually, with location matching, we expect close levels.
            best, score = candidates[0]
            
            # Relax tolerance for strong matches (Score 0)
            tolerance = 15 if score == 0 else 5
            
            if level_diff(best) <= tolerance:
                match = best
        
        if match:
             unique_name = f"{name} ({loc})" if loc else name
             if unique_name in added_names: unique_name += " (Duplicate)"
             
             new_spec = TrainerSpec(name=unique_name, team=match.team, level_cap=cap)
             final_gauntlet_list.append(new_spec)
             added_names.add(unique_name)
        else:
             # pass
             # print(f"[DEBUG] Failed to match: '{name}' Loc:'{loc}' Cap:{cap}")
             # Print top candidates
             if candidates:
                 print(f"   Top Candidates: {[(c[0].name, c[1], level_diff(c[0])) for c in candidates[:3]]}")


    return GauntletSpec(trainers=final_gauntlet_list)


def load_kanto_leaders() -> GauntletSpec:
    path = "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Kanto Leaders.csv"
    trainers = parse_boss_csv(path)
    return GauntletSpec(trainers=trainers)

def load_indigo_league() -> GauntletSpec:
    path = "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Indigo League.csv"
    trainers = parse_boss_csv(path)
    return GauntletSpec(trainers=trainers)

def load_team_rocket() -> GauntletSpec:
    path = "data/Default Mode Bosses v4.1 (with EVs) - Radical Red - Team Rocket.csv"
    trainers = parse_boss_csv(path)
    return GauntletSpec(trainers=trainers)

def load_extended_gauntlet() -> GauntletSpec:
    return load_complete_gauntlet() # Alias for now

def load_encounters(file_path: str = "data/Pokémon Locations & Raid Dens v4.1 - Radical Red - Grass & Caves.csv") -> Dict[str, List[Dict]]:
    """
    Parses the Encounter CSV.
    Returns { "Route 1": [ {species, rate, level}, ... ], ... }
    """
    try:
        df = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"Warning: Encounters file {file_path} not found.")
        return {}

    # Find Header Row (Look for ROUTE 1)
    header_row_idx = -1
    for i in range(len(df)):
        row = df.iloc[i]
        row_str = " ".join([str(x) for x in row if isinstance(x, str)])
        if "ROUTE 1" in row_str:
            header_row_idx = i
            break
            
    if header_row_idx == -1: return {}
    
    # Identify Blocks from Header Row
    row_locs = df.iloc[header_row_idx]
    sub_header = df.iloc[header_row_idx + 1]
    
    # Identify indices of "Pokémon" in sub_header to find blocks
    poke_col_indices = []
    for c in range(len(sub_header)):
        val = sub_header[c]
        if isinstance(val, str) and "Pok" in val: # Pokémon
            poke_col_indices.append(c)
            
    encounters_map = {}
    
    for p_col in poke_col_indices:
        # Find Location Name (Look to left in header_row)
        loc_name = None
        for offset in range(-3, 3):
             if p_col + offset >= 0 and p_col + offset < len(row_locs):
                 val = row_locs[p_col + offset]
                 if isinstance(val, str) and len(val) > 2:
                     loc_name = val
                     break
        
        if not loc_name: continue
        
        loc_encounters = []
        # Read rows
        r = header_row_idx + 2
        while r < len(df):
            # Check Pokemon (p_col + 1 based on debug)
            if p_col + 1 >= df.shape[1]: break
            species = df.iloc[r, p_col + 1]
            
            if not isinstance(species, str):
                if pd.isna(species): break
                
            # Check Rarity (p_col - 1 based on debug)
            rarity_str = df.iloc[r, p_col - 1]
            rarity = 0.0
            if isinstance(rarity_str, str):
                try:
                    rarity = float(rarity_str.replace("%", "")) / 100.0
                except: pass
            
            # Check Level (p_col + 2)
            lvl = df.iloc[r, p_col + 2] 
            
            loc_encounters.append({
                "species": species.strip(),
                "rate": rarity,
                "level": str(lvl)
            })
            r += 1
            
        encounters_map[loc_name] = loc_encounters
        
    return encounters_map

def build_gauntlet_graph(gauntlet_spec: GauntletSpec) -> GauntletMap:
    """
    Constructs a GauntletMap from a linear GauntletSpec.
    Currently creates a linear chain of GYM nodes.
    """
    g_map = GauntletMap()
    
    prev_node_id = None
    
    for i, trainer in enumerate(gauntlet_spec.trainers):
        node_id = str(uuid.uuid4())
        
        # Determine Type (Heuristic for now, or just all Gyms?)
        # Logic: If it's in the main list, it's a mandatory fight for now.
        node_type = GauntletNode.TYPE_GYM
        
        node = GauntletNode(
            node_id=node_id,
            node_type=node_type,
            name=trainer.name,
            data={"trainer": trainer, "trainer_idx": i}
        )
        
        g_map.add_node(node)
        
        if prev_node_id:
            # Link previous to current
            prev_node = g_map.get_node(prev_node_id)
            prev_node.add_edge(node_id)
        else:
            g_map.set_start(node_id)
            
        prev_node_id = node_id
        
    g_map.end_node_id = prev_node_id
    return g_map
