import json
import os
import random
import urllib.request
import urllib.error
from typing import Dict, List, Optional
from threading import Lock

class SmogonDataFetcher:
    """
    Fetches and parses Smogon Chaos data to generate realistic random teams.
    """
    BASE_URL = "https://www.smogon.com/stats"
    _shared_data: Dict[str, Dict] = {} # Class-level cache shared by all instances in process
    _shared_lock = Lock()

    def __init__(self, cache_dir: str = "data/smogon_cache", formats: List[str] = None, rating: int = 1695):
        self.cache_dir = cache_dir
        self.formats = formats if formats else ["gen9ou"]
        self.rating = rating
        self._lock = Lock()
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.load_all_data()
        
    def load_all_data(self):
        for fmt in self.formats:
            self.load_data_for_format(fmt)

    @property
    def data(self):
        return self._shared_data

    def load_data_for_format(self, format_id: str):
        """Loads data from cache or downloads it."""
        # Check if already loaded in shared cache
        if format_id in self._shared_data:
            return

        with self._shared_lock:
            # Double-check locking pattern
            if format_id in self._shared_data:
                return
                
            filename = f"{format_id}-{self.rating}.json"
            path = os.path.join(self.cache_dir, filename)
            
            if os.path.exists(path):
                print(f"Loading Smogon data for {format_id} from {path}...")
                with open(path, "r") as f:
                    self._shared_data[format_id] = json.load(f)
            else:
                print(f"Downloading Smogon data for {format_id} to {path}...")
                self.download_data(format_id, path)
                # After download, load it
                if os.path.exists(path):
                     with open(path, "r") as f:
                        self._shared_data[format_id] = json.load(f)
        
    def download_data(self, format_id: str, path: str):
        # Determine latest stats URL (Hardcoded for now to avoid scraping index directory)
        # Using 2024-11 as a recent safe bet, or fallback to older if needed.
        date = "2025-11" 
        url = f"{self.BASE_URL}/{date}/chaos/{format_id}-{self.rating}.json"
        
        try:
            self._download_url(url, path)
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            # Fallback to previous month
            date = "2024-10"
            url = f"{self.BASE_URL}/{date}/chaos/{format_id}-{self.rating}.json"
            print(f"Retrying with {url}...")
            try:
                self._download_url(url, path)
            except Exception as e2:
                print(f"Failed fallback date: {e2}")
                # Try rating 0 (often more available) if not already
                if self.rating != 0:
                     url = f"{self.BASE_URL}/{date}/chaos/{format_id}-0.json"
                     print(f"Retrying with rating 0: {url}...")
                     try:
                         self._download_url(url, path)
                         return
                     except Exception as e3:
                         print(f"Failed rating 0 fallback: {e3}")
                
                # Create empty dummy data to prevent crash
                self.data[format_id] = {"data": {}}
                with open(path, "w") as f:
                    json.dump({"data": {}}, f)
                    
    def _download_url(self, url: str, path: str):
         with urllib.request.urlopen(url) as response:
             content = response.read()
             data = json.loads(content)
             with open(path, "w") as f:
                 json.dump(data, f)
             # Just updated storage via load_data_for_format logic mostly, but here we set it explicitly if needed
             # self.data is updated by the caller usually? No, download_url is helper.
             # We need to ensure self.data is populated in the wrapper.
             # Actually, download_url just downloads. load_data reads it back.
             # But let's return it or set it.
             pass
             
    def get_weighted_choice(self, items: Dict[str, float]) -> Optional[str]:
        if not items: return None
        total = sum(items.values())
        if total == 0: return random.choice(list(items.keys()))
        
        r = random.uniform(0, total)
        current = 0
        for item, weight in items.items():
            current += weight
            if r <= current:
                return item
        return list(items.keys())[-1]
        
    def generate_team(self, size: int = 6, format_id: str = None) -> str:
        """
        Generates a team string in Showdown format.
        """
        if not format_id:
            # Pick random format from loaded ones
            format_id = random.choice(self.formats)
            
        if format_id not in self.data:
            # Try to load it?
            self.load_data_for_format(format_id)
            
        data_source = self.data.get(format_id)
        if not data_source or "data" not in data_source:
             return ""
            
        usage_data = data_source["data"]
        
        # Pick 6 distinct pokemon based on raw usage?
        # Chaos data keys are species names. Values are usage objects.
        # "usage" field is raw count or usage percentage? In chaos it's "usage" float.
        
        all_mons = list(usage_data.keys())
        # To scale properly, we might heavily weight top used.
        # Simple weighted sample without replacement is tricky.
        # We'll just sample repeatedly and uniqueify.
        
        team_parts = []
        
        # Filter out weird stuff
        valid_mons = [m for m in all_mons if usage_data[m].get("usage", 0) > 0.01] # >1% usage
        if not valid_mons: 
            print(f"Warning: No valid mons found for format {format_id}")
            return ""

        # Simple random sample purely by index might ignore usage weights, 
        # but let's just pick from weighted distribution of usage.
        weights = [usage_data[m]["usage"] for m in valid_mons]
        
        chosen_mons = []
        while len(chosen_mons) < size:
             # Weighted random choice
             mon = random.choices(valid_mons, weights=weights, k=1)[0]
             if mon not in chosen_mons:
                 chosen_mons.append(mon)
                 
        for mon_name in chosen_mons:
            info = usage_data[mon_name]
            
            # 1. Moves
            # "Moves" is Dict[MoveName, Weight]
            # Filter valid moves first (exclude empty strings common in raw stats)
            move_pool = info.get("Moves", {})
            valid_pool = {k: v for k, v in move_pool.items() if k and k.strip() != ""}
            
            moves = []
            if valid_pool:
                attempts = 0
                max_moves = min(4, len(valid_pool))
                
                while len(moves) < max_moves and attempts < 20:
                    m = self.get_weighted_choice(valid_pool)
                    if m and m not in moves:
                        moves.append(m)
                    attempts += 1
                
                # Critical Fallback: Ensure at least one move
                if not moves:
                    # Pick absolute highest usage move
                    best_move = max(valid_pool, key=valid_pool.get)
                    moves.append(best_move)
            else:
                # No valid moves in pool? (Unlikely unless weird data)
                # Fallback to Struggle or just ignore (Showdown might reject)
                pass
                
            # 2. Ability
            ability = self.get_weighted_choice(info.get("Abilities", {}))
            
            # 3. Item
            item = self.get_weighted_choice(info.get("Items", {}))
            if item and item.lower() == "nothing": item = None
            
            # 4. Nature/Spread (Simplified)
            # Chaos data stores 'Spreads' as "Nature:HP/Atk/Def/SpA/SpD/Spe": weight
            spread_str = self.get_weighted_choice(info.get("Spreads", {}))
            nature = "Serious"
            evs = {}
            if spread_str:
                parts = spread_str.split(":")
                if len(parts) == 2:
                    nature = parts[0].strip()
                    ev_vals = parts[1].split("/")
                    if len(ev_vals) == 6:
                        evs = {
                            "HP": ev_vals[0], "Atk": ev_vals[1], "Def": ev_vals[2],
                            "SpA": ev_vals[3], "SpD": ev_vals[4], "Spe": ev_vals[5]
                        }
            
            # Build String
            lines = [f"{mon_name} @ {item}" if item else mon_name]
            lines.append(f"Ability: {ability}")
            # lines.append(f"Nature: {nature}")
            if evs:
                 ev_str = " / ".join([f"{v} {k}" for k, v in evs.items() if int(v) > 0])
                 if ev_str: lines.append(f"EVs: {ev_str}")
            
            # Nature line disabled due to persistent parsing error (Showdown sees 'naturetimid')
            # lines.append(f"Nature: {nature}")
            
            for m in moves:
                lines.append(f"- {m}")
                
            team_parts.append("\n".join(lines))
            
        return "\n\n".join(team_parts)

if __name__ == "__main__":
    fetcher = SmogonDataFetcher(formats=["gen9ou", "gen9uber", "gen9uu"], rating=0) # 0 for general stats often available
    team = fetcher.generate_team()
    print("Generated Team:")
    print(team)
