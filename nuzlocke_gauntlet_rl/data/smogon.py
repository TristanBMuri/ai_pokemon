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
    
    def __init__(self, cache_dir: str = "data/smogon_cache", format_id: str = "gen9ou", rating: int = 1695):
        self.cache_dir = cache_dir
        self.format_id = format_id
        self.rating = rating
        self.data: Dict = {}
        self._lock = Lock()
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.load_data()
        
    def load_data(self):
        """Loads data from cache or downloads it."""
        filename = f"{self.format_id}-{self.rating}.json"
        path = os.path.join(self.cache_dir, filename)
        
        if os.path.exists(path):
            print(f"Loading Smogon data from {path}...")
            with open(path, "r") as f:
                self.data = json.load(f)
        else:
            print(f"Downloading Smogon data to {path}...")
            self.download_data(path)
            
    def download_data(self, path: str):
        # Determine latest stats URL (Hardcoded for now to avoid scraping index directory)
        # Using 2024-11 as a recent safe bet, or fallback to older if needed.
        date = "2024-11" 
        url = f"{self.BASE_URL}/{date}/chaos/{self.format_id}-{self.rating}.json"
        
        try:
            self._download_url(url, path)
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            # Fallback to previous month
            date = "2024-10"
            url = f"{self.BASE_URL}/{date}/chaos/{self.format_id}-{self.rating}.json"
            print(f"Retrying with {url}...")
            try:
                self._download_url(url, path)
            except Exception as e2:
                print(f"Failed fallback: {e2}")
                # Create empty dummy data to prevent crash
                self.data = {"data": {}}
                with open(path, "w") as f:
                    json.dump(self.data, f)
                    
    def _download_url(self, url: str, path: str):
         with urllib.request.urlopen(url) as response:
             content = response.read()
             data = json.loads(content)
             with open(path, "w") as f:
                 json.dump(data, f)
             self.data = data
             
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
        
    def generate_team(self, size: int = 6) -> str:
        """
        Generates a team string in Showdown format.
        """
        if "data" not in self.data:
            return "" # No data
            
        usage_data = self.data["data"]
        
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
        if not valid_mons: valid_mons = all_mons
        
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
            moves = []
            move_pool = info.get("Moves", {})
            # Pick 4 distinct moves
            # For simplicity, pick top 4 weighted or weighted random?
            # Weighted random 4 times.
            attempts = 0
            while len(moves) < 4 and attempts < 20:
                m = self.get_weighted_choice(move_pool)
                if m and m not in moves and m != "":
                    moves.append(m)
                attempts += 1
                
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
    fetcher = SmogonDataFetcher(format_id="gen9ou", rating=0) # 0 for general stats often available
    team = fetcher.generate_team()
    print("Generated Team:")
    print(team)
