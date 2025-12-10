import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from rich.live import Live
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn
from collections import Counter, deque

class RichDashboardCallback(BaseCallback):
    """
    Displays a live training dashboard using Rich.
    - Gauntlet Progress
    - Win Rate
    - Roster Analytics
    - Recent Battles
    """
    def __init__(self, verbose=0):
        super(RichDashboardCallback, self).__init__(verbose)
        self.console = Console()
        self.layout = Layout()
        
        # Stats
        self.wins = deque(maxlen=100)
        self.recent_battles = deque(maxlen=5) # (Trainer, Win/Loss, Dead)
        self.roster_usage = Counter()
        self.current_trainer_idx = 0
        self.max_trainers = 60
        self.episode_count = 0
        
        # Setup Layout
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="stats"),
            Layout(name="roster")
        )
        
        # Live Context
        self.live = Live(self.layout, refresh_per_second=4, console=self.console)
        
    def _on_training_start(self):
        self.live.start() # Start rendering loop
        
    def _on_training_end(self):
        self.live.stop() # Stop rendering
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "metrics" in info:
                m = info["metrics"]
                if "win" in m:
                    self.wins.append(m["win"])
                    self.episode_count += 1
                    
                    # Log Battle
                    res = "[green]WIN[/]" if m["win"] else "[red]LOSS[/]"
                    t_idx = m.get("trainer_idx", 0)
                    self.current_trainer_idx = t_idx
                    dead = m.get("pokemon_fainted", 0)
                    self.recent_battles.append(f"Trainer {t_idx} | {res} | Dead: {dead}")
                    
            # Track Roster Usage (Need to extract from somewhere? Maybe info?)
            # Ideally the env would pass 'selected_party' in info.
            
        self.update_display()
        return True

    def update_display(self):
        # Header
        wr = np.mean(self.wins) if self.wins else 0.0
        self.layout["header"].update(
            Panel(f"Training Dashboard | Episodes: {self.episode_count} | Win Rate (100): {wr:.1%}", style="bold white on blue")
        )
        
        # Stats Table
        table = Table(title="Recent Battles")
        table.add_column("Trainer")
        table.add_column("Result")
        table.add_column("Deaths")
        
        for battle in reversed(self.recent_battles):
             parts = battle.split("|")
             if len(parts) >= 3:
                 table.add_row(parts[0], parts[1], parts[2])
                 
        self.layout["stats"].update(Panel(table, title="Battle Log"))
        
        # Roster Table (Placeholder for now)
        r_table = Table(title="Top Roster Picks")
        r_table.add_column("Pokemon")
        r_table.add_column("Count")
        # Populate if we had real usage data
        
        self.layout["roster"].update(Panel(r_table, title="Analytics"))
        
        # Footer (Progress)
        self.layout["footer"].update(
            Panel(f"Current Max Progress: Trainer {self.current_trainer_idx}/{self.max_trainers}")
        )
