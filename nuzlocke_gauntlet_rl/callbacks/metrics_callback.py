import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    Tracks:
    - Win Rate (rolling average)
    - Average Turns per Battle
    - Pokemon Fainted (Player vs Opponent)
    """
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = []
        self.turns = []
        
    def _on_step(self) -> bool:
        # Check for info dicts in locals
        # SB3 stores env infos in `self.locals['infos']`
        infos = self.locals.get("infos", [])
        
        for info in infos:
            if "episode" in info:
                # Standard SB3 episode info
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                
            # Custom metrics from NuzlockeEnv
            if "metrics" in info:
                metrics = info["metrics"]
                if "win" in metrics:
                    self.wins.append(metrics["win"])
                    self.logger.record("custom/win_rate", np.mean(self.wins[-100:])) # Rolling 100
                    
                if "turns" in metrics:
                    self.turns.append(metrics["turns"])
                    self.logger.record("custom/avg_turns", np.mean(self.turns[-100:]))
                    
                if "pokemon_fainted" in metrics:
                    self.logger.record("custom/pokemon_fainted", metrics["pokemon_fainted"])
                    
                if "opponent_fainted" in metrics:
                    self.logger.record("custom/opponent_fainted", metrics["opponent_fainted"])
                    
                if "trainer_idx" in metrics:
                    # Track max trainer reached in this rollout/episode
                    # Since we get step-by-step, we can just log the current one or max seen
                    self.logger.record("custom/trainer_idx", metrics["trainer_idx"])
                    
        return True
