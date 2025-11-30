import numpy as np
from gymnasium import spaces
from poke_env.environment.singles_env import SinglesEnv
from poke_env.battle import AbstractBattle

class BattleEnv(SinglesEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_token = 0 # 0: Safe, 1: Balanced, 2: Desperate
        
        # Workaround for SingleAgentWrapper if observation_spaces is missing
        if not hasattr(self, "observation_spaces") or not self.observation_spaces:
            low, high, shape, dtype = self.describe_embedding()
            space = spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
            self.observation_spaces = {agent: space for agent in self.possible_agents}
            
        if not hasattr(self, "action_spaces") or not self.action_spaces:
            # Assuming standard Gen 8 action space size of 22
            # 4 moves + 6 switches + 4 moves (mega) + 4 moves (z-move) + 4 moves (dynamax)
            # Actually poke-env usually uses 22.
            # But let's try to get it from the class if possible, or hardcode.
            space = spaces.Discrete(22) 
            self.action_spaces = {agent: space for agent in self.possible_agents}
        
    def reset(self, seed=None, options=None):
        import sys
        # sys.stderr.write(f"DEBUG: reset called on instance {id(self)}. _challenge_task: {getattr(self, '_challenge_task', 'Not Set')}\n")
        # if hasattr(self, '_challenge_task') and self._challenge_task:
        #     sys.stderr.write(f"DEBUG: _challenge_task done? {self._challenge_task.done()}\n")
        #     if self._challenge_task.done():
        #         try:
        #             sys.stderr.write(f"DEBUG: _challenge_task result: {self._challenge_task.result()}\n")
        #         except Exception as e:
        #             sys.stderr.write(f"DEBUG: _challenge_task exception: {e}\n")
        # sys.stderr.flush()
        
        self._last_fainted = {} # Reset fainted tracking
                    
        if options and "risk_token" in options:
            self.risk_token = options["risk_token"]
        else:
            # Default or random if not specified
            self.risk_token = np.random.randint(0, 3)
            
        print(f"DEBUG: BattleEnv.reset calling super().reset()...", flush=True)
        ret = super().reset(seed=seed, options=options)
        print(f"DEBUG: BattleEnv.reset returned from super().reset().", flush=True)
        return ret

    def calc_reward(self, battle: AbstractBattle) -> float:
        reward = 0.0
        
        # Win/Loss
        if battle.finished:
            if battle.won:
                reward += 1.0
            else:
                reward -= 1.0
                
        # Death penalty based on risk
        # Count fainted mons in current battle
        current_fainted = len([mon for mon in battle.team.values() if mon.fainted])
        
        # Get last fainted count for this specific battle instance
        last_fainted = self._last_fainted.get(battle, 0)
        
        new_deaths = current_fainted - last_fainted
        
        if new_deaths > 0:
            base_penalty = 0.1
            if self.risk_token == 0: # Safe
                penalty = 0.5
            elif self.risk_token == 1: # Balanced
                penalty = 0.1
            else: # Desperate
                penalty = 0.0
            
            reward -= penalty * new_deaths
            
        # Update last fainted count
        self._last_fainted[battle] = current_fainted
            
        # HP preservation reward/penalty could also be added here
        
        return reward

    def embed_battle(self, battle: AbstractBattle):
        # Very simple embedding for now
        # 1. Active mon HP ratio
        # 2. Opponent active mon HP ratio
        # 3. Risk token (one-hot encoded: 3 dims)
        # Total dims: 2 + 3 = 5
        
        # We can add more features later
        
        my_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        op_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0.0
        
        risk_encoding = np.zeros(3)
        risk_encoding[self.risk_token] = 1.0
        
        return np.concatenate(([my_hp, op_hp], risk_encoding))

    def describe_embedding(self):
        # Low, High, Shape, Dtype
        return (
            np.array([0, 0, 0, 0, 0], dtype=np.float32),
            np.array([1, 1, 1, 1, 1], dtype=np.float32),
            (5,),
            np.float32
        )

    def action_to_order(self, action, battle, fake=False, strict=True):
        from poke_env.player.battle_order import BattleOrder
        try:
            return super().action_to_order(action, battle, fake=fake, strict=strict)
        except (AssertionError, ValueError) as e:
            # Fallback to a random valid move
            # print(f"DEBUG: Invalid action {action} for battle {battle.battle_tag}: {e}. Picking random move.")
            return self.choose_random_move(battle)

    def choose_random_move(self, battle):
        from poke_env.player.battle_order import SingleBattleOrder, ForfeitBattleOrder
        import random
        
        if battle.available_moves:
            return SingleBattleOrder(random.choice(battle.available_moves))
        
        if battle.available_switches:
            return SingleBattleOrder(random.choice(battle.available_switches))
            
        return ForfeitBattleOrder()


