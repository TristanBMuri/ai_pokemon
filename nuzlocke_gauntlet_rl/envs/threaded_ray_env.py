import asyncio
import threading
import queue
import uuid
import time
import numpy as np
from gymnasium import Env, spaces
from poke_env.player import Player
from nuzlocke_gauntlet_rl.envs.embedding import BattleEmbedder

# Import smogon for team generation if needed, but we can pass it in
# We need to ensure we can pass server config and account config

class BridgePlayer(Player):
    def __init__(self, action_queue, obs_queue, **kwargs):
        super().__init__(**kwargs)
        self.action_queue = action_queue
        self.obs_queue = obs_queue
        self.embedder = BattleEmbedder()
        self.current_risk = 0
        self._last_fainted_count = 0 
        self._current_battle_ref = None

    def calc_reward(self, battle):
        reward = 0.0
        # Win/Loss
        if battle.finished:
            if battle.won:
                reward += 1.0
            else:
                reward -= 1.0
        
        # Fainted Penalty
        # Count fainted in current battle
        current_fainted = len([mon for mon in battle.team.values() if mon.fainted])
        
        # Use stored reference to check if this is a new battle or same battle
        if self._current_battle_ref != battle:
             self._current_battle_ref = battle
             self._last_fainted_count = 0
             
        new_deaths = current_fainted - self._last_fainted_count
        if new_deaths > 0:
            penalty = 0.1 # Base
            # Simplified risk penalty for now
            reward -= penalty * new_deaths
            
        self._last_fainted_count = current_fainted
        return reward

    async def choose_move(self, battle):
        # 1. Embed State
        obs = self.embedder.embed_battle(battle, risk_token=self.current_risk)
        
        # 2. Calculate continuous reward/term/trunc
        reward = self.calc_reward(battle)
        terminated = False # We don't terminate mid-battle usually, unless win/loss
        truncated = False
        info = {}
        
        # Note: poke-env calls choose_move only when a move is needed.
        # But for the very last step (game over), it often DOESN'T call choose_move.
        # It handles battle_end separately. However, for RLLib we need to yield the final reward.
        # This is a bit tricky in poke-env. 
        # Usually, the 'battle_end' event helps. 
        # But let's assume standard steps first.
        
        # Send Observation to Ray
        self.obs_queue.put((obs, reward, terminated, truncated, info))
        
        # 3. Wait for Action from Ray (Async Sleep Loop)
        while self.action_queue.empty():
            await asyncio.sleep(0.001) # Critical for Heartbeat
            
        action = self.action_queue.get()
        
        # Special signal to stop?
        if action == "STOP":
            return "/forfeit"
            
        # Convert Action to Move
        # We need the action space wrapper logic here or just raw mapping.
        # Using a simple fallback for now.
        
        if battle.available_moves:
             # Basic mapping: 0-3 moves, 4-9 switches
             # This assumes action is an int
             if action < 4 and len(battle.available_moves) > action:
                 return self.create_order(battle.available_moves[action])
             elif action < 4:
                 return self.create_order(battle.available_moves[0]) # Fallback
             else:
                 return self.choose_random_move(battle)
        
        return self.choose_random_move(battle)

    def choose_random_move(self, battle):
        import random
        if battle.available_moves:
            return self.create_order(random.choice(battle.available_moves))
        if battle.available_switches:
            return self.create_order(random.choice(battle.available_switches))
        return self.create_order("/skip")

    def _battle_finished_callback(self, battle):
        # When battle finishes, we need to send the final reward/observation to Ray
        # so it knows the episode is done.
        reward = self.calc_reward(battle)
        terminated = True
        truncated = False
        info = {}
        
        # Send final transition
        # We need a dummy observation or the final state
        obs = self.embedder.embed_battle(battle, risk_token=self.current_risk)
        self.obs_queue.put((obs, reward, terminated, truncated, info))
        
        # We don't wait for an action here because the episode is over.
        pass


class ThreadedBattleEnv(Env):
    def __init__(self, agent_config, agent_team, opponent_cls, opponent_config, server_configuration, start_challenge_loop=True):
        self.action_queue = queue.Queue(maxsize=1)
        self.obs_queue = queue.Queue(maxsize=1)
        
        self.server_configuration = server_configuration
        
        self.embedder = BattleEmbedder()
        low, high, shape, dtype = self.embedder.describe_embedding()
        # Relax bounds to avoid crashes if embedder produces out-of-range values
        self.observation_space = spaces.Box(low=float("-inf"), high=float("inf"), shape=shape, dtype=dtype)
        self.action_space = spaces.Discrete(22)
        
        # Dynamic ID
        suffix = uuid.uuid4().hex[:8]
        self.username = f"RayBot_{suffix}"
        
        # Create NEW config instead of mutating
        from poke_env import AccountConfiguration
        self.agent_config = AccountConfiguration(self.username, agent_config.password)
        
        self.agent_team = agent_team
        
        self.opponent_cls = opponent_cls
        self.opponent_config = opponent_config
        
        # Start Thread
        self.loop_thread = threading.Thread(target=self._start_background_loop, daemon=True)
        self.loop_thread.start()
        
    def _start_background_loop(self):
        import sys
        try:
            sys.stderr.write(f"[{self.username}] Thread STARTING\n")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create Opponent ON THIS LOOP
            # Use dynamic opponent name too
            otp_suffix = uuid.uuid4().hex[:8]
            opp_username = f"RayOpp_{otp_suffix}"
            
            # Re-create config
            from poke_env import AccountConfiguration
            opp_config = AccountConfiguration(opp_username, self.opponent_config.password)
            
            self.opponent = self.opponent_cls(
                account_configuration=opp_config,
                server_configuration=self.server_configuration, # Same server
                battle_format="gen9customgame",
            )
            
            self.player = BridgePlayer(
                action_queue=self.action_queue,
                obs_queue=self.obs_queue,
                account_configuration=self.agent_config,
                server_configuration=self.server_configuration,
                battle_format="gen9customgame",
                team=self.agent_team
            )
            
            async def main_loop():
                # Explicit Login
                # Wait for websocket to be initialized (Hack for race condition)
                import time
                start_wait = time.time()
                while not hasattr(self.player.ps_client, "websocket") or self.player.ps_client.websocket is None:
                    if time.time() - start_wait > 5:
                         break
                    await asyncio.sleep(0.01)
                
                # Use ps_client.log_in which uses the correct method name
                await self.player.ps_client.log_in(self.agent_config)
                
                try:
                    await asyncio.wait_for(self.player.ps_client.logged_in.wait(), timeout=5)
                except asyncio.TimeoutError:
                    pass
                
                print(f"[{self.username}] Logged in.", flush=True)
                
                # Wait for opponent websocket too
                while not hasattr(self.opponent.ps_client, "websocket") or self.opponent.ps_client.websocket is None:
                    await asyncio.sleep(0.01)
                    
                await self.opponent.ps_client.log_in(opp_config)
                await self.opponent.ps_client.logged_in.wait()
                # print(f"[{self.opponent.username}] Opponent Logged in.", flush=True)
                
                # Challenge Loop
                while True:
                    # if len(self.player.battles) > 0:
                    #    await asyncio.sleep(0.01)
                    #    continue
                        
                    try:
                        await self.player.battle_against(self.opponent, n_battles=1)
                    except Exception as e:
                        # print(f"Error challenging: {e}", flush=True)
                        await asyncio.sleep(1)
            
            # Run
            loop.run_until_complete(main_loop())
        except Exception as e:
            print(f"CRITICAL THREAD ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    def reset(self, *, seed=None, options=None):
        # 1. Clear Queues to avoid stale data - REMOVED TO PREVENT RACE CONDITION DEADLOCK
        # In this continuous threaded architecture, if the background thread has already 
        # placed the first observation of the next battle in the queue, clearing it 
        # creates a deadlock where reset() waits forever.
        # with self.action_queue.mutex:
        #    self.action_queue.queue.clear()
        # with self.obs_queue.mutex:
        #    self.obs_queue.queue.clear()
            
        # 2. Trigger a Battle
        # In this architecture, we need to tell the background thread "Start a battle".
        # But 'choose_move' only happens when a battle starts.
        # So we need a mechanism to TRIGGER the battle.
        
        # Since 'BridgePlayer' is running in the background, we can define a 
        # method on it that initiates a challenge, and call it thread-safely.
        
        from poke_env.concurrency import POKE_LOOP
        # This is tricky because POKE_LOOP is inside the thread.
        # We can use a separate control queue or just assume constant challenges.
        
        # For the prototype, let's assume the background loop (lines 142-146) 
        # handles challenging. 
        # But we need to BLOCK here until 'choose_move' is called for the first time.
        
        # Wait for first observation
        print("DEBUG: ThreadedEnv Waiting for Reset Obs...", flush=True)
        obs, reward, term, trunc, info = self.obs_queue.get()
        print("DEBUG: ThreadedEnv Received Reset Obs!", flush=True)
        
        return obs, info

    def step(self, action):
        self.action_queue.put(action)
        obs, reward, term, trunc, info = self.obs_queue.get()
        return obs, reward, term, trunc, info
