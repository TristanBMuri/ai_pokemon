import os
import asyncio
import threading
import time
import uuid
import gymnasium as gym
from poke_env import ServerConfiguration, AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import ConstantTeambuilder
from nuzlocke_gauntlet_rl.envs.battle_env import BattleEnv
from nuzlocke_gauntlet_rl.wrappers.single_agent_battle_wrapper import MySingleAgentWrapper
from nuzlocke_gauntlet_rl.data.smogon import SmogonDataFetcher

# Mimic the Global Fetcher
SMOGON_FETCHER = SmogonDataFetcher(format_id="gen9ou", rating=0)

class DebugEnv(gym.Env):
    def __init__(self):
        worker_index = 0
        port = 8000
        server_url = f"ws://localhost:{port}/showdown/websocket"
        print(f"Connecting to {server_url}...", flush=True)
        
        server_config = ServerConfiguration(server_url, None)
        self.opp_name = f"DebugOpp_{uuid.uuid4().hex[:6]}"
        self.agent_name = f"DebugAg_{uuid.uuid4().hex[:6]}"
        
        # Create Loop for this environment
        self.loop = asyncio.new_event_loop()
        
        # Start Thread 
        # CRITICAL: We might need to create players INSIDE this thread?
        self.thread_ready = threading.Event()
        
        def run_loop(loop):
            asyncio.set_event_loop(loop)
            self.thread_ready.set()
            loop.run_forever()
            
        t = threading.Thread(target=run_loop, args=(self.loop,), daemon=True)
        t.start()
        self.thread_ready.wait()
        
        # Instantiate Players safely?
        # poke-env players usually bind to the current loop on __init__ or connect?
        # Let's try creating them here (Main Thread) and see if they work when scheduled on Background Loop.
        
        class DojoOpponent(SimpleHeuristicsPlayer):
            def update_team(self):
                team_str = SMOGON_FETCHER.generate_team()
                self._team = ConstantTeambuilder(team_str)

        self.opponent = DojoOpponent(
            battle_format="gen9customgame",
            server_configuration=server_config,
            account_configuration=AccountConfiguration(self.opp_name, None)
        )
        self.opponent.update_team()
        
        agent_team = SMOGON_FETCHER.generate_team()
        self.pz_env = BattleEnv(
            battle_format="gen9customgame",
            server_configuration=server_config,
            account_configuration1=AccountConfiguration(self.agent_name, None),
        )
        self.pz_env.agent1._team = ConstantTeambuilder(agent_team)
        
        self._env = MySingleAgentWrapper(self.pz_env, opponent=self.opponent)
        self.action_space = self._env.action_space
        
        # START CHALLENGE LOOP
        self._start_challenge_loop()
        
    def _start_challenge_loop(self):
        def trigger():
            print(f"[{self.opp_name}] Challenge Loop Active", flush=True)
            time.sleep(2)
            while True:
                try:
                    async def do_challenge():
                        if not self.opponent.ps_client.logged_in.is_set():
                             print(f"[{self.opp_name}] Logging in...", flush=True)
                             await self.opponent.ps_client.logged_in.wait()
                             print(f"[{self.opp_name}] Logic in complete.", flush=True)

                        # print(f"[{self.opp_name}] Challenging...", flush=True)
                        await self.opponent.battle_against(self.pz_env.agent1, n_battles=1)
                        
                    asyncio.run_coroutine_threadsafe(do_challenge(), self.loop)
                except Exception as e:
                    print(f"Error: {e}")
                time.sleep(10)
                
        t2 = threading.Thread(target=trigger, daemon=True)
        t2.start()

    def reset(self):
        print("Reset called...", flush=True)
        # Force clear to be safe
        self.pz_env.agent1._battles = {}
        return self._env.reset()

    def step(self, action):
        print(f"Step called with action {action}", flush=True)
        return self._env.step(action)

if __name__ == "__main__":
    print("Starting Debug Env...", flush=True)
    env = DebugEnv()
    
    print("Resetting...", flush=True)
    obs, info = env.reset()
    print("Reset Complete! Obs received.", flush=True)
    
    print("Reset Complete! Obs received.", flush=True)
    
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        action = env.action_space.sample()
        obs, rew, done, trunc, info = env.step(action)
        total_reward += rew
        # print(f"Turn {steps}: Reward {rew}, Done {done}, Info {info}", flush=True)
        if steps % 10 == 0:
            print(f"Turn {steps}...", flush=True)
            
        if done or trunc:
            print(f"Episode Done at step {steps}. Total Reward: {total_reward}", flush=True)
            break
            # env.reset()
            
    print("Debug Success.", flush=True)
