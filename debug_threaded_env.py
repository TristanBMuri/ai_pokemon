
import logging
import sys
import time
from nuzlocke_gauntlet_rl.envs.threaded_ray_env import ThreadedBattleEnv
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder
from nuzlocke_gauntlet_rl.data.smogon import SmogonDataFetcher

# Configure logging to see everything
logging.basicConfig(level=logging.INFO)

def test_threaded_env():
    print("STARTING DEBUG SCRIPT...", flush=True)
    
    # Setup Configs
    smogon = SmogonDataFetcher()
    team_str = smogon.generate_team()
    
    agent_config = AccountConfiguration("Placeholder", None)
    agent_team = ConstantTeambuilder(team_str)
    
    opponent_config = AccountConfiguration("Placeholder", None)
    
    class DojoOpponent(SimpleHeuristicsPlayer):
         def __init__(self, **kwargs):
             super().__init__(**kwargs)
             self._team = ConstantTeambuilder(smogon.generate_team())

    print("Creating ThreadedBattleEnv...", flush=True)
    # Using default server url
    server_config = ServerConfiguration("ws://localhost:8000/showdown/websocket", None)
    
    env = ThreadedBattleEnv(
        agent_config=agent_config,
        agent_team=agent_team,
        opponent_cls=DojoOpponent,
        opponent_config=opponent_config,
        server_configuration=server_config
    )
    
    print("Calling env.reset()...", flush=True)
    obs, info = env.reset()
    print("Reset Complete! Got Obs.", flush=True)
    
    for i in range(5):
        print(f"Step {i}", flush=True)
        env.step(0)
        time.sleep(0.1)
        
    print("Test Complete.", flush=True)

if __name__ == "__main__":
    test_threaded_env()
