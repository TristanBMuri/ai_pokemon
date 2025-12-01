from typing import Any, Dict, Tuple, Optional
import gymnasium as gym
from poke_env.environment.env import PokeEnv

class MySingleAgentWrapper(gym.Env):
    def __init__(self, env: PokeEnv, opponent=None):
        self.env = env
        self.opponent = opponent
        self.observation_space = list(env.observation_spaces.values())[0]
        self.action_space = list(env.action_spaces.values())[0]
        
    def reset(self, seed=None, options=None):
        print("MySingleAgentWrapper.reset called", flush=True)
        obs, infos = self.env.reset(seed=seed, options=options)
        agent_id = self.env.agents[0]
        return obs[agent_id], infos[agent_id]
        
    def step(self, action):
        # We only care about agent1
        actions = {
            self.env.agents[0]: action,
        }
        
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        
        agent_id = self.env.agents[0]
        return (
            obs[agent_id],
            rewards[agent_id],
            terms[agent_id],
            truncs[agent_id],
            infos[agent_id],
        )
        
    def render(self, mode="human"):
        return self.env.render(mode)
        
    def close(self):
        self.env.close()
