import argparse
import os
import glob
import time
import threading
import asyncio
import uuid
import re

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import RecurrentPPO # Requires sb3-contrib

from poke_env import ServerConfiguration, AccountConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.teambuilder import ConstantTeambuilder

from nuzlocke_gauntlet_rl.envs.battle_env import BattleEnv
from nuzlocke_gauntlet_rl.wrappers.single_agent_battle_wrapper import MySingleAgentWrapper
from nuzlocke_gauntlet_rl.data.smogon import SmogonDataFetcher

def get_latest_checkpoint(model_dir: str, prefix: str) -> str:
    """Finds the latest checkpoint file in the directory."""
    files = glob.glob(os.path.join(model_dir, f"{prefix}_*.zip"))
    if not files: return None
    # Sort by modification time or step number in filename
    # Assuming standard format "name_steps.zip"
    def extract_steps(f):
        match = re.search(r'_(\d+)_steps', f)
        return int(match.group(1)) if match else 0
        
    latest = max(files, key=extract_steps)
    return latest

def train_dojo(total_steps: int, model_name: str, server_url: str):
    # Dirs
    models_dir = f"models/{model_name}"
    logs_dir = f"logs/{model_name}"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 1. Setup Smogon Data
    print("Initializing Smogon Data Engine...", flush=True)
    smogon = SmogonDataFetcher(format_id="gen9ou", rating=0) # General stats often better for variety
    
    # 2. Setup Opponent (Heuristic w/ Dynamic Teams)
    # We need a way to update the team. 
    # Option A: Subclass SimpleHeuristicsPlayer to pick a new team on reset?
    # Option B: Just update `_team` manually periodically.
    
    class DojoOpponent(SimpleHeuristicsPlayer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.smogon_fetcher = smogon
            self.update_team()
            
        def update_team(self):
            print("Generating new Dojo Opponent Team...", flush=True)
            team_str = self.smogon_fetcher.generate_team()
            # print(f"DEBUG TEAM:\n{team_str}\nEND DEBUG TEAM", flush=True)
            self._team = ConstantTeambuilder(team_str)
            
    # Config
    server_config = ServerConfiguration(server_url, None)
    # Pokemon Showdown limit is 18 characters.
    # DojoOpp_ (8) + 6 chars = 14 chars.
    opp_name = f"DojoOpp_{uuid.uuid4().hex[:6]}"
    agent_name = f"DojoAg_{uuid.uuid4().hex[:6]}"
    
    print(f"Connecting to Showdown at {server_url}...", flush=True)
    
    opponent = DojoOpponent(
        battle_format="gen9customgame", # Or gen9ou if valid
        server_configuration=server_config,
        account_configuration=AccountConfiguration(opp_name, None)
    )
    
    # Agent Env
    # Note: Agent Team is also static in standard BattleEnv unless we wrap it or use a similar custom player.
    # ideally Agent should also get random teams to learn general skills? OR keep a fixed team to master it?
    # For "Pretraining", random teams for Agent is BEST.
    
    # We don't have a "RandomTeamAgent" wrapper easily. 
    # Let's assume for now we use a fixed strong OU team for the agent, 
    # OR we make the agent randomly switch teams too.
    
    # Let's use a fixed "Balanced" team for the agent for now to ensure stability.
    agent_team = smogon.generate_team() # Generate once
    
    pz_env = BattleEnv(
        battle_format="gen9customgame",
        server_configuration=server_config,
        account_configuration1=AccountConfiguration(agent_name, None),
    )
    # Hack to set agent team since BattleEnv might not expose it easily or expects it in config?
    # BattleEnv usually just facilitates. The Player (Agent) inside needs the team.
    # pz_env.agent1 is the Player object.
    pz_env.agent1._team = ConstantTeambuilder(agent_team)
    
    env = MySingleAgentWrapper(pz_env, opponent=opponent)
    
    # 3. Challenge Loop (Async Thread)
    # We use a persistent background loop for async operations to avoid conflicts
    challenge_loopr = asyncio.new_event_loop()
    
    def _background_loop_runner(loop):
        asyncio.set_event_loop(loop)
        loop.run_forever()
        
    t_loop = threading.Thread(target=_background_loop_runner, args=(challenge_loopr,), daemon=True)
    t_loop.start()

    def challenge_trigger():
        # Wait for login
        time.sleep(5)
        print("Challenge Trigger: Background loop started. Sending challenges...", flush=True)
        
        while True:
            try:
                async def trigger_async():
                    # Ensure logged in
                    if not opponent.ps_client.logged_in.is_set():
                         # print("Waiting for opponent login...", flush=True)
                         await opponent.ps_client.logged_in.wait()
                    
                    # Check if we should challenge
                    # If using `n_battles=1`, it waits for completion? 
                    # No, battle_against returns a future.
                    # We just fire and forget, but let's not spam if busy.
                    
                    # print("Sending challenge...", flush=True)
                    await opponent.battle_against(pz_env.agent1, n_battles=1)
                
                # Schedule on the persistent loop
                asyncio.run_coroutine_threadsafe(trigger_async(), challenge_loopr)
                
            except Exception as e:
                print(f"Challenge Trigger Error: {e}", flush=True)
            
            # Wait before next challenge check
            # Battles take time. If we spam, we get "User busy".
            time.sleep(5)
            
    t_trigger = threading.Thread(target=challenge_trigger, daemon=True)
    t_trigger.start()
    
    # 4. Model Setup (Load or New)
    latest_checkpoint = get_latest_checkpoint(models_dir, model_name)
    
    if latest_checkpoint:
        print(f"Resuming training from {latest_checkpoint}...", flush=True)
        # Load env is crucial
        model = RecurrentPPO.load(latest_checkpoint, env=env)
        # Reset num_timesteps if you want to continue counting, 
        # but usually we want to add to it. SB3 tracks `num_timesteps`.
    else:
        print("Starting NEW training session...", flush=True)
        model = RecurrentPPO("MlpLstmPolicy", env, verbose=1, tensorboard_log=logs_dir)
        
    # 5. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=models_dir,
        name_prefix=model_name
    )
    
    # 6. Train
    print(f"Training for {total_steps} steps. Press Ctrl+C to pause/save.", flush=True)
    try:
        model.learn(total_timesteps=total_steps, callback=checkpoint_callback, progress_bar=True, reset_num_timesteps=False)
        print("Training Goal Reached.")
    except KeyboardInterrupt:
        print("\nTraining Interrupted by User.")
    finally:
        save_path = f"{models_dir}/{model_name}_final"
        print(f"Saving Final Model to {save_path}.zip...", flush=True)
        model.save(save_path)
        env.close()
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--model", type=str, default="dojo_master")
    parser.add_argument("--url", type=str, default="ws://localhost:8000/showdown/websocket")
    
    args = parser.parse_args()
    
    train_dojo(args.steps, args.model, args.url)
