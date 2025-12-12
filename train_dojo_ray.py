import os
# AGGRESSIVE MEMORY SAVING:
# Prevent PyTorch/NumPy from spawning massive thread pools in each worker.
# With 30 workers, we don't want 30 * 32 threads!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# Prevent memory fragmentation (classic Python/Linux fix for "slow crawl")
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["RAY_DEDUP_LOGS"] = "0" # Keep full logs for debugging

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from nuzlocke_gauntlet_rl.envs.threaded_ray_env import ThreadedBattleEnv

# --- NEW: Import Transformer Model ---
from models.ray_transformer import PokemonTransformerModel

# Register Custom Model
ModelCatalog.register_custom_model("pokemon_transformer", PokemonTransformerModel)
from nuzlocke_gauntlet_rl.data.smogon import SmogonDataFetcher
from nuzlocke_gauntlet_rl.players.heuristics import RadicalRedLogic, SimpleHeuristicsLogic
from poke_env.player import Player
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder, Teambuilder
import logging
import uuid
import os

# Configure Logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    pass

# ==============================================================================
# CONFIGURATION & SCALING
# ==============================================================================
# 1. PARALLEL WORKERS (Ray "num_env_runners")
#    - How many separate python processes to spawn to run battles?
#    - Recommendation: 20-25 (Safe), 35 (Aggressive - Risk of OOM).
NUM_PARALLEL_WORKERS = 15

# 2. ENVS PER WORKER (Ray "num_envs_per_env_runner")
#    - How many battles happen inside EACH worker process at the same time?
#    - Keep at 1 for this ThreadedEnv to avoid complexity.
ENVS_PER_WORKER = 1 # Keep at 1. Scale NUM_PARALLEL_WORKERS instead.

# 3. SHOWDOWN SERVERS
#    - 4 Servers (Ports 8000-8003) are sufficient for 20+ workers.
#    - Workers are automatically load-balanced across these ports.

# 4. CHECKPOINT RESTORATION
#    - Path to a checkpoint directory to resume training from.
#    - Example: "/home/tristan/CodingProjects/ai_pokemon/models/ray_dojo"
#    - Set to None to start fresh.
CHECKPOINT_PATH = "/home/tristan/CodingProjects/ai_pokemon/models/ray_dojo_perfect_info"
# ==============================================================================

def env_creator(config):
    worker_index = config.worker_index if hasattr(config, "worker_index") else 0
    # Port Distribution
    port = 8000 + (worker_index % 4)
    print(f"DEBUG: env_creator worker={worker_index} port={port}", flush=True)

    server_url = f"ws://localhost:{port}/showdown/websocket"
    server_config = ServerConfiguration(server_url, None)
    
    # Smogon Data
    # Load multiple tiers for diversity
    # NOTE: The first battle might take a few seconds to download these if not cached.
    # gen9ubers (plural) and gen9uu
    # Added Gen 8-6 OU for historical diversity
    # Added Gen 9 RU, NU, PU, LC for power diversity (Lower Tiers & Level 5 battles)
    all_formats = [
        "gen9ou", "gen9ubers", "gen9uu",
        "gen9ru", "gen9nu", "gen9pu", "gen9lc", 
        "gen8ou", "gen7ou", "gen6ou"
    ]
    smogon = SmogonDataFetcher(formats=all_formats)
    
    # Tier Synchronization logic
    import random
    class TierManager:
        def __init__(self, formats):
             self.formats = formats
             self.current_tier = "gen9ou" # Default
             
        def pick_new_tier(self):
             self.current_tier = random.choice(self.formats)
             return self.current_tier

    tier_manager = TierManager(all_formats)

    # Agent Teambuilder (Calls Step)
    class AgentTeambuilder(Teambuilder):
         def __init__(self, smogon, tier_manager):
             self.smogon = smogon
             self.tier_manager = tier_manager
             
         def yield_team(self):
             # 1. Decide Tier for this Match
             new_tier = self.tier_manager.pick_new_tier()
             print(f"[Match Setup] Selected Tier: {new_tier}", flush=True)
             
             # 2. Generate Team
             raw = self.smogon.generate_team(format_id=new_tier)
             parsed = self.parse_showdown_team(raw)
             return self.join_team(parsed)

    agent_team = AgentTeambuilder(smogon, tier_manager)
    
    # Opponent Setup
    # ThreadedBattleEnv creates the opponent internally on the background thread
    # opponent_cls = SimpleHeuristicsPlayer # Removed
    
    # We define a custom Opponent class to inject the team
    class DynamicSmogonTeambuilder(Teambuilder):
        def __init__(self, smogon_fetcher, tier_manager):
             self.smogon = smogon_fetcher
             self.tier_manager = tier_manager
             
        def yield_team(self):
             # Use the SAME tier that the Agent just selected
             current_tier = self.tier_manager.current_tier
             
             # Generate raw Showdown text
             raw_team = self.smogon.generate_team(format_id=current_tier)
             # Parse and Pack it (removes newlines causing protocol errors)
             parsed = self.parse_showdown_team(raw_team)
             return self.join_team(parsed)

    # Mixed Opponent with Curriculum
    class CurriculumDojoOpponent(Player):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Two engines - Lightweight Logic Classes (No Player overhead)
            self.simple_ai = SimpleHeuristicsLogic()
            self.radical_ai = RadicalRedLogic()
            self._team = DynamicSmogonTeambuilder(smogon, tier_manager)
            
            # Curriculum State
            self.agent_wins = [] # Rolling buffer of bools (True = Agent Won)
            self.radical_prob = 0.0
            
        def choose_move(self, battle):
            # Decide for this turn based on pre-selected mode?
            # Or mix per turn? Usually per-battle is cleaner.
            # But here we are inside choose_move.
            # We should decide "mode" at start of battle.
            # Hack: Store mode in battle object or attribute?
            # Or just check randomly every turn? No, consistency is good.
            # Use `battle.turn` == 1 to decide?
            
            # Better: Decide based on probability derived from win rate
            
            # If we don't have a mode for this battle ID yet, pick one
            if not hasattr(self, "current_battle_mode") or self.current_battle_mode[0] != battle.battle_tag:
                 # Pick mode
                 roll = random.random()
                 if roll < self.radical_prob:
                     mode = "radical"
                 else:
                     mode = "simple"
                 self.current_battle_mode = (battle.battle_tag, mode)
                 # Log only occasionally to avoid spam
                 if random.random() < 0.05:
                     print(f"[Curriculum] WR: {self._get_agent_wr():.2f} -> Mode: {mode}", flush=True)
            else:
                 mode = self.current_battle_mode[1]

            if mode == "radical":
                return self.radical_ai.choose_move(battle)
            else:
                return self.simple_ai.choose_move(battle)

        @property
        def active_bot_name(self):
             if hasattr(self, "current_battle_mode"):
                 return self.current_battle_mode[1].capitalize() # "Radical" or "Simple"
             return "Unknown"

        def _battle_finished(self, battle, won):
            # won = True means WE (Opponent) won -> Agent Lost
            agent_won = not won
            self.agent_wins.append(agent_won)
            if len(self.agent_wins) > 50: self.agent_wins.pop(0)
            
            # Update Probability
            wr = self._get_agent_wr()
            # Scaling:
            # < 40% WR: 0% Radical
            # > 80% WR: 100% Radical
            # Linear in between
            if wr < 0.4:
                self.radical_prob = 0.0
            elif wr > 0.8:
                self.radical_prob = 1.0
            else:
                self.radical_prob = (wr - 0.4) * 2.5 # (0.4->0, 0.6->0.5, 0.8->1.0)
                
        def _get_agent_wr(self):
            if not self.agent_wins: return 0.5 # Assume balanced start
            return sum(self.agent_wins) / len(self.agent_wins)
            
        # teambuilder property is inherited from Player and uses self._team
        # def teambuilder(self):
        #      return self._team

    class DojoOpponent(CurriculumDojoOpponent):
         pass
             
    opponent_config = AccountConfiguration("Placeholder", None) # Name set dynamically in Env
    
    # Agent Config
    agent_config = AccountConfiguration("Placeholder", None) # Name set dynamically
    # agent_team is already defined above as Teambuilder instance
    
    return ThreadedBattleEnv(
        agent_config=agent_config,
        agent_team=agent_team, # Now passing Teambuilder instance, not constant
        opponent_cls=DojoOpponent,
        opponent_config=opponent_config,
        server_configuration=server_config
    )

class MetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, env_index, **kwargs):
        if not episode.last_info_for(): return
        info = episode.last_info_for()
        
        # General Opponent Appearance Logging
        # We assume a known list of bots for now, or discover them?
        # Fixed list ensures 0s are logged correctly for averages.
        KNOWN_BOTS = ["Simple", "Radical"]
        
        active_bot = info.get("opponent_bot_name", "Unknown")
        
        for bot in KNOWN_BOTS:
             # Log 1.0 if this bot was used, 0.0 otherwise
             # This makes the "mean" in TensorBoard equal to the usage percentage (e.g. 0.3 = 30%)
             is_used = 1.0 if bot == active_bot else 0.0
             episode.custom_metrics[f"opponent_appearance_{bot}"] = is_used
             
        # Also log the raw difficulty/prob if available
        if "opponent_radical_prob" in info:
             episode.custom_metrics["opponent_radical_prob"] = info["opponent_radical_prob"]

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Log number of active (reporting) workers
        # Ray Tune usually provides this in 'perf' stats
        reporting = result.get("perf", {}).get("num_env_runners_reporting", 0)
        
        # Start initializing custom_metrics if missing (rare but possible)
        if "custom_metrics" not in result:
            result["custom_metrics"] = {}
            
        result["custom_metrics"]["active_workers_count"] = reporting
        
        # Also log configured workers for comparison
        if hasattr(algorithm, "config"):
             result["custom_metrics"]["configured_workers_count"] = algorithm.config.num_env_runners

register_env("dojo_env", env_creator)

if __name__ == "__main__":
    print("STARTING THREADED RAY DOJO...", flush=True)
    
    # Using default Ray init (auto-detect resources)
    # Disable dashboard to prevent metrics buffering leak (connection errors seen in logs)
    # Cap Object Store Memory to 2GB to prevent ballooning
    # runtime_env guarantees these vars reach the workers!
    ray.init(
        ignore_reinit_error=True, 
        include_dashboard=False, 
        object_store_memory=2*1024*1024*1024,
        # EFFECTIVELY DISABLE METRICS AGENT:
        # Set reporting interval to 11 days (infinity) so it never tries to push data.
        _system_config={
            "metrics_report_interval_ms": 1000000000,
            "task_events_report_interval_ms": 1000000000
        },
        runtime_env={
            "env_vars": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "MALLOC_ARENA_MAX": "2",
                "RAY_DEDUP_LOGS": "0",
                "RAY_USAGE_STATS_ENABLED": "0",
                "RAY_DISABLE_DASHBOARD_LOG_INFO": "1"
            }
        }
    )
    
    config = (
        PPOConfig()
        .environment(env="dojo_env", env_config={})
        .callbacks(MetricsCallback)
        .api_stack(
            enable_rl_module_and_learner=False, 
            enable_env_runner_and_connector_v2=False
        )
        # PERFORMANCE & SCALING RECOMMENDATIONS:
        # See "CONFIGURATION & SCALING" section above.
        .env_runners(
            num_env_runners=NUM_PARALLEL_WORKERS, 
            num_envs_per_env_runner=ENVS_PER_WORKER,
            sample_timeout_s=600,
            rollout_fragment_length="auto"
        )
        .resources(num_gpus=1)
        .training(
            model={
                "custom_model": "pokemon_transformer",
                # "vf_share_layers": True, # Handled internally
            },
            gamma=0.99,
            lr=1e-4,
            train_batch_size=8000, 
            minibatch_size=512,
            num_epochs=10,
            entropy_coeff=0.01,
            clip_param=0.2,
        )
        .framework("torch")
    )
    
    # Custom Logger to save to a specific directory (cleaner than ~/ray_results)
    from ray.tune.logger import UnifiedLogger
    import datetime
    
    def custom_logger_creator(config):
        # Create a unique subdir for this run
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.expanduser(f"~/ray_results_dojo/PPO_dojo_{timestamp}")
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir)

    algo = config.build(logger_creator=custom_logger_creator)

    if CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"Restoring checkpoint from {CHECKPOINT_PATH}...", flush=True)
        try:
            algo.restore(CHECKPOINT_PATH)
            print("Checkpoint restored successfully!", flush=True)
        except Exception as e:
             print(f"Failed to restore checkpoint: {e}", flush=True)
    
    print("Training Started...", flush=True)
    import time
    start_time = time.time()
    
    # Training Loop (Increased to 10000 for longer run)
    try:
        for i in range(10000):
            iter_start = time.time()
            result = algo.train()
            iter_dur = time.time() - iter_start
            
            # Calculate throughput
            total_steps = result.get("timesteps_total", 0)
            episode_reward = result.get('env_runners', {}).get('episode_reward_mean', 0.0)
            episode_len = result.get('env_runners', {}).get('episode_len_mean', 0.0)
            
            print(f"Iter {i}: reward={episode_reward:.2f} len={episode_len:.2f} "
                  f"dur={iter_dur:.2f}s total_steps={total_steps}", flush=True)
            
            if i % 5 == 0:
                checkpoint_dir = algo.save("/home/tristan/CodingProjects/ai_pokemon/models/ray_dojo_perfect_info")
                print(f"Saved checkpoint to {checkpoint_dir}", flush=True)
                
    except KeyboardInterrupt:
        print("\nStopping training (Ctrl+C detected)...", flush=True)
        # Optional: Save a final checkpoint on exit
        checkpoint_dir = algo.save("/home/tristan/CodingProjects/ai_pokemon/models/ray_dojo_perfect_info")
        print(f"Saved final checkpoint to {checkpoint_dir}", flush=True)
        
    finally:
        print("Shutting down Ray...", flush=True)
        ray.shutdown()
        print("Ray shutdown complete.", flush=True)
