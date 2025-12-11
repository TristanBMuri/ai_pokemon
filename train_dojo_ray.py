
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from nuzlocke_gauntlet_rl.envs.threaded_ray_env import ThreadedBattleEnv
from nuzlocke_gauntlet_rl.data.smogon import SmogonDataFetcher
from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.teambuilder import ConstantTeambuilder
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
NUM_PARALLEL_WORKERS = 30

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
CHECKPOINT_PATH = "/home/tristan/CodingProjects/ai_pokemon/models/ray_dojo"
# ==============================================================================

def env_creator(config):
    worker_index = config.worker_index if hasattr(config, "worker_index") else 0
    # Port Distribution
    port = 8000 + (worker_index % 4)
    print(f"DEBUG: env_creator worker={worker_index} port={port}", flush=True)

    server_url = f"ws://localhost:{port}/showdown/websocket"
    server_config = ServerConfiguration(server_url, None)
    
    # Smogon Data
    smogon = SmogonDataFetcher()
    agent_team_str = smogon.generate_team()
    
    # Opponent Setup
    # ThreadedBattleEnv creates the opponent internally on the background thread
    opponent_cls = SimpleHeuristicsPlayer
    
    # We define a custom Opponent class to inject the team
    class DojoOpponent(SimpleHeuristicsPlayer):
         def __init__(self, **kwargs):
             super().__init__(**kwargs)
             self._team = ConstantTeambuilder(smogon.generate_team())
             
    opponent_config = AccountConfiguration("Placeholder", None) # Name set dynamically in Env
    
    # Agent Config
    agent_config = AccountConfiguration("Placeholder", None) # Name set dynamically
    agent_team = ConstantTeambuilder(agent_team_str) 
    
    return ThreadedBattleEnv(
        agent_config=agent_config,
        agent_team=agent_team,
        opponent_cls=DojoOpponent,
        opponent_config=opponent_config,
        server_configuration=server_config
    )

register_env("dojo_env", env_creator)

if __name__ == "__main__":
    print("STARTING THREADED RAY DOJO...", flush=True)
    
    # Using default Ray init (auto-detect resources)
    # Disable dashboard to prevent metrics buffering leak (connection errors seen in logs)
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    
    algo = (
        PPOConfig()
        .environment(env="dojo_env", env_config={})
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
                "use_lstm": True,
                "lstm_cell_size": 128, # Compact model
                "fcnet_hiddens": [128, 128],
                "max_seq_len": 20,
            },
            train_batch_size=2048, # 2048 total steps across 4 workers
            minibatch_size=256,
            num_epochs=5, # Renamed from num_sgd_iter as per warning
            lr=5e-5,
            gamma=0.99,
        )
        .framework("torch")
        .build()
    )

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
                checkpoint_dir = algo.save("/home/tristan/CodingProjects/ai_pokemon/models/ray_dojo")
                print(f"Saved checkpoint to {checkpoint_dir}", flush=True)
                
    except KeyboardInterrupt:
        print("\nStopping training (Ctrl+C detected)...", flush=True)
        # Optional: Save a final checkpoint on exit
        checkpoint_dir = algo.save("/home/tristan/CodingProjects/ai_pokemon/models/ray_dojo")
        print(f"Saved final checkpoint to {checkpoint_dir}", flush=True)
        
    finally:
        print("Shutting down Ray...", flush=True)
        ray.shutdown()
        print("Ray shutdown complete.", flush=True)
