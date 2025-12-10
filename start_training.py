import os
import sys
import subprocess
import time
import socket
import argparse

# ==========================================
#        HYPERPARAMETER CONFIGURATION
# ==========================================
MODEL_NAME = "ppo_nuzlocke_natdex_v1" # Updated for NatDex format
GAUNTLET_NAME = "complete"      # Gauntlet to run: "complete", "kanto_leaders", etc.
TOTAL_STEPS = 100_000           # Total training steps
N_ENVS = 1                      # Number of parallel environments (1 for safer Dashboard/Showdown)
N_STEPS = 2048                  # Steps per update (Buffer size)
BATCH_SIZE = 64                 # Minibatch size
LEARNING_RATE = 0.0003          # Learning Rate (PPO default: 3e-4)
ENT_COEF = 0.0                  # Entropy Coefficient (Increase to encourage exploration)
GAMMA = 0.99                    # Discount Factor

# ==========================================
#             SERVER CONFIG
# ==========================================
SHOWDOWN_port = 8000
SHOWDOWN_DIR = "./pokemon-showdown" # Path to local showdown folder

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_showdown():
    if is_port_open(SHOWDOWN_port):
        print(f"✅ Pokemon Showdown is already running on port {SHOWDOWN_port}.")
        return None

    print(f"⚠️  Pokemon Showdown not found on port {SHOWDOWN_port}.")
    
    if not os.path.exists(SHOWDOWN_DIR):
        print(f"❌ Error: Could not find Showdown directory at {SHOWDOWN_DIR}")
        print("Please configure SHOWDOWN_DIR in start_training.py or start it manually.")
        sys.exit(1)
        
    print(f"🚀 Starting Pokemon Showdown from {SHOWDOWN_DIR}...")
    try:
        # Start in a separate process
        # We use 'node pokemon-showdown' command
        log_file = open("showdown_server.log", "w")
        proc = subprocess.Popen(
            ["node", "pokemon-showdown"], 
            cwd=SHOWDOWN_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT
        )
        print("Waiting for server to initialize...", end="", flush=True)
        for _ in range(10):
            time.sleep(1)
            if is_port_open(SHOWDOWN_port):
                print(" Done!")
                return proc
            print(".", end="", flush=True)
            
        print("\n❌ Failed to detect server on port 8000. Check showdown_server.log.")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: 'node' command not found. Do you have Node.js installed?")
        sys.exit(1)

def run_training():
    # Construct command
    # We explicitly use 'uv run' to ensure we use the virtual environment
    cmd = [
        "uv", "run", "python", "train_manager.py",
        "--model_name", MODEL_NAME,
        "--gauntlet", GAUNTLET_NAME,
        "--steps", str(TOTAL_STEPS),
        "--n_envs", str(N_ENVS),
        "--n_steps_per_update", str(N_STEPS),
        "--learning_rate", str(LEARNING_RATE),
        "--batch_size", str(BATCH_SIZE),
        "--ent_coef", str(ENT_COEF),
    ]
    
    print("\n========================================")
    print("      Reforged Nuzlocke Trainer")
    print("========================================")
    print(f"Gauntlet: {GAUNTLET_NAME}")
    print(f"Steps:    {TOTAL_STEPS}")
    print("========================================\n")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")

if __name__ == "__main__":
    # Check Server
    server_proc = start_showdown()
    
    try:
        # Run Training
        run_training()
    finally:
        # Cleanup
        if server_proc:
            print("Stopping Showdown Server...")
            server_proc.terminate()
            server_proc.wait()
