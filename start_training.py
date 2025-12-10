import os
import sys
import subprocess
import time
import shutil
import socket
import argparse

# ==========================================
#        HYPERPARAMETER CONFIGURATION
# ==========================================
MODEL_NAME = "ppo_nuzlocke_natdex_v1" # Updated for NatDex format
GAUNTLET_NAME = "complete"      # Gauntlet to run: "complete", "kanto_leaders", etc.
TOTAL_STEPS = 100_000           # Total training steps
N_ENVS = 24                     # Increased to 24 to saturate CPU/GPU better (Showdown is IO bound)
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

def start_showdown(start_port=8000, n_servers=4):
    processes = []
    
    # 1. Try to Autostart if Node exists
    node_cmd = shutil.which("node") or os.environ.get("NODE_PATH")
    
    if node_cmd:
        print(f"✅ Found Node.js at {node_cmd}")
        for i in range(n_servers):
            port = start_port + i
            if is_port_open(port):
                print(f"✅ Port {port} is active.")
                continue
                
            print(f"🚀 Launching server on {port}...")
            try:
                log_file = open(f"showdown_server_{port}.log", "w")
                proc = subprocess.Popen(
                    [node_cmd, "pokemon-showdown", str(port)], 
                    cwd=SHOWDOWN_DIR,
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
                processes.append(proc)
            except Exception as e:
                print(f"❌ Failed to start server on {port}: {e}")
    else:
        print("⚠️  Node.js not found. Skipping auto-start.")
        print(f"ℹ️  Please ensure {n_servers} Showdown servers are running on ports {start_port}-{start_port + n_servers - 1}")

    # 2. Wait for Ports to be Ready
    print("Waiting for servers to be ready...")
    ready = False
    while not ready:
        open_count = 0
        for i in range(n_servers):
            port = start_port + i
            if is_port_open(port):
                open_count += 1
        
        if open_count == n_servers:
            ready = True
            print("✅ All servers ready!")
        else:
            print(f"⏳ Waiting... ({open_count}/{n_servers} ready). Start them manually!")
            time.sleep(3)
        
    return processes

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
    server_proc_list = start_showdown(n_servers=4)
    
    try:
        # Run Training
        run_training()
    finally:
        # Cleanup
        if server_proc_list:
            print("Stopping Showdown Servers...")
            for p in server_proc_list:
                p.terminate()
                p.wait()
