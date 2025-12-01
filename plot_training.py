import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_metrics(log_dir: str, output_file: str = "training_plot.png"):
    """
    Reads TensorBoard logs and plots key metrics.
    """
    # Find all event files
    event_files = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))
                
    if not event_files:
        print(f"No event files found in {log_dir}")
        return
        
    print(f"Found {len(event_files)} event files. Parsing...")
    
    # Data containers
    steps = []
    rewards = []
    win_rates = []
    avg_turns = []
    
    for event_file in event_files:
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        # Extract scalars
        if "rollout/ep_rew_mean" in ea.Tags()["scalars"]:
            for e in ea.Scalars("rollout/ep_rew_mean"):
                steps.append(e.step)
                rewards.append(e.value)
                
        if "custom/win_rate" in ea.Tags()["scalars"]:
            # We might need to align steps if they differ
            # For simplicity, let's just plot what we have
            pass
            
    # Better approach: Use a DataFrame for each metric and merge/plot
    # Since we might have multiple runs or restarts, plotting them all might be messy.
    # Let's focus on the most recent run or aggregate.
    
    # Actually, let's just plot the raw data points we find
    data = {}
    
    for event_file in event_files:
        ea = EventAccumulator(event_file)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        
        for tag in ["rollout/ep_rew_mean", "custom/win_rate", "custom/avg_turns", "custom/pokemon_fainted", "custom/opponent_fainted"]:
            if tag in tags:
                if tag not in data: data[tag] = {"steps": [], "values": []}
                for e in ea.Scalars(tag):
                    data[tag]["steps"].append(e.step)
                    data[tag]["values"].append(e.value)
                    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # 1. Reward & Win Rate
    ax1 = axes[0]
    if "rollout/ep_rew_mean" in data:
        ax1.plot(data["rollout/ep_rew_mean"]["steps"], data["rollout/ep_rew_mean"]["values"], label="Mean Reward", color="blue")
    if "custom/win_rate" in data:
        ax1_twin = ax1.twinx()
        ax1_twin.plot(data["custom/win_rate"]["steps"], data["custom/win_rate"]["values"], label="Win Rate", color="green", linestyle="--")
        ax1_twin.set_ylabel("Win Rate")
        ax1_twin.legend(loc="upper right")
        
    ax1.set_title("Training Progress: Reward & Win Rate")
    ax1.set_ylabel("Mean Reward")
    ax1.legend(loc="upper left")
    ax1.grid(True)
    
    # 2. Battle Length
    ax2 = axes[1]
    if "custom/avg_turns" in data:
        ax2.plot(data["custom/avg_turns"]["steps"], data["custom/avg_turns"]["values"], label="Avg Turns", color="orange")
    ax2.set_title("Battle Efficiency: Average Turns")
    ax2.set_ylabel("Turns")
    ax2.legend()
    ax2.grid(True)
    
    # 3. K/D Ratio (Fainted)
    ax3 = axes[2]
    if "custom/pokemon_fainted" in data:
        ax3.scatter(data["custom/pokemon_fainted"]["steps"], data["custom/pokemon_fainted"]["values"], label="My Deaths", color="red", alpha=0.3, s=10)
    if "custom/opponent_fainted" in data:
        ax3.scatter(data["custom/opponent_fainted"]["steps"], data["custom/opponent_fainted"]["values"], label="Opponent Deaths", color="purple", alpha=0.3, s=10)
        
    ax3.set_title("Attrition: Pokemon Fainted per Battle")
    ax3.set_ylabel("Count")
    ax3.set_xlabel("Timesteps")
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./tmp/manager/", help="Path to TensorBoard logs")
    parser.add_argument("--output", type=str, default="training_plot.png", help="Output image file")
    args = parser.parse_args()
    
    plot_metrics(args.log_dir, args.output)
