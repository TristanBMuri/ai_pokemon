# Nuzlocke Gauntlet RL - Usage Guide

This guide provides step-by-step instructions for setting up, training, and visualizing the Nuzlocke Reinforcement Learning agents.

## 1. Prerequisites & Installation

### System Requirements
- **Python 3.10+**
- **Node.js 18+** (for Pokemon Showdown)
- **Linux/MacOS** (Windows requires WSL2)

### Install Dependencies
We use `uv` for fast Python package management.

```bash
# Install dependencies
uv sync
```

### Setup Local Showdown Server
The agents require a local instance of Pokemon Showdown to simulate battles.

```bash
# Run the setup script (clones repo and installs node modules)
./setup_showdown.sh
```

**Important**: Ensure `config/config.js` in `pokemon-showdown` has `exports.noguestsecurity = true;`. The setup script should handle this, but verify if you encounter login issues.

## 2. Running the Environment

### Start Pokemon Showdown
Before running any training, you must start the Showdown server in a separate terminal.

```bash
cd pokemon-showdown
node pokemon-showdown 8000
```
*Keep this terminal open.*

## 3. Training the Agents

### Step A: Train the Battle Agent (Low-Level Policy)
The Battle Agent learns how to battle with a given team and risk profile.
**New in v0.6**: Uses `RecurrentPPO` (LSTM) for memory-based tactics.

```bash
# Train for 10,000 steps (adjust as needed)
uv run python train_battle_agent.py --steps 10000 --model ppo_risk_agent_lstm_v1
```

### Step B: Train the Manager Agent (High-Level Policy)
The Manager Agent learns to build teams from the roster to defeat the Gauntlet.
**New in v0.7**: Uses "Complete Overview" observation (Stats, Moves, Abilities).
**New in v0.8**: Logs detailed metrics (Win Rate, K/D) to TensorBoard.

```bash
# Train with 8 parallel environments for stability
uv run python train_manager.py --steps 100000 --n_envs 8 --model_name ppo_manager_v4
```

## 4. Visualization & Interpretation (v0.8)

### Method 1: Generate Training Plots
We provide a script to generate clear graphs of Win Rate, Battle Duration, and K/D Ratio.

```bash
# Generate 'training_plot.png' from the latest logs
uv run python plot_training.py --log_dir ./tmp/manager/ --output training_plot.png
```

### Method 2: TensorBoard
For real-time monitoring and deep dives.

```bash
# Start TensorBoard
uv run tensorboard --logdir ./tmp/manager/
```
Open `http://localhost:6006` in your browser. Look for the **Custom** tab to see:
- `custom/win_rate`: Rolling average win rate.
- `custom/avg_turns`: Average turns per battle.
- `custom/pokemon_fainted`: Your deaths per battle.
- `custom/trainer_idx`: Max trainer reached (Progress Tracking).

## 5. Watch Mode (Live Battles)

To watch the best agent fight in real-time on your local Showdown server:

```bash
# Runs 1 episode and prints Battle URLs
uv run python watch_agent.py --model_name ppo_manager_v4 --gauntlet extended
```
Click the `http://localhost:8000/...` link when it appears to open the battle.

## 6. Evaluation

To test the trained Manager against the Gauntlet without training noise:

```bash
# Run 10 evaluation episodes
uv run python evaluate_manager.py --model_name ppo_manager_v4 --episodes 10
```

## Troubleshooting

- **"Connection Refused"**: Ensure `node pokemon-showdown 8000` is running.
- **"KeyError: calyrexi"**: The roster filter should handle this, but if new invalid species appear, check `nuzlocke_env.py` filtering logic.
- **"sb3-contrib not found"**: Run `uv sync` to install the new dependencies for LSTM.

## 7. Ray Distributed Training (Dojo) - v0.9

To scale training massively, we use Ray RLlib with a "Dojo" setup where agents fight diverse opponents across multiple tiers.

### Features
- **Multi-Server Support**: Automatically spawns local Showdown instances on ports 8000+ for each worker.
- **Curriculum Learning**: Opponent difficulty adapts to agent win rate.
    - < 40% WR: **Simple Heuristics** (Easy)
    - 40-80% WR: Mixed
    - > 80% WR: **Radical Red AI** (Hard/Competitive, uses switch prediction and damage calc)
- **Multi-Tier**: Trains on Gen 9 OU, Ubers, UU, RU, NU, PU, LC, and Gen 8-6 OU randomly.
- **Memory Optimized**: Uses lightweight logic engines to support 30+ parallel workers on 64GB RAM.
- **Full Team & Perfect Info Architecture**: Agent receives deep knowledge of the battle state:
    - **Opponent**: Moves (Power/Acc/Eff), Items, Abilities, Relative Stats.
    - **Teammates (NEW)**: Full visibility into the **Bench** (Moves, Items, Abilities, **Types**) for all 6 team members.
    - **Smart Switching**: Actions 16-21 now reliably map to Team Slots 1-6, allowing precise team management.
    - **Total**: **583 Input Features** (up from 121) processed by a **512-unit LSTM** model.

### Running the Dojo
This script handles everything (server spawning, training, logging).

```bash
# Run with defaults (20 workers, 1 envs/worker)
# Note: Ensure you have 'gputil' installed if using GPU monitoring
uv run python train_dojo_ray.py
```

**Note**: Checkpoints are saved to `models/ray_dojo_perfect_info`.
**Important**: TensorBoard logs are now saved to a **clean directory** to avoid clutter: `~/ray_results_dojo/`.

### TensorBoard Metrics
The Dojo logs detailed metrics. Run:

```bash
uv run tensorboard --logdir ~/ray_results_dojo/
```

**Key Metrics to Watch:**
- `custom_metrics/agent_win_rate_mean`: Global win rate.
- `custom_metrics/opponent_appearance_Simple_mean`: % of battles against Easy AI.
- `custom_metrics/opponent_appearance_Radical_mean`: % of battles against Hard AI.
    - *Goal*: You want "Radical" appearance to increase as the agent gets better.
- `episode_reward_mean`: Approx 1.0 = Win, -1.0 = Loss (differs by shaping).
- `ray/tune/perf/ram_util_percent`: Monitor system memory stability.
