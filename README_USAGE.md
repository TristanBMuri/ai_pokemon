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
tensorboard --logdir ./tmp/manager/
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
