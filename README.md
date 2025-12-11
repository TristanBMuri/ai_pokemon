# Nuzlocke Gauntlet RL: Project "Triumvirate"

**Hierarchical Reinforcement Learning for Pokémon Nuzlocke Gauntlets**

Current Status: **Active Development**

This project trains a system of reinforcement learning agents to play Pokémon Nuzlocke gauntlets. It uses a hierarchical "Triumvirate" architecture to separate high-level strategy (team building, risk management) from low-level tactics (battling).

## 1. Core Tech Stack

-   **Language**: Python 3.13 (Strict)
-   **Package Manager**: `uv`
-   **Framework**: Ray RLLib (Distributed RL)
-   **Simulation**: poke-env + Local Pokemon Showdown (Docker/Node)
-   **Hardware Optimization**: Linux KSM (Kernel Same-page Merging) + Ray Object Store (Plasma) for memory efficiency.

## 2. The Architecture: "The Triumvirate"

The system is composed of three distinct policies working in unison:

### A. The Strategist (High-Level Policy)
*   **Role**: Campaign Manager & Risk Assessor.
*   **Responsibilities**:
    *   Navigates the Gauntlet Map (Mandatory Gyms vs. Optional Events).
    *   Manages the Inventory and Items.
    *   Decides when to rebuild the team.
    *   Outputs a **Risk Vector** to guide the Battler (e.g., "Play safe" vs. "Sacrifice for the win").
    *   **Reward**: Progress through Gauntlet + Item Value - Death Cost.

### B. The Team Builder (Mid-Level Policy)
*   **Role**: Roster Construction.
*   **Type**: Autoregressive Transformer (Sequential Builder).
*   **Responsibilities**:
    *   Constructs the optimal 6-Pokémon party from the "Box" (roster of alive Pokémon).
    *   Selects Species, Moves, and Items for each slot.
    *   Adapts to the upcoming enemy preview and the Strategist's Risk Vector.

### C. The Battler (Low-Level Policy)
*   **Role**: Tactical Execution.
*   **Type**: LSTM-PPO (Recurrent).
*   **Responsibilities**:
    *   Executes the actual battle moves in Pokémon Showdown.
    *   Respects the Risk Vector (e.g., switching out VIPs if requested).
    *   **Stall Policy**: PP Stalling is allowed; supports turn limits (200 turns).

## 3. Data & Training

-   **Sources**: Smogon Chaos Data (Download Once Rule).
-   **Hall of Fame**: `data/hall_of_fame.json` stores winning teams to train the Builder against.
-   **Curriculum**:
    *   **The Dojo**: Battler trains to imitate/beat SimpleHeuristicPlayer.
    *   **The Gauntlet**: Tiers progressing from PU/NU -> UU/RU -> OU.
    *   **The Underdog Test**: Final validation (e.g., beating OU teams with a UU box).

## 4. Usage

For detailed instructions on setting up the environment, training agents, and running visualizations, please refer to the **[Usage Guide](README_USAGE.md)**.

### Quick Start
All commands should be run using `uv`.

```bash
# Install dependencies
uv sync

# Set up Showdown
./setup_showdown.sh

# Start Showdown (in separate terminal)
cd pokemon-showdown && node pokemon-showdown 8000
```
