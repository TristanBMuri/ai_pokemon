Nuzlocke Gauntlet RL

Train a reinforcement learning agent to beat Pokémon Nuzlocke-style gauntlets built on top of Pokémon Showdown + poke-env, without needing to interact with an actual ROM.

The long-term goal:
An agent that can reliably survive a distribution of hard, Nuzlocke-like gauntlets (inspired by ROM hacks), making good battle decisions and run-level strategic decisions (team management, encounters, move choices, and resource conservation).

High-Level Architecture: Manager-Operator Model

We split the problem into two distinct, hierarchical agents to avoid credit assignment issues:

1. The Nuzlocke Manager (High-Level Policy)

The "Brain" of the run. It does not play the battles; it prepares the team and sets the strategy.

Input: Full Roster (Box + Party), Inventory, Next Opponent Preview.

Responsibilities:

Roster Ops: Selecting the optimal 6-mon party from the box for the specific upcoming fight.

Resource Ops: Managing movesets (TMs/Move Tutors) and items.

Encounter Ops: Deciding whether to catch or skip new encounters ("Use it or lose it").

Risk Command: Issuing a Risk Token (e.g., Safe, Balanced, Desperate) to the Battle Agent.

2. The Battle Agent (Low-Level Policy)

The "Hands" of the run. It executes the battle based on the Manager's instructions.

Input: The Active Party, the Opponent, and the Risk Token.

Responsibilities:

Executes moves and switches in the battle engine (Pokemon Showdown).

Adapts playstyle based on the Risk Token (e.g., if Risk=Safe, avoid recoil moves or risky predictions; if Risk=Desperate, sacrifice mons to secure a win).

Gauntlet Structure & Rules

A run consists of a sequence of trainers and route encounters.

Battles are simulated via Pokémon Showdown accessed from Python via poke-env.

Nuzlocke Rules:

Permadeath: Fainted Pokémon are permanently removed from the roster.

Limited Encounters: Route encounters are stochastic and finite.

No Backtracking: Encounters are "use it or lose it" to keep the state space manageable.

Planned Features / Components

1. Gauntlet & World Specification

PokemonSpec: Species, movesets, items, abilities.

TrainerSpec: Fixed opponent teams.

RouteSpec: Encounter tables with probabilities.

GauntletSpec: An ordered list of Trainers and Routes.

2. The Manager's Domain (Run-Level Env)

The NuzlockeGauntletEnv cycles through these phases:

PRE_BATTLE (The Core Strategic Phase)

Observation: Full roster state (HP, status, moves), next opponent intel.

Action:

Select 6 mons for the party.

Assign a Risk Token (e.g., 0=Conservative, 1=Neutral, 2=Aggressive).

Transition: Passes control to the Battle Agent/Simulator.

MOVE_SELECTION (Resource Phase)

Triggered when a mon learns a new move (Level-up or TM).

Action: Replace an existing move or skip learning.

Crucial for long-term strategy (e.g., saving a coverage move for a specific boss).

ENCOUNTER (Growth Phase)

Triggered after unlocking a route.

Action: Catch (add to box/party) or Skip.

Note: If box is full, may involve releasing a mon.

3. The Battle Engine

Simulator: Local Pokémon Showdown server.

Interface: simulate_battle(my_team, enemy_team, risk_token) -> Result

Battle Policy Training:

Trained separately on BattleEnv.

Conditioned on the risk_token to alter its value function (e.g., penalizing deaths more heavily when risk_token=Safe).

Reward Structure

Manager Rewards

Survival Reward: Large positive reward for clearing the gauntlet.

Progress Reward: Reward for each trainer defeated.

Death Penalty: Significant negative reward for every permanent death (scaled by mon quality/importance).

Battle Agent Rewards

Win/Loss: Standard binary reward.

Risk Alignment:

If Risk=Safe: Extra penalty for taking damage or fainting.

If Risk=Desperate: Purely focuses on winning, ignoring HP costs.

Roadmap

v0.1 – Toy Manager (No Real Battles)

Implement GauntletSpec and Roster logic.

Mock simulate_battle (returns random results based on team strength).

Train Manager to optimize encounters and party selection.

v0.2 – Battle Agent & Risk Integration (Complete)

[x] Set up local Showdown + poke-env.

[x] Train Battle Agent with risk_token input.

[x] Verify Battle Agent plays differently when asked to be "Safe" vs "Risky".

v0.3 – Integration

Connect Manager and Battle Agent.

Full run simulation: Manager picks team + risk -> Battler executes -> Deaths reported back.

v0.4 – Advanced Features

Moveset Management: Enable MOVE_SELECTION phase.

Complex Gauntlets: Import data from hard ROM hacks (e.g., Emerald Kaizo-lite).

Repository Structure

.
├── nuzlocke_gauntlet_rl/
│   ├── envs/
│   │   ├── nuzlocke_env.py       # The Manager Environment
│   │   └── battle_env.py         # The Battle Agent Environment
│   ├── agents/
│   │   ├── manager.py            # High-level policy (PPO/DQN)
│   │   └── battler.py            # Low-level policy (PPO/DQN)
│   └── data/                     # Gauntlet definitions (JSON/YAML)
└── scripts/
    ├── train_manager.py
    ├── train_battler.py
    └── evaluate_risk.py

Usage

1. Set up Local Showdown Server
   ./setup_showdown.sh
   # Important: Ensure 'exports.noguestsecurity = true;' is in pokemon-showdown/config/config.js
   node pokemon-showdown 8000

2. Train Battle Agent
   python train_battler.py
   # Saves model to models/ppo_risk_agent

3. Evaluate Risk Behavior
   python evaluate_risk.py
   # Runs battles with Safe vs Desperate risk tokens and reports win rates/deaths.

