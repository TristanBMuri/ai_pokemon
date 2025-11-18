# Nuzlocke Gauntlet RL

Train a reinforcement learning agent to beat **Pokémon Nuzlocke-style gauntlets** built on top of **Pokémon Showdown + poke-env**, without needing to interact with an actual ROM.

The long-term goal:
An agent that can reliably survive a distribution of hard, Nuzlocke-like gauntlets (inspired by ROM hacks), making good **battle decisions** and **run-level strategic decisions** (team management, encounters, move choices, and resource conservation).

---

## High-Level Idea

Instead of playing through a full game via emulator, we model a **Nuzlocke run as a structured gauntlet**:

* A run consists of a sequence of **trainers** and **route encounters**.
* Battles are simulated via **Pokémon Showdown** accessed from Python via **poke-env**.
* Nuzlocke-style rules apply:

  * Fainted Pokémon are **permanently dead** for that run.
  * Route encounters are **limited** and **stochastic**.
  * The agent manages both a **party** and a **box** (global roster of alive mons), switching between them over the run.
* The agent’s reward depends on:

  * How far it gets through the gauntlet.
  * How many of its Pokémon survive.
  * How well it manages movesets and long-term risk.

We explicitly split the problem into two policies:

1. **Battle policy**
   Learns to play *individual battles* optimally given a fixed team and opponent.

2. **Run-level policy (Nuzlocke policy)**
   Learns *long-horizon strategy*:

   * Which Pokémon to bring into each fight (party selection from box).
   * What to do with encounters (catch/skip/replace).
   * How to manage movesets when new moves become available.
   * How to conserve key resources (strong mons, good coverage) over the whole run.

Episodes are sampled from a **distribution of gauntlets**, not a single fixed one, to capture Nuzlocke-style variance.

---

## Planned Features / Components

### 1. Gauntlet & World Specification

Planned data structures:

* `PokemonSpec`

  * Species, base level, default moveset, item, ability.
* `TrainerSpec`

  * Trainer name and fixed team (`List[PokemonSpec]`).
* `RouteSpec`

  * Route name and encounter table: list of `(PokemonSpec, probability)`.
* `GauntletSpec`

  * Ordered list of trainers.
  * Set of routes and **unlock conditions** (e.g. "Route 2 unlocks after Trainer 3").

On top of that, a **run-level instance** uses `MonInstance` objects:

* `MonInstance` (per-run Pokémon instance)

  * Unique ID within the run.
  * Species, current level, current moves, item, ability.
  * `alive: bool` (permadeath flag).
  * `in_party: bool` (party vs box flag).

Additional planned features:

* A small library of **example gauntlets**:

  * Toy gauntlets for quick debugging (2–3 trainers, tiny pool of mons).
  * More complex gauntlets inspired by harder ROM hacks (without needing the actual ROM).
* Support for **multiple gauntlets** and sampling one per episode:

  * Basic distribution = uniform over a list of gauntlets.
  * Later: configurable distributions (difficulty tiers, seeds, etc.).
* Config files (YAML/JSON) to define trainers, routes, gauntlets and available builds/moves without editing Python code.

---

### 2. Box, Party & Mon Management

We distinguish between:

* **Roster** = all alive `MonInstance`s in the run (party + box).
* **Party** = up to 6 mons marked with `in_party = True`.
* **Box** = alive mons with `in_party = False`.

The run-level policy interacts with these via phases:

* At **pre-battle time**, the agent chooses which subset of the **alive roster** to bring into the next fight.

  * This implicitly defines party vs box for that battle.
  * Later we can optionally constrain this to only change at special "PC" moments.
* On **encounters**, the agent can:

  * Skip the encounter.
  * Catch and place into the roster (party or box) if there is space.
  * Catch and replace an existing mon (e.g. release or box an old one).

Perma-death is enforced by the environment:

* After each battle, any mon that fainted gets `alive = False` and is removed from the usable roster for the rest of the run.

This gives the run-level policy control over **which mons to risk when**, and how to manage the roster as a scarce resource.

---

### 3. Movesets, Levels & TMs (Simplified but Explicit)

Moves and move progression are a big part of Nuzlocke difficulty, but the full legal move space is huge. To keep the problem tractable, we plan a **layered approach**.

#### 3.1. Stage 0 – Fixed Movesets (Bootstrap)

* Each `PokemonSpec` has a **fixed default moveset**.
* No level-ups or move changes.
* Used for the first prototypes to:

  * Implement battle policy and Nuzlocke env.
  * Debug reward shaping, death handling, encounters, etc.

#### 3.2. Stage 1 – Predefined Builds per Species

* For each species (or role), define a **small set of builds**:

  ```python
  class BuildSpec:
      name: str
      level: int
      moves: list[str]
  ```

* When a new mon is obtained, the environment either:

  * assigns a default build,
  * or lets the agent **choose among a few builds** (e.g. "bulky", "offensive", "support").

This gives the agent some control over movesets without exploding the combinatorial space.

#### 3.3. Stage 2 – Move-Learn Events (Level-Up / TMs)

Later, we add **discrete move events** that mimic level-ups or TMs:

* For each species, predefine a small list of important additional moves that become available at certain gauntlet milestones (e.g. after certain trainers or stages).

* Model each move opportunity as a `MoveEvent`:

  ```python
  class MoveEvent:
      mon_id: int
      move_name: str
      source: Literal["level_up", "tm"]
  ```

* When a `MoveEvent` is triggered, the env enters a `MOVE_SELECTION` phase:

  * Observation includes: the mon, its current moves, and the candidate move.
  * Action is a small discrete choice:

    * `0` = skip / do not learn.
    * `1..4` = replace that move slot with the new move.

We do **not** initially model full TM inventories, shops, or money. Instead, we approximate the hack by tying move availability to gauntlet progression.

If desired later, we can:

* Add explicit levels and XP per mon.
* Trigger `MoveEvent`s when level thresholds are reached.
* Introduce more detailed TM systems.

---

### 4. Battle Engine & Battle Policy

We use **Pokémon Showdown** (local server) and **poke-env** as the battle engine.

Planned:

* Run a local Showdown server with a custom or standard format (e.g. `genXgauntlet-singles`).

* Use **poke-env** to:

  * Connect to the Showdown server.
  * Represent battles as Gymnasium-compatible environments (`BattleEnv`).

* Define a `simulate_battle(...)` interface used by the run-level env:

  ```python
  def simulate_battle(my_team, enemy_team) -> tuple[bool, list[bool]]:
      """Simulate a battle via poke-env.

      Returns:
        win: True if we win, False otherwise.
        alive_mask: len(my_team) booleans, indicating which mons survived.
      """
  ```

* **Battle policy**:

  * Trained separately using `BattleEnv` and standard RL algorithms (PPO/DQN/etc.).
  * Learns move/switch decisions given battle state.
  * Later integrated into `simulate_battle` so the run-level policy sees realistic outcomes.

Future / stretch:

* Showdown **mods** approximating ROM-hack mechanics:

  * Adjusted stats/movesets/abilities.
  * Custom data-based modifications for hack-inspired formats.

---

### 5. Run-Level Environment: `NuzlockeGauntletEnv`

The **NuzlockeGauntletEnv** is a Gymnasium environment modeling an entire Nuzlocke-style run.

#### 5.1. Phases

Internally, the environment cycles through several phases:

* `PRE_BATTLE`
  Agent selects which mons from the alive roster are used in the next fight (party selection).

* `ENCOUNTER`
  A route has unlocked, an encounter is sampled, and the agent decides whether and how to add it to the roster.

* `MOVE_SELECTION`
  A move-learn event is available; agent decides whether to learn it and which move to replace.

* `DONE`
  Run finished (either cleared the gauntlet or wiped).

(Optionally later)

* `BOX_MANAGEMENT`
  Dedicated phase to reshape party vs box at PC-like checkpoints.

#### 5.2. Episode Lifecycle

1. **reset()**

   * Sample a `GauntletSpec` from a collection (distribution of gauntlets).
   * Initialize starter roster: a few `MonInstance`s with default builds.
   * Set `trainer_idx = 0` and `phase = PRE_BATTLE`.

2. **PRE_BATTLE phase**

   * Observation encodes:

     * Current alive mons (roster), with party/box membership and summaries.
     * Next trainer index and a summary of their team.
     * Route/gauntlet progression state, pending move events, etc.
   * Action: choose subset of alive mons to bring into this battle.
   * Environment calls `simulate_battle` with the chosen party and trainer team.
   * Apply **permadeath**: update `alive` flags based on `alive_mask` returned.
   * Give reward:

     * +R for beating the trainer.
     * -penalty per mon death.
   * Progress trainer index.
   * If wiped (no alive mons): `phase = DONE`.
   * Else if gauntlet finished: big success reward, `phase = DONE`.
   * Else:

     * Trigger any `MoveEvent`s (push into queue).
     * Check for route unlocks (potential `ENCOUNTER`).
     * Move to `MOVE_SELECTION`, `ENCOUNTER`, or back to `PRE_BATTLE`.

3. **MOVE_SELECTION phase**

   * For each `MoveEvent` in queue:

     * Observation highlights the target mon and candidate move.
     * Action: skip or replace one of up to 4 moves.
     * Update `MonInstance.moves` accordingly.
   * After queue is empty, move to `ENCOUNTER` (if route unlock) or `PRE_BATTLE`/`DONE`.

4. **ENCOUNTER phase**

   * Sample a mon from the unlocked route's encounter table.
   * Observation includes new mon spec and current roster.
   * Action:

     * Skip encounter, or
     * Catch and insert into roster (potentially replacing an existing mon or filling free slot).
   * Then transition to `PRE_BATTLE`/`MOVE_SELECTION`/`DONE` as appropriate.

#### 5.3. Observations & Actions

* **Observation space**:

  * Will initially be a hand-crafted vector / dict encoding:

    * Roster summary (e.g. per-mon typing, level, maybe coarse roles).
    * `trainer_idx` and next trainer type summary.
    * Route/encounter availability flags.
    * Move-selection flags (when relevant).
  * Later we can replace parts with learned embeddings.

* **Action space**:

  * Single `spaces.Discrete(N)` interpreted differently per phase:

    * `PRE_BATTLE`: encode subset selection of mons (party). For early versions, restrict to small rosters or fixed-size parties to keep encoding simple.
    * `ENCOUNTER`: choice to skip/add/replace.
    * `MOVE_SELECTION`: choice among {skip, replace move 1..4}.

---

### 6. Algorithms & Training Pipeline

Planned:

* Use **Stable-Baselines3** or **RLlib** as RL backends.
* Separate training loops:

  1. **Battle policy training** on `BattleEnv` (poke-env), with fixed team pools.
  2. **Run-level policy training** on `NuzlockeGauntletEnv`, calling `simulate_battle`.
* Possible future hierarchical RL:

  * Battle policy as a low-level controller.
  * Nuzlocke policy as high-level controller.
* Curriculum learning:

  * Start with short, easy gauntlets.
  * Gradually increase length, difficulty, and variance.
* Logging & monitoring:

  * Win-rate vs trainers and gauntlets.
  * Run success rate.
  * Average number of deaths per run.
  * Distribution of move choices and encounter outcomes.

---

### 7. Evaluation & Metrics

Planned evaluation metrics:

* **Run success rate** (clearing the full gauntlet).
* **Average trainer depth reached** when failing.
* **Expected survivors** at the end of runs.
* **Deaths per run** and which mons are typically sacrificed.
* **Encounter efficiency** (how often beneficial encounters are taken).
* **Move choice quality** (does the agent converge to strong movesets?).

We will also run ablation studies:

* With vs without move-learning decisions.
* With vs without route encounters.
* Different reward shapings (death penalty magnitude, final reward scaling).
* Fixed vs varied gauntlet distributions.

---

### 8. Approximating ROM Hacks (Future Work)

Longer-term, we may approximate specific ROM hacks:

* Create Showdown **mods** that adjust:

  * Base stats, typings, and movesets.
  * Learnsets and availability of key moves.
* Define gauntlets matching key boss fights from those hacks.
* Approximate route encounter tables from the hacks.

This allows us to study:

* Transfer from "vanilla" gauntlets to "hack-like" ones.
* Robustness under meta shifts (different boss teams, encounter distributions).

---

## Planned Repository Structure

*This is a rough plan; names may change as things get implemented.*

```text
.
├── README.md
├── pyproject.toml / setup.cfg          # Package configuration (planned)
├── nuzlocke_gauntlet_rl/
│   ├── __init__.py
│   ├── config/
│   │   ├── gauntlets/                  # YAML/JSON for gauntlet definitions
│   │   └── builds/                     # Species builds and move event configs
│   ├── data/
│   │   └── examples/                   # Example trainers/routes/gauntlets
│   ├── envs/
│   │   ├── nuzlocke_env.py             # NuzlockeGauntletEnv
│   │   └── battle_env_wrapper.py       # poke-env / Showdown battle wrapper
│   ├── models/
│   │   ├── battle_policy/              # Battle RL policy code
│   │   └── run_policy/                 # Run-level RL policy code
│   ├── showdown/
│   │   ├── server_config/              # Docs / scripts for local Showdown setup
│   │   └── mods/                       # (Future) ROM-hack-like mods
│   └── utils/
│       ├── specs.py                    # PokemonSpec, MonInstance, TrainerSpec, RouteSpec, GauntletSpec
│       ├── observation_builder.py      # Observation construction helpers
│       └── evaluation.py               # Evaluation helpers and metrics
└── scripts/
    ├── train_battle_policy.py          # Train battle policy on BattleEnv
    ├── train_nuzlocke_policy.py        # Train run-level policy on NuzlockeGauntletEnv
    └── run_evaluation.py               # Run evaluation / ablations
```

---

## Roadmap / Milestones

### v0.1 – Toy Nuzlocke Without Real Battles

* [ ] Implement `PokemonSpec`, `MonInstance`, `TrainerSpec`, `RouteSpec`, `GauntletSpec`.
* [ ] Implement a **toy** `NuzlockeGauntletEnv`:

  * Fixed movesets, small roster, no move-learning yet.
  * Tiny toy gauntlet (few trainers, few mons).
* [ ] Implement a stub `simulate_battle` (simple deterministic or random outcome using mon "power" scores).
* [ ] Train a basic PPO (or similar) agent and verify that it learns:

  * To value encounters.
  * To preserve strong mons for later fights.

### v0.2 – Showdown Integration & Battle Policy

* [ ] Set up a local Pokémon Showdown server.
* [ ] Integrate **poke-env** and create a `BattleEnv` with:

  * Fixed format (e.g. gen X singles).
  * Small pool of Pokémon and trainer teams.
* [ ] Train a battle policy (e.g. PPO) to a reasonable level.
* [ ] Implement `simulate_battle` using the trained battle policy and return win/loss + per-mon survival.

### v0.3 – Full Nuzlocke Gauntlet RL (Fixed Movesets)

* [ ] Plug real `simulate_battle` into `NuzlockeGauntletEnv`.
* [ ] Expand gauntlet definitions (more trainers, more routes, more mons).
* [ ] Train a run-level policy on a **distribution of gauntlets**.
* [ ] Add basic evaluation scripts and metrics.

### v0.4 – Builds & Move Events

* [ ] Introduce `BuildSpec` per species and allow build selection on new encounters.
* [ ] Add `MoveEvent`s and `MOVE_SELECTION` phase.
* [ ] Tie move availability to gauntlet progression or approximate level thresholds.
* [ ] Evaluate whether the run-level policy learns sensible move choices.

### v0.5+ – Hack-Like Extensions & Advanced Experiments

* [ ] Design hack-inspired gauntlets with buffed bosses and constrained encounters.
* [ ] (Optional) Implement Showdown mods approximating specific ROM hacks.
* [ ] Add curriculum learning over increasing difficulty.
* [ ] Advanced analyses:

  * Risk-sensitive vs risk-neutral policies.
  * Transfer between gauntlets.
  * Sensitivity to move/encounter availability.

---

## Status

Currently, this repository is in the **design / planning** phase.

The README serves as the main specification of:

* The environment design (gauntlets, box/party, movesets, phases).
* The separation between battle policy and run-level policy.
* The planned implementation roadmap.

As code is implemented, this document will be updated to mark which features are complete, in progress, or deferred.
