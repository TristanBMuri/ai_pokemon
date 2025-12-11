Project "Triumvirate": Hierarchical RL for Nuzlocke Gauntlets (Gold Spec)
1. Core Tech Stack

    Language: Python 3.13 (Strict).

    Framework: Ray RLLib (Distributed RL).

    Simulation: poke-env + Local Pokemon Showdown (Docker).

    Hardware Opt: Linux KSM (Kernel Same-page Merging) + Ray Object Store (Plasma) for memory efficiency on 60GB RAM.

2. The Architecture: "The Triumvirate"
A. The Strategist (High-Level Policy)

    Role: Campaign Manager & Risk Assessor.

    The Map: A graph containing Mandatory Nodes (Gyms) and Optional Nodes (Item/Encounter events).

    Input: Map State, Inventory, FreeRebuildAvailable flag, Confidence Score (0.0 - 1.0) from Builder.

    Simplified State: Party Health is assumed MAX (100%) at the start of every node.

    Action:

        Choose Next Node (Skip optional vs. Fight optional).

        Output Risk Vector (Scalar importance for each Pokemon).

    Reward: Progress through Gauntlet + Value of Items obtained - Cost of deaths.

B. The Team Builder (Mid-Level Policy)

    Type: Autoregressive Transformer (Sequential Builder).

    Input: The "Box" (Set), Enemy Preview, Risk Vector.

    Action Space (The "Granular Build" Flow):

        Rebuild Decision: Keep Team vs Rebuild.

        Slot 1: Select Species → Select Move 1 → Move 2 → Move 3 → Move 4 → Select Item.

        Slot 2: Select Species → Select Move 1... (Conditioned on Slot 1).

        (Repeat for all 6 slots).

    Future Horizon: Pre-damaging/pre-statusing team members (currently disabled; health is always max).

    Economy (Rebuild Costs):

        Rebuild #1 (Post-Fight): Cost 0 (Free Rebuild).

        Rebuild #2+ (Same Node): Cost 5.

C. The Battler (Low-Level Policy)

    Type: LSTM-PPO (Recurrent).

    Role: Tactical Execution.

    Constraint: Respects Risk Vector (e.g., switches out VIPs).

    Stall Policy:

        PP Stalling is Allowed.

        Turn Limit: 200 Turns. (Penalty applied only if this limit is exceeded).

    State: Standard Battle/Team state (PP/Items reset between battles for now).

3. Data Strategy

    Source: Smogon Chaos Data (Download Once Rule).

    Hall of Fame: Database data/hall_of_fame.json stores winning teams to train the Builder against.

4. Training Curriculum

    The Dojo: Battler trains to imitate SimpleHeuristicPlayer.

    The Gauntlet (Tiers):

        PU/NU -> UU/RU -> OU.

    The "Underdog" Gold Standard:

        Final Test A: UU Box vs. OU Team.

        Final Test B: OU Box vs. Ubers Team.