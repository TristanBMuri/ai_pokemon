
import asyncio
from nuzlocke_gauntlet_rl.envs.real_battle_simulator import RealBattleSimulator
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec

def test_simulator():
    print("Initializing RealBattleSimulator...", flush=True)
    # Use the same model path
    simulator = RealBattleSimulator(model_path="models/ppo_risk_agent_v1")
    
    # Create dummy teams
    my_team = [
        PokemonSpec(species="Charizard", level=50, moves=["Flamethrower"]),
        PokemonSpec(species="Blastoise", level=50, moves=["Surf"])
    ]
    
    enemy_team = [
        PokemonSpec(species="Venusaur", level=50, moves=["Vine Whip"]),
        PokemonSpec(species="Pikachu", level=50, moves=["Thunderbolt"])
    ]
    
    print("Starting simulation...", flush=True)
    win, survivors = simulator.simulate_battle(my_team, enemy_team, risk_token=0)
    
    print(f"Simulation done. Win: {win}, Survivors: {survivors}", flush=True)

if __name__ == "__main__":
    test_simulator()
