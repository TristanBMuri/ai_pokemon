from nuzlocke_gauntlet_rl.utils.moveset_generator import MovesetGenerator
from nuzlocke_gauntlet_rl.utils.specs import PokemonSpec

def test_movesets():
    gen = MovesetGenerator()
    
    species_list = ["Charizard", "Blastoise", "Venusaur", "Pikachu"]
    
    print("Testing Moveset Generation...")
    for species in species_list:
        spec = PokemonSpec(species=species, level=50, moves=[])
        builds = gen.generate_builds(spec, n_builds=3)
        
        print(f"\n--- {species} ---")
        for i, build in enumerate(builds):
            print(f"Build {i}: {build}")
            
if __name__ == "__main__":
    test_movesets()
