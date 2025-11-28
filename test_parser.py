from nuzlocke_gauntlet_rl.data.parsers import load_kanto_leaders

def test_parser():
    print("Loading Kanto Leaders...")
    gauntlet = load_kanto_leaders()
    print(f"Found {len(gauntlet.trainers)} trainers.")
    
    for t in gauntlet.trainers:
        print(f"\nTrainer: {t.name}")
        print(f"Team Size: {len(t.team)}")
        for p in t.team:
            print(f"  - {p.species} (Lv {p.level}) @ {p.item}")
            print(f"    Moves: {p.moves}")

if __name__ == "__main__":
    test_parser()
