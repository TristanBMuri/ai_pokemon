from nuzlocke_gauntlet_rl.envs.embedding import BattleEmbedder
import numpy as np

def test_embedding_shape():
    embedder = BattleEmbedder()
    low, high, shape, dtype = embedder.describe_embedding()
    print(f"Embedding Shape: {shape}")
    print(f"Expected: (583,)")
    
    assert shape == (583,), f"Shape mismatch! Got {shape}"
    print("SUCCESS: Shape matches 583 (Full Team Info + Types + Abilities).")

if __name__ == "__main__":
    test_embedding_shape()
