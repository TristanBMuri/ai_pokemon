from nuzlocke_gauntlet_rl.envs.embedding import BattleEmbedder
import numpy as np

def test_embedding_shape():
    embedder = BattleEmbedder()
    low, high, shape, dtype = embedder.describe_embedding()
    print(f"Embedding Shape: {shape}")
    print(f"Expected: (75,)")
    
    assert shape == (75,), f"Shape mismatch! Got {shape}"
    print("SUCCESS: Shape matches 75 (Original).")

if __name__ == "__main__":
    test_embedding_shape()
