import numpy as np
import os

os.makedirs("../data", exist_ok=True)

np.random.seed(42)

seq_lengths = [128, 256, 512, 1024, 2048, 4096]
d_model = 512   # = n_heads * d_head (8 * 64)

for S in seq_lengths:
    print(f"Generating data for S={S}")

    X  = np.random.randn(1, S, d_model).astype(np.float32)
    Wq = np.random.randn(d_model, d_model).astype(np.float32)
    Wk = np.random.randn(d_model, d_model).astype(np.float32)
    Wv = np.random.randn(d_model, d_model).astype(np.float32)

    np.save(f"../data/X_{S}.npy", X)
    np.save(f"../data/Wq_{S}.npy", Wq)
    np.save(f"../data/Wk_{S}.npy", Wk)
    np.save(f"../data/Wv_{S}.npy", Wv)