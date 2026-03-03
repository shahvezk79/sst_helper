import numpy as np
import os

filepath = ".cache/embeddings/sst_embeddings_qwen3_8b.npy"
if os.path.exists(filepath):
    arr = np.load(filepath)
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}")
    print(f"Max: {np.max(arr)}")
    print(f"Min: {np.min(arr)}")
    print(f"Mean: {np.mean(arr)}")
    print(f"Any NaN: {np.isnan(arr).any()}")
    print(f"Any Inf: {np.isinf(arr).any()}")
    print(f"L2 Norms (first 10): {np.linalg.norm(arr[:10], axis=1)}")
    
    # Check for denormals / very small numbers
    print(f"Values close to 0 but not 0: {np.sum((np.abs(arr) > 0) & (np.abs(arr) < 1e-30))}")
else:
    print(f"File not found: {filepath}")
