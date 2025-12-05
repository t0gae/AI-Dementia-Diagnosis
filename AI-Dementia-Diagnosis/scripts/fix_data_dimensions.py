#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def fix_data_dimensions():
    print("Fixing data dimensions...")
    
    data_path = Path('data/processed/mri_scans.npy')
    X = np.load(data_path)
    
    print(f"Original shape: {X.shape}")
    
    if X.ndim == 6:
        X = X.squeeze(axis=-1)
        print(f"Fixed shape: {X.shape}")
        np.save(data_path, X)
        print("Data fixed and saved")
    else:
        print("Shape already correct")
    
    print(f"\nData stats:")
    print(f"  Min: {X.min():.4f}")
    print(f"  Max: {X.max():.4f}")
    print(f"  Mean: {X.mean():.4f}")

if __name__ == '__main__':
    fix_data_dimensions()