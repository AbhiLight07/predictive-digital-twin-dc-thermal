import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import numpy as np

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import (
    split_chronological,
    get_group_kfold_splits,
    split_strict,
    get_split_b_seeds,
)

print("="*70)
print("EXACT SPLIT MEASUREMENTS")
print("="*70)

# 1. Dataset 1 (76 servers)
d1 = load_parquet("data/interim/dataset1/_combined.parquet")
d1["Series"] = d1["Series"].astype(str).str.strip()
no_fan = {"qh2-rcc196", "qh2-rcc197", "qh2-rcc198", "qh2-rcc199", "qh2-rcc200"}
d1 = d1[~d1["Series"].isin(no_fan)].copy()

print(f"\n--- DATASET 1 (76 SERVERS, {len(d1):,} ROWS) ---")

# Split A
tr_a, val_a, te_a = split_chronological(d1, train_frac=0.70, val_frac=0.10, embargo_steps=36)
print("SPLIT A (Dataset 1):")
print(f"  Train: {tr_a['Time'].min()} to {tr_a['Time'].max()} | {len(tr_a):,} rows | {tr_a['Series'].nunique()} servers")
print(f"  Val:   {val_a['Time'].min()} to {val_a['Time'].max()} | {len(val_a):,} rows | {val_a['Series'].nunique()} servers")
print(f"  Test:  {te_a['Time'].min()} to {te_a['Time'].max()} | {len(te_a):,} rows | {te_a['Series'].nunique()} servers")
embargo_delta = te_a['Time'].min() - val_a['Time'].max()
print(f"  Embargo gap (val end to test start): {embargo_delta} (steps: 36 x 10m = 6.0h)")

# Split B GroupKFold (seed=42)
splits_b = get_group_kfold_splits(d1, n_splits=5, seed=42)
print("\nSPLIT B (GroupKFold, 5 Folds, Dataset 1):")
for f_idx, (tr_s, te_s) in enumerate(splits_b):
    sub_te = d1[d1["Series"].isin(te_s)]
    print(f"  Fold {f_idx+1}: {len(te_s)} test servers | {len(sub_te):,} rows | Test Servers: {te_s}")

# Split C (seed=42)
tr_c, val_c, te_c = split_strict(d1, seed=42)
print("\nSPLIT C (Strict Spatiotemporal, Dataset 1, seed=42):")
print(f"  Train: {tr_c['Time'].min()} to {tr_c['Time'].max()} | {len(tr_c):,} rows | {tr_c['Series'].nunique()} servers")
print(f"  Val:   {val_c['Time'].min()} to {val_c['Time'].max()} | {len(val_c):,} rows | {val_c['Series'].nunique()} servers")
print(f"  Test:  {te_c['Time'].min()} to {te_c['Time'].max()} | {len(te_c):,} rows | {te_c['Series'].nunique()} servers")
print(f"  Held-out test servers ({te_c['Series'].nunique()}): {sorted(te_c['Series'].unique())}")
emb_c = te_c['Time'].min() - val_c['Time'].max()
print(f"  Embargo gap: {emb_c}")

# 2. Dataset 2 (60 servers)
d2 = load_parquet("data/interim/dataset2/_combined.parquet")
d2["Series"] = d2["Series"].astype(str).str.strip()
disqual = {"qh2-rcc263", "qh2-rcc264", "qh2-rcc226", "qh2-rcc205"}
d2 = d2[~d2["Series"].isin(disqual)].copy()

print(f"\n--- DATASET 2 (60 SERVERS, {len(d2):,} ROWS) ---")

tr_a2, val_a2, te_a2 = split_chronological(d2, train_frac=0.70, val_frac=0.10, embargo_steps=36)
print("SPLIT A (Dataset 2):")
print(f"  Train: {tr_a2['Time'].min()} to {tr_a2['Time'].max()} | {len(tr_a2):,} rows | {tr_a2['Series'].nunique()} servers")
print(f"  Val:   {val_a2['Time'].min()} to {val_a2['Time'].max()} | {len(val_a2):,} rows | {val_a2['Series'].nunique()} servers")
print(f"  Test:  {te_a2['Time'].min()} to {te_a2['Time'].max()} | {len(te_a2):,} rows | {te_a2['Series'].nunique()} servers")

tr_c2, val_c2, te_c2 = split_strict(d2, seed=42)
print("\nSPLIT C (Dataset 2, seed=42):")
print(f"  Train: {tr_c2['Time'].min()} to {tr_c2['Time'].max()} | {len(tr_c2):,} rows | {tr_c2['Series'].nunique()} servers")
print(f"  Val:   {val_c2['Time'].min()} to {val_c2['Time'].max()} | {len(val_c2):,} rows | {val_c2['Series'].nunique()} servers")
print(f"  Test:  {te_c2['Time'].min()} to {te_c2['Time'].max()} | {len(te_c2):,} rows | {te_c2['Series'].nunique()} servers")
print(f"  Held-out test servers ({te_c2['Series'].nunique()}): {sorted(te_c2['Series'].unique())}")
