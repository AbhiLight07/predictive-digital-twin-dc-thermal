import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
from pathlib import Path

out_dir = Path("docs/snapshot/server_sets")
out_dir.mkdir(parents=True, exist_ok=True)

d1 = pd.read_parquet("data/interim/dataset1/_combined.parquet")
d1_srvs = sorted(list(set(d1["Series"].astype(str).str.strip().unique())))

d2 = pd.read_parquet("data/interim/dataset2/_combined.parquet")
d2_srvs = sorted(list(set(d2["Series"].astype(str).str.strip().unique())))

# 1. D1 E1 Split A (76 servers, excluding 5 with no fan/inlet)
no_fan = ["qh2-rcc196", "qh2-rcc197", "qh2-rcc198", "qh2-rcc199", "qh2-rcc200"]
d1_s4_76 = [s for s in d1_srvs if s not in no_fan]
pd.DataFrame({"server": d1_s4_76}).to_csv(out_dir / "dataset1_e1_split_a_servers.csv", index=False)

# 2. D2 E1 Split A (60 servers, excluding 4 disqualified in prepare_data: rcc263, rcc264, rcc226, rcc205)
d2_disqual = ["qh2-rcc205", "qh2-rcc226", "qh2-rcc263", "qh2-rcc264"]
d2_qual_60 = [s for s in d2_srvs if s not in d2_disqual]
pd.DataFrame({"server": d2_qual_60}).to_csv(out_dir / "dataset2_e1_split_a_servers.csv", index=False)

# 3. D1 Split C Test Servers (seed=42)
from ttwin.splits import split_strict
_, _, d1_te = split_strict(d1[~d1["Series"].astype(str).str.strip().isin(no_fan)], seed=42)
d1_c_test_srvs = sorted(list(set(d1_te["Series"].astype(str).str.strip().unique())))
pd.DataFrame({"server": d1_c_test_srvs}).to_csv(out_dir / "dataset1_split_c_test_servers.csv", index=False)

# 4. D2 Split C Test Servers (seed=42)
_, _, d2_te = split_strict(d2[~d2["Series"].astype(str).str.strip().isin(d2_disqual)], seed=42)
d2_c_test_srvs = sorted(list(set(d2_te["Series"].astype(str).str.strip().unique())))
pd.DataFrame({"server": d2_c_test_srvs}).to_csv(out_dir / "dataset2_split_c_test_servers.csv", index=False)

# 5. D1 E4 Adaptation Target Servers (16 servers)
from ttwin.splits import get_group_kfold_splits
splits = get_group_kfold_splits(d1[~d1["Series"].astype(str).str.strip().isin(no_fan)], n_splits=5, seed=42)
_, e4_target_srvs = splits[0]
pd.DataFrame({"server": sorted(e4_target_srvs)}).to_csv(out_dir / "dataset1_e4_target_servers.csv", index=False)

# 6. D2 E3 Cross-Cluster Servers (60 servers)
pd.DataFrame({"server": d2_qual_60}).to_csv(out_dir / "dataset2_e3_cross_cluster_servers.csv", index=False)

print("Generated canonical experiment server set CSVs:")
for p in sorted(out_dir.glob("*.csv")):
    n = len(pd.read_csv(p))
    print(f"  {p.name}: {n} servers")
