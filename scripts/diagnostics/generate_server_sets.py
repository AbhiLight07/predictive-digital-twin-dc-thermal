import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import numpy as np
from pathlib import Path

d1 = pd.read_parquet('data/interim/dataset1/_combined.parquet')
d2 = pd.read_parquet('data/interim/dataset2/_combined.parquet')

out_dir = Path('docs/snapshot/server_sets')
out_dir.mkdir(parents=True, exist_ok=True)

# 1. D1 Raw All Servers (81)
s_d1_all = sorted(list(set(d1['Series'].astype(str).str.strip().unique())))
pd.DataFrame({'server': s_d1_all}).to_csv(out_dir / 'dataset1_all_servers.csv', index=False)

# 2. D1 S4 Servers (76 - excluding 5 without fan/inlet)
# Who are the 5 servers without fan/inlet?
fan_missing = d1.groupby(d1['Series'].astype(str).str.strip())['Inlet_Temperature'].apply(lambda x: x.isna().mean())
s_d1_s4 = sorted(fan_missing[fan_missing < 0.5].index.tolist())
s_d1_no_fan = sorted(fan_missing[fan_missing >= 0.5].index.tolist())
pd.DataFrame({'server': s_d1_s4}).to_csv(out_dir / 'dataset1_s4_servers.csv', index=False)
pd.DataFrame({'server': s_d1_no_fan}).to_csv(out_dir / 'dataset1_no_fan_servers.csv', index=False)

# 3. D2 Raw All Servers (64)
s_d2_all = sorted(list(set(d2['Series'].astype(str).str.strip().unique())))
pd.DataFrame({'server': s_d2_all}).to_csv(out_dir / 'dataset2_all_servers.csv', index=False)

# 4. D2 Excluded servers in clean/audit:
# rcc263, rcc264 (100% missing temp), rcc226 (~79% missing temp), rcc205 (why?)
d2_temp_missing = d2.groupby(d2['Series'].astype(str).str.strip())['CPU1_Temperature'].apply(lambda x: x.isna().mean())
print('D2 high temp missing servers (>10%):')
print(d2_temp_missing[d2_temp_missing > 0.10])

# Look at E1 / E3 / E4 server sets used in experiments
import glob
print('\nChecking experiment server sets:')
for exp_dir in sorted(glob.glob('experiments/*')):
    p = Path(exp_dir)
    if not p.is_dir(): continue
    for csv_file in p.glob('*.csv'):
        try:
            df_exp = pd.read_csv(csv_file)
            if 'server' in df_exp.columns:
                unique_s = sorted(df_exp['server'].unique())
                print(f'{p.name}/{csv_file.name}: {len(unique_s)} servers')
                # write out server set
                safe_name = f'{p.name}_{csv_file.stem}_servers.csv'
                pd.DataFrame({'server': unique_s}).to_csv(out_dir / safe_name, index=False)
        except Exception as e:
            pass

print('\nDone writing server sets to', out_dir)
