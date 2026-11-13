import sys
sys.stdout.reconfigure(line_buffering=True)
import pandas as pd
import numpy as np

for ds_name, p in [('Dataset 1', 'data/interim/dataset1/_combined.parquet'), ('Dataset 2', 'data/interim/dataset2/_combined.parquet')]:
    print('='*50)
    print(ds_name)
    df = pd.read_parquet(p)
    print(f'Rows: {len(df)}')
    print(f'Columns: {list(df.columns)}')
    servers = sorted(df['Series'].unique())
    print(f'Server count: {len(servers)}')
    print(f'Servers: {servers}')
    tmin = df['Time'].min()
    tmax = df['Time'].max()
    print(f'Timestamp min: {tmin}, max: {tmax}')
    # sampling interval
    srv0 = servers[0]
    sub = df[df['Series']==srv0].sort_values('Time')
    td = sub['Time'].diff().dropna()
    print(f'Sampling interval median: {td.median()}, mode: {td.mode().iloc[0]}')

    # Missingness of columns actually used
    used_cols = [c for c in df.columns if c not in ['Series', 'Time']]
    print('Missingness:')
    for c in used_cols:
        pct = df[c].isna().mean() * 100
        print(f'  {c}: {df[c].isna().sum()} ({pct:.2f}%)')
