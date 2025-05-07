#!/usr/bin/env python3
# %%
# Pack
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

print(f'cwd: {os.getcwd()}')
outdir = f"{os.getcwd()}/../build"

# %%
# Packing
files = glob.glob(f"{outdir}/bm-pack-*")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df = df.drop(['sample', 'nsamples', 'jobid'], axis=1)
df = df.groupby(list(df.columns.difference(['ns']))).describe()
df = df.xs(256, level='dim')
df = df.xs(3, level='radius')
df = df.xs(3, level='ndims')
df = df['ns']['50%']

df = df.unstack('impl')
df

# %%
# Rank reordering
# files = glob.glob(f"{outdir}/bm-rank-reordering-*")
# files = glob.glob(f"{outdir}/../bm-test-rank-reordering.csv")
files = glob.glob(f"{outdir}/bm-rank-reordering-*.csv")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df = df.drop(['sample', 'nsamples', 'jobid'], axis=1)
df = df.groupby(list(df.columns.difference(['ns']))).describe()
df = df.xs(0, level='rank')
df = df.xs(64, level='nprocs')
df = df.xs(3, level='radius')
df = df['ns']['50%']

df = df.unstack('impl')
# df = df.drop(['mpi-default-custom-decomp'], axis=1) # Debug
# df = df.drop(['mpi-no-custom-decomp'], axis=1) # Debug
# df['hierarchical-to-no'] = df['hierarchical'] / df['mpi-no']
# df['hierarchical-to-default'] = df['hierarchical'] / df['mpi-default']
df

# %%
# Collective communication
files = glob.glob(f"{outdir}/bm-collective-comm*.csv")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df = df.drop(['sample', 'nsamples', 'jobid'], axis=1)
df = df.groupby(list(df.columns.difference(['ns']))).describe()
df = df.xs(0, level='rank')
df = df.xs(1, level='nprocs')
df = df.xs(3, level='radius')

df = df['ns']['50%']
df = df.unstack('impl')
df

# %%
# Strong scaling
files = glob.glob(f"/users/pekkila/astaroth/build/bm-tfm-mpi*.csv")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df = df.drop(['sample', 'nsamples', 'jobid'], axis=1)
df = df.groupby(list(df.columns.difference(['ns']))).describe()
df = df.xs(0, level='rank')
df = df.xs(3, level='radius')
df = df.xs(256, level='gnx')
df = df.xs(256, level='gny')
df = df.xs(256, level='gnz')

# Ensure the local dims make sense before dropping
df = df.droplevel(['lnx', 'lny', 'lnz'])

df = df['ns']['50%']
df = df.unstack('impl')
df

# %%
# Weak scaling
files = glob.glob(f"/users/pekkila/astaroth/build/bm-tfm-mpi*.csv")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df = df.drop(['sample', 'nsamples', 'jobid'], axis=1)
df = df.groupby(list(df.columns.difference(['ns']))).describe()
df = df.xs(0, level='rank')
df = df.xs(3, level='radius')
df = df.xs(256, level='lnx')
df = df.xs(256, level='lny')
df = df.xs(256, level='lnz')

# Ensure the global dims make sense before dropping
df = df.droplevel(['gnx', 'gny', 'gnz'])

df = df['ns']['50%']
df = df.unstack('impl')
df
