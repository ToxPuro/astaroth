#!/usr/bin/env python3
# %%
# Packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

print(f'cwd: {os.getcwd()}')
outdir = "/users/pekkila/astaroth/build"

def select(df, column, value):
    return df[df[column] == value].drop(column, axis=1)

def scatter(data, label):
    print(len(data))
    plt.scatter(np.arange(len(data)), data, label=label, alpha=0.6, s=15.0)

# %%
# Packing
files = glob.glob(f"{outdir}/bm-pack-*")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

nwarmup_steps = 5
df = df[df['sample'] >= nwarmup_steps]
# Check the distribution
for impl in df['impl'].unique():
    plt.plot(df[(df['impl'] == impl) & (df['dim'] == 256)]['ns'].values[:], label=impl)
plt.legend()
plt.show()

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

# Drop warmup steps
nwarmup_steps = 5
df = df[df['sample'] >= nwarmup_steps]
scatter(df[(df['impl'] == 'hierarchical') & (df['rank'] == 0)]['ns'].values, label='hierarchical')
scatter(df[(df['impl'] == 'mpi-no') & (df['rank'] == 0)]['ns'].values, label='mpi-no')
scatter(df[(df['impl'] == 'mpi-default') & (df['rank'] == 0)]['ns'].values, label='mpi-default')
plt.legend()
plt.show()

# Filter outliers and report (IQR)
# df = select(df, 'radius', 3)
# df = select(df, 'rank', 0)
# df = select(df, 'nprocs', 64)
# df = select(df, 'nsamples', 100)
# assert(len(df['jobid'].unique()) == 1)
# df = select(df, 'jobid', df['jobid'].unique()[-1])
# df = df.drop('sample', axis=1)
# df['nxyz'] = df['nx'].astype(str) + ',' + df['ny'].astype(str) + ',' + df['nz'].astype(str)
# df = df.drop(['nx', 'ny', 'nz'], axis=1)
# group = df.groupby(list(df.columns.difference(['ns'])))
# iqr = group.quartile(0.75) - group.quartile(0.25)
# df = df[]

# Produce the results
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

nwarmup_steps = 5
df = df[(df['sample'] >= nwarmup_steps) & (df['rank'] == 0)]
for value in [1,2,4,8,16]:
    scatter(df[(df['impl'] == 'mpi-he') & (df['rank'] == 0) & (df['npack'] == value)]['ns'].values, label='mpi-he')
    scatter(df[(df['impl'] == 'mpi-he-hindexed') & (df['rank'] == 0) & (df['npack'] == value)]['ns'].values, label='mpi-he-hindexed')
    scatter(df[(df['impl'] == 'acm-packed-he') & (df['rank'] == 0) & (df['npack'] == value)]['ns'].values, label='acm-packed-he')
    scatter(df[(df['impl'] == 'acm-batched-he') & (df['rank'] == 0) & (df['npack'] == value)]['ns'].values, label='acm-batched-he')
    plt.yscale('log')
    plt.legend()
    plt.show()

df = df.drop(['sample', 'nsamples', 'jobid'], axis=1)
df = df.groupby(list(df.columns.difference(['ns']))).describe()
df = df.xs(0, level='rank')
df = df.xs(64, level='nprocs')
df = df.xs(3, level='radius')

df = df.xs(512, level='nx')
df = df.xs(512, level='ny')
df = df.xs(512, level='nz')

df = df['ns']['50%']
df = df.unstack('impl')
df

# %%
# Strong scaling
files = glob.glob(f"{outdir}/bm-tfm-mpi*.csv")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

nwarmup_steps = 1
nn = 256
df = df[(df['sample'] >= nwarmup_steps) & (df['rank'] == 0)]
plt.plot(df[(df['nprocs'] == 1) & (df['rank'] == 0) & (df['gnz'] == nn)]['ns'].values, label='1')
plt.plot(df[(df['nprocs'] == 2) & (df['rank'] == 0) & (df['gnz'] == nn)]['ns'].values, label='2')
plt.plot(df[(df['nprocs'] == 4) & (df['rank'] == 0) & (df['gnz'] == nn)]['ns'].values, label='4')
plt.plot(df[(df['nprocs'] == 8) & (df['rank'] == 0) & (df['gnz'] == nn)]['ns'].values, label='8')
plt.plot(df[(df['nprocs'] == 64) & (df['rank'] == 0) & (df['gnz'] == nn)]['ns'].values, label='64')
plt.yscale('log')
plt.legend()
plt.show()

df = df.drop(['sample', 'nsamples', 'jobid'], axis=1)
df = df.groupby(list(df.columns.difference(['ns']))).describe()
df = df.xs(0, level='rank')
df = df.xs(3, level='radius')
df = df.xs(nn, level='gnx')
df = df.xs(nn, level='gny')
df = df.xs(nn, level='gnz')
df = df.xs(100, level='nsteps_per_samples')

# Ensure the local dims make sense before dropping
df = df.droplevel(['lnx', 'lny', 'lnz'])

df = df['ns']['50%']
# df = df.unstack(['impl'])
df



# %%
# Weak scaling
files = glob.glob(f"{outdir}/bm-tfm-mpi*.csv")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

nwarmup_steps = 0
nn = 256
df = df[(df['sample'] >= nwarmup_steps) & (df['rank'] == 0)]
plt.plot(df[(df['nprocs'] == 1) & (df['rank'] == 0) & (df['lnz'] == 256)]['ns'].values, label='1')
plt.plot(df[(df['nprocs'] == 2) & (df['rank'] == 0) & (df['lnz'] == 256)]['ns'].values, label='2')
plt.plot(df[(df['nprocs'] == 4) & (df['rank'] == 0) & (df['lnz'] == 256)]['ns'].values, label='4')
plt.plot(df[(df['nprocs'] == 8) & (df['rank'] == 0) & (df['lnz'] == 256)]['ns'].values, label='8')
plt.plot(df[(df['nprocs'] == 64) & (df['rank'] == 0) & (df['lnz'] == 256)]['ns'].values, label='64')
plt.plot(df[(df['nprocs'] == 512) & (df['rank'] == 0) & (df['lnz'] == 256)]['ns'].values, label='512')
plt.plot(df[(df['nprocs'] == 4096) & (df['rank'] == 0) & (df['lnz'] == 256)]['ns'].values, label='4096')
plt.yscale('log')
plt.legend()
plt.show()

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

# %%
# Strong scaling revisited (need to take multiple samples to mitigate noise)
# Quick results (take just the final value)
files = glob.glob(f"{outdir}/bm-tfm-mpi*.csv")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df = df[df['sample'] == df['sample'].max()] # Select the last sample
df = df[(df['gnx'] == 256) & (df['gny'] == 256) & (df['gnz'] == 256)]
df = df[['nprocs', 'ns']].sort_values('nprocs')

df['speedup'] = df[df['nprocs']==1]['ns'].values / df['ns']

df['ideal'] = df['nprocs']
print(df[['nprocs', 'ideal', 'speedup']].to_csv(index=False))
df


# df = df[['nprocs', 'ideal', 'speedup']]
# df.columns['speedup'] = 'measured'
# plt.loglog(df['nprocs'], df[['nprocs', 'speedup']])
# plt.show()
#df['ideal'] = df[df['nprocs'] == 1]['ns'].values / df['nprocs']
#df

# %%
# Weak scaling, quick efficiency results

files = glob.glob(f"{outdir}/bm-tfm-mpi*.csv")
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df = df[df['sample'] == df['sample'].max()] # Select the last sample
df = df[(df['lnx'] == 256) & (df['lny'] == 256) & (df['lnz'] == 256)]
df = df.sort_values('nprocs')
df['ideal'] = 100
df['efficiency'] = 100*df[df['nprocs'] == 1]['ns'].values / df['ns']
print(df[['nprocs', 'ideal', 'efficiency']].to_csv(index=False))
plt.plot(df['nprocs'], df[['ideal','efficiency']])

# %%
# TMP drafts for measuring times
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv(f"{outdir}/timeline.csv")
df = pd.read_csv(f"{outdir}/timeline_verify.csv")
df
plt.barh(df['label'], df['ns']/1e9)

# %%
ns = 2634484064
ns / 1e6 / 100

# %%
import matplotlib.pyplot as plt
import pandas as pd
files = [f'{outdir}/bm-tfm-mpi-0-122191-0.csv', f'{outdir}/bm-tfm-mpi-0-31510-0.csv']
df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)

df = df.drop(['sample', 'nsamples', 'jobid'], axis=1)
# df = df.groupby(list(df.columns.difference(['ns']))).describe()
df = df[['nprocs', 'ns']]

plt.plot(df[df['nprocs']==1]['ns'].values, label="1 device")
plt.plot(df[df['nprocs']==512]['ns'].values, label="512 devices")
ax = plt.gca()
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.legend()
ax.set_ylabel("Running time")
ax.set_xlabel("nth step")

df
