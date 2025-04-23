#!/usr/bin/env python3
# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../build/scaling.csv', names=['case','nprocs', 'us', 'jobid'])
df = df.sort_values(by='nprocs')
df

for case in df['case'].unique():
    df0 = df[df['case'] == case]
    plt.scatter(df0['nprocs'], df0['us'], label=case)

plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

# df


# %%
# Pack
import matplotlib.pyplot as plt
import pandas as pd
import os

print(f'cwd: {os.getcwd()}')

df = pd.read_csv('../build/bm-pack.csv')
df

stats = df.groupby('impl')['ns'].describe()


stats_median = stats['50%'].sort_values()
plt.barh(stats_median.index, stats_median)


# %%
# Rank reordering
import matplotlib.pyplot as plt
import pandas as pd
import os

print(f'cwd: {os.getcwd()}')

df = pd.read_csv('acc-comm/build/bm-rank-reordering-3-0.csv')
df1 = pd.read_csv('acc-comm/build/bm-rank-reordering-4-0.csv')
df2 = pd.read_csv('acc-comm/build/bm-rank-reordering-5-0.csv')
df3 = pd.concat((df, df1, df2))

df3.groupby(['impl', 'nx', 'ny', 'nz']).describe()['ns']['50%']

# df

# stats = df.groupby('impl')['ns'].describe()
# stats['50%']
# stats_normalized = stats['50%'] / stats['50%'].min()
# stats_normalized

stats_median = stats['50%'].sort_values()
plt.barh(stats_median.index, stats_median)
plt.show()

plt.plot(df[df['impl'] == 'mpi-no']['sample'], df[df['impl'] == 'mpi-no']['ns'], label='no')
plt.plot(df[df['impl'] == 'mpi-default']['sample'], df[df['impl'] == 'mpi-default']['ns'], label='default')
plt.plot(df[df['impl'] == 'hierarchical']['sample'], df[df['impl'] == 'hierarchical']['ns'], label='hierarchical')
plt.legend()
plt.show()

df = df[['impl', 'sample', 'ns']]
df['normalized'] = df['ns'] / df['ns'].min()
plt.plot(df[df['impl'] == 'mpi-no']['sample'], df[df['impl'] == 'mpi-no']['normalized'], label='no')
plt.plot(df[df['impl'] == 'mpi-default']['sample'], df[df['impl'] == 'mpi-default']['normalized'], label='default')
plt.plot(df[df['impl'] == 'hierarchical']['sample'], df[df['impl'] == 'hierarchical']['normalized'], label='hierarchical')
plt.legend()
plt.show()

stats

# %%
# Pipeline
import matplotlib.pyplot as plt
import pandas as pd
import os

print(f'cwd: {os.getcwd()}')

df = pd.read_csv('../build/bm-pipelining-0-0.csv')
df

stats = df.groupby('impl')['ns'].describe()
stats
