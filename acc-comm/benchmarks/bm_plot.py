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
