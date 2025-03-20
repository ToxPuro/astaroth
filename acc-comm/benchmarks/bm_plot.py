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
