#!/usr/bin/env python3
# %%
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../build/scaling.csv', names=['case','nprocs', 'us', 'jobid'])
df = df.sort_values(by='nprocs')
df

df_cart = df[df['case']=='cart']
df_pack = df[df['case']=='pack']
plt.scatter(df_cart['nprocs'], df_cart['us'], label='cart')
plt.scatter(df_pack['nprocs'], df_pack['us'], label='pack')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()

df
