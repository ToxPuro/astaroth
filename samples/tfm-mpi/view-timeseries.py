#!/usr/bin/env python3
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# %%
# Load timeseries
filepath = 'timeseries.csv' # Path to the timeseries (current working directory by default)
df = pd.read_csv(filepath)
df


# %%
# Plot
# def plot(df, label):
#     df0 = df[df['label'] == label] # Filter rows matching label

#     plt.title(label)
#     plt.plot(df0['step'], df0['min'], label='min')
#     plt.plot(df0['step'], df0['rms'], label='rms')
#     plt.plot(df0['step'], df0['avg'], label='avg')
#     plt.plot(df0['step'], df0['max'], label='max')
#     plt.xlabel('Step')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.savefig(f'{label}-timeseries.png')
#     plt.show()
#     plt.close()


# %%
# View some statistics
# plot(df, 'uu')
# plot(df, 'TF_a11')
# plot(df, 'TF_a12')
# plot(df, 'TF_a21')
# plot(df, 'TF_a22')
# plot(df, 'TF_uxb11')
# plot(df, 'TF_uxb12')
# plot(df, 'TF_uxb21')
# plot(df, 'TF_uxb22')

# %%
fields = df['label'].unique()
cols = int(np.ceil(np.sqrt(len(fields))))
rows = int(np.ceil(len(fields) / cols))

diagnostics = ['min', 'rms', 'avg', 'max']

fig, axs = plt.subplots(rows, cols, layout="constrained")
fig.set_figheight(15)
fig.set_figwidth(20)
for i, field in enumerate(fields):
    for diagnostic in diagnostics:
        curr_col = i % cols
        curr_row = i // cols
        df0 = df[df['label'] == field].sort_values(by='step')
        axs[curr_row, curr_col].plot(df0['step'], df0[diagnostic], label=diagnostic)
        axs[curr_row, curr_col].set_title(field)
        axs[curr_row, curr_col].legend()

outdir='output'
outfile = f'{outdir}/timeseries.png'
print(f"Creating output directory at {outdir}")
os.makedirs(outdir, exist_ok=True)
print(f"Writing {outfile}")
plt.savefig(outfile)
