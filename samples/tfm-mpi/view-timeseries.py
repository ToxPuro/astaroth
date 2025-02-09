#!/usr/bin/env python3
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Load timeseries
filepath = 'timeseries.csv' # Path to the timeseries (current working directory by default)
df = pd.read_csv(filepath)
df


# %%
# Plot
def plot(df, label):
    df0 = df[df['label'] == label] # Filter rows matching label

    plt.title(label)
    plt.plot(df0['step'], df0['min'], label='min')
    plt.plot(df0['step'], df0['rms'], label='rms')
    plt.plot(df0['step'], df0['max'], label='max')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'{label}-timeseries.png')
    plt.show()
    plt.close()


# %%
# View some statistics
plot(df, 'uu')
plot(df, 'TF_a11')
plot(df, 'TF_a12')
plot(df, 'TF_a21')
plot(df, 'TF_a22')
plot(df, 'TF_uxb11')
plot(df, 'TF_uxb12')
plot(df, 'TF_uxb21')
plot(df, 'TF_uxb22')
