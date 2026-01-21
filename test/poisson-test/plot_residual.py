import pandas as pd
import matplotlib.pyplot as plt
data1 = pd.read_csv("bicgstab.dat",header=None)
data2 = pd.read_csv("sor.dat",header=None)

fig, ax = plt.subplots()

data1.iloc[0].plot(ax=ax, label="BICGSTAB")
data2.iloc[0].plot(ax=ax, label="SOR")
ax.legend()
plt.show()
