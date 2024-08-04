#!/usr/bin/env python3
# %%
import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("../build/*.data")

for file in files:
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    arr = arr.reshape((38, 38, 38))

    plt.imshow(arr[:, :, 16])
    plt.title(file)
    plt.show()
