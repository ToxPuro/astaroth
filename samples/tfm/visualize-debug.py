#!/usr/bin/env python3
# %%

import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("../../build/*UUX*.data")

%matplotlib widget


# Animate
import matplotlib.animation as animation
import numpy as np

frames = []
for file in files:
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    arr = arr.reshape((38, 38, 38))

    frames.append(arr[:, :, 16])

fig, ax = plt.subplots()
im = ax.imshow(frames[0])
cbar = fig.colorbar(im, ax=ax)


def update(frame):
    im.set_array(frames[frame % len(frames)])
    #ax.set_title(f"Frame {frame}")
    return [im]


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, blit=True)
plt.show()

# %%
# Plot
for file in files:
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    arr = arr.reshape((38, 38, 38))

    plt.imshow(arr[:, :, 16])
    plt.title(file)
    # plt.colorbar()
    plt.show()
