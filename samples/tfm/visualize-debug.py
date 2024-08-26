#!/usr/bin/env python3
# %%

import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("../../build/*LNRHO*.mesh")

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


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=500, blit=True)
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


# %%
# Profiles
import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("../../build/*step*000000000*00*B11mean_x.profile")
files += glob.glob("../../build/*step*000000000*00*B12mean_x.profile")
files += glob.glob("../../build/*step*000000000*00*B21mean_y.profile")
files += glob.glob("../../build/*step*000000000*00*B22mean_y.profile")
files += glob.glob("../../build/*step*000000000*00*Umean*.profile")
files += glob.glob("../../build/*step*000000000*00*ucross*.profile")

for file in files:
    arr = np.fromfile(
        file,
        dtype=np.double,
    )

    plt.plot(arr)
    plt.title(file)
    # plt.colorbar()
    plt.show()