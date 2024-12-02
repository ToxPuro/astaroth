#!/usr/bin/env python3
# %%

import matplotlib.pyplot as plt
import numpy as np
import glob

field = "UUX"
files = glob.glob(f"../../build/*{field}*.mesh")
files.sort()

# Animate
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()
ims = []
for i, file in enumerate(files):
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    arr = arr.reshape((38, 38, 38))

    data = arr[:, :, 16]
    im = ax.imshow(data, animated=True)
    if i == 0:
        im = ax.imshow(data)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# plt.show()
writer = animation.FFMpegWriter(fps=24, bitrate=1800)
ani.save(f"{field}.mp4", writer=writer)


# %%
# Plot
import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("../../build/*LNRHO*.mesh")
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
import os

# files = glob.glob("../../build/*step*000000000*00*B11mean_x.profile")
# files += glob.glob("../../build/*step*000000000*00*B12mean_x.profile")
# files += glob.glob("../../build/*step*000000000*00*B21mean_y.profile")
# files += glob.glob("../../build/*step*000000000*00*B22mean_y.profile")
# files += glob.glob("../../build/*step*000000000*00*Umean*.profile")
# files += glob.glob("../../build/*step*000000000*00*ucross*.profile")
# files = glob.glob("../../build/debug-step-000000000200-*ucrossb11mean*")
# files = glob.glob("../../build/debug-step-000000000200-*.profile")
# files = glob.glob("../../build/debug-step-000000000200-*ucrossb21mean_x*.profile")
# files += glob.glob("../../build/debug-step-000000000100-*ucrossb22mean_y*.profile")
# files += glob.glob("../../build/debug-step-000000000100-*Umean_x*.profile")
# files = glob.glob("../../build/debug-step-000000000100-*ucrossb22mean_z*.profile")

fig, axs = plt.subplots(9, 3, layout="constrained")
fig.set_figheight(15)
fig.set_figwidth(20)

# for step in range(0, 10+1):
step = 0
files = glob.glob(f"../../build/debug-step-{str(step).zfill(12)}-tfm-*.profile")
print(files)
for i, file in enumerate(files):
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    col = i % 3
    row = i // 3

    axs[row, col].plot(arr)
    axs[row, col].set_title(os.path.basename(file))
    # plt.plot(arr)
    # plt.title(file)
    # plt.show()

plt.show()

# ani = animation.FuncAnimation(fig, update)
