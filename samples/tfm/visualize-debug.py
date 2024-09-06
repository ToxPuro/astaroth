#!/usr/bin/env python3
# %%

import matplotlib.pyplot as plt
import numpy as np
import glob

files = glob.glob("../../build/*LNRHO*.mesh")
files.sort()
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


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=10, blit=True)
plt.show()

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
#files = glob.glob("../../build/debug-step-000000000200-*ucrossb11mean*")
#files = glob.glob("../../build/debug-step-000000000200-*.profile")
# files = glob.glob("../../build/debug-step-000000000200-*ucrossb21mean_x*.profile")
# files += glob.glob("../../build/debug-step-000000000100-*ucrossb22mean_y*.profile")
# files += glob.glob("../../build/debug-step-000000000100-*Umean_x*.profile")
#files = glob.glob("../../build/debug-step-000000000100-*ucrossb22mean_z*.profile")

fig, axs = plt.subplots(9,3, layout='constrained')
fig.set_figheight(15)
fig.set_figwidth(20)

#for step in range(0, 10+1):
step = 200
files = glob.glob(f"../../build/debug-step-{str(step).zfill(12)}-*.profile")
print(files)
for i, file in enumerate(files):
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    col = i %3
    row = i//3

    axs[row, col].plot(arr)
    axs[row, col].set_title(os.path.basename(file))
    # plt.plot(arr)
    # plt.title(file)
    # plt.show()

plt.show()

#ani = animation.FuncAnimation(fig, update)