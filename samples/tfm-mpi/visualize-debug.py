#!/usr/bin/env python3
# %%

import matplotlib.pyplot as plt
import numpy as np
import glob

field = "UUX"
files = glob.glob(f"../../build/*{field}*.mesh")
files.sort()

nn = np.array((32, 32, 32))

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
    arr = arr.reshape(nn)

    data = arr[int(nn[0]/2), :, :]
    im = ax.imshow(data, animated=True)
    if i == 0:
        im = ax.imshow(data)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# plt.show()
writer = animation.FFMpegWriter(fps=24, bitrate=1800)
ani.save(f"{field}.mp4", writer=writer)


# %%
# Plot collective
import matplotlib.pyplot as plt
import numpy as np
import glob

nn = np.array((32, 32, 32))

files = glob.glob("../../build/debug*00000*uxb*.mesh")
# files = glob.glob(f'../../build/test.mesh')
files.sort()
for file in files:
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    arr = arr.reshape(nn)

    plt.imshow(arr[int(nn[0]/2), :, :])
    plt.title(file)
    # plt.colorbar()
    plt.show()

# %%
# Plot distributed
import matplotlib.pyplot as plt
import numpy as np
import glob

nn = np.array((16,16,16))
rr = np.array((3,3,3))
mm = 2*rr + nn
nprocs = 8

files = glob.glob("../../build/proc-*00002-*UUX.mesh")
# files = glob.glob(f'../../build/test.mesh')
files.sort()
for file in files:
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    arr = arr.reshape(mm)

    plt.imshow(arr[int(nn[0]/2), :, :], vmin=0, vmax=nprocs)
    # plt.imshow(arr[0, :, :], vmin=0, vmax=nprocs)
    plt.colorbar()
    plt.title(file)
    # plt.colorbar()
    plt.show()

# %%
# Plot distributed
import matplotlib.pyplot as plt
import numpy as np
import glob

nn = np.array((16,16,16))
rr = np.array((3,3,3))
mm = 2*rr + nn

files = glob.glob("../../build/proc-0-*00004*TF_a*_x*.mesh")
# files = glob.glob(f'../../build/test.mesh')
files.sort()
for file in files:
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    arr = arr.reshape(mm)

    plt.imshow(arr[int(nn[0]/2), :, :])
    # plt.imshow(arr[0, :, :], vmin=0, vmax=nprocs)
    plt.colorbar()
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
step = 6
files = glob.glob(f"../../build/debug-step-{str(step).zfill(12)}-tfm-*.profile")
files.sort()
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
