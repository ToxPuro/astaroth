#!/usr/bin/env python3
# %%
import numpy as np
import matplotlib.pyplot as plt

# Input parameters
box_size = 2 * np.pi
nz = 32
radius = 3
amplitude = 1
wavenumber = 1

# Derived parameters
mz = nz + 2 * radius
spacing = box_size / (nz - 1)

x = np.arange(0, mz)
y = amplitude * np.sin(wavenumber * spacing * (x - radius))
plt.plot(x, y)


plt.axhline(0, color="red", alpha=0.5)
plt.axvline(radius, color="red", alpha=0.5)
plt.axvline(radius + nz - 1, color="red", alpha=0.5)

print(f"len(x): {len(x)}")
print(f"x[radius]: {x[radius]}")
print(f"x[radius+nz]: {x[radius+nz-1]}")
print(f"y[radius]: {y[radius]}")
print(f"y[radius+nz]: {y[radius+nz-1]}")


# %%
# Check the file
plt.plot(np.fromfile("../../build/cosine-profile.out"))
plt.plot(np.fromfile("../../build/sine-profile.out"))

plt.axhline(0, color="red", alpha=0.5)
plt.axvline(radius, color="red", alpha=0.5)
plt.axvline(radius + nz - 1, color="red", alpha=0.5)
plt.title("Amplitude 1, Wavenumber 1")
