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
spacing = box_size / nz

def pos(i):
    return (i - radius + 0.5) * spacing - box_size/2

x = np.arange(0, mz)
ysin = amplitude * np.sin(wavenumber * pos(x))
ycos = amplitude * np.cos(wavenumber * pos(x))
plt.plot(x, ysin)
plt.plot(x, ycos)
plt.grid(which='both')
plt.xticks(np.arange(0, mz))


plt.axhline(0, color="red", alpha=0.5)
plt.axvline(radius, color="red", alpha=0.5)
plt.axvline(radius + (nz-1)/2, color="red", alpha=0.5)
plt.axvline(radius + nz - 1, color="red", alpha=0.5)

print(f"len(x): {len(x)}")
print(f"x[radius]: {x[radius]}")
print(f"x[radius+nz]: {x[radius+nz-1]}")
print(f"ysin[radius]: {ysin[radius]}")
print(f"ysin[radius+nz]: {ysin[radius+nz-1]}")
print(f"ycos[radius]: {ycos[radius]}")
print(f"ycos[radius+nz]: {ycos[radius+nz-1]}")

# %%
# Roberts

# Setup grid
iz = np.arange(0, mz)
z = pos(iz)
x, y = np.meshgrid(z, z)

# Setup data
amplitude = 1
wavenumber = 2
ux = amplitude * -np.cos(wavenumber * x) * np.sin(wavenumber * y)
uy = amplitude * np.sin(wavenumber * x) * np.cos(wavenumber * y)
uz = amplitude * np.sqrt(2) * np.cos(wavenumber * x) * np.cos(wavenumber * y)

def plot(x, y, u):
    plt.pcolormesh(x, y, u)
    plt.axis('scaled')
    plt.colorbar()
    plt.axvline(pos(radius-0.5), color='black')
    plt.axvline(pos(radius+1-0.5), color='red')
    plt.axhline(pos(radius-0.5), color='black')
    plt.axhline(pos(radius+1-0.5), color='red')

    plt.axvline(pos(mz-1-radius-0.5), color='red')
    plt.axvline(pos(mz-1-radius+1-0.5), color='black')
    plt.axhline(pos(mz-1-radius-0.5), color='red')
    plt.axhline(pos(mz-1-radius+1-0.5), color='black')
    # plt.axhline(pos(radius), color='red')
    # plt.axhline(pos(radius+1), color='red')
    # plt.axvline(pos(radius+1), color='red')
    # plt.axhline(pos(radius+1), color='red')
    # plt.axvline(pos(mz-radius-1), color='red')
    # plt.axhline(pos(mz-radius-1), color='red')
    plt.show()

plot(x, y, ux)
plot(x, y, uy)
plot(x, y, uz)

uxc = ux[radius:-radius, radius:-radius]
uyc = uy[radius:-radius, radius:-radius]
uzc = uz[radius:-radius, radius:-radius]
urms = np.sqrt(np.sum((uxc**2 + uyc**2 + uzc**2))/nz**2)
assert(np.allclose(urms, 1))


# %%

# %%

# %%
# Check the file
plt.plot(np.fromfile("../../build/cosine-profile.out"))
plt.plot(np.fromfile("../../build/sine-profile.out"))

plt.axhline(0, color="red", alpha=0.5)
plt.axvline(radius, color="red", alpha=0.5)
plt.axvline(radius + nz - 1, color="red", alpha=0.5)
plt.title("Amplitude 1, Wavenumber 1")
