#!/usr/bin/env python3
# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import glob
import os
import argparse
from pathlib import Path

# %%
# Parse arguments
parser = argparse.ArgumentParser(description='A tool for visualizing Astaroth slices',
epilog='''EXAMPLES:
    ./animate-slices.py --dims 128 128 --input *.slice          # Visualizes all slices
    
See Unix globbing for passing files/directories to the script more easily.
    For example:
        ??.sh matches two characters
        *.sh matches any number of characters
        [1-8] matches a character in range 1-8
        {1..16} expands to 1,2,3,...,16
        ?([0-9]) matches zero or one number
        [0-9]?([0-9]) matches one number and an optional second number
        ?[0-9] matches one character and one number
    ''',
    formatter_class=argparse.RawDescriptionHelpFormatter)


parser.add_argument('--dims', type=int, default=[32, 32], nargs=2, help='The dimensions of the computational domain in the order [y x] (note y first). Should be the same as global_nn defined in mhd.ini.')
parser.add_argument('--inputs', type=str, nargs='+', default=["*.slice"], help='Input files (supports globbing)')
parser.add_argument('--output-file', type=str, default='output.gif', help='Output filename')
parser.add_argument('--output-dir', type=str, default='output', help='Output directory')
args = parser.parse_args()

# Make directories
print(f"Creating output directory at {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)

# Read files
files = args.inputs
files = [os.path.join(os.getcwd(), file) for file in files]   # Relative path
files = [glob.glob(file) for file in files]                   # Glob
files = np.concatenate(files)                                 # Flatten
files.sort()
files

# Create animation
fig, ax = plt.subplots()
ims = []
for i, file in enumerate(files):
    arr = np.fromfile(
        file,
        dtype=np.double,
    )
    data = arr.reshape(args.dims)
    im = ax.imshow(data, animated=True)
    if i == 0:
        im = ax.imshow(data)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
ani.save(f'{args.output_dir}/{args.output_file}')
