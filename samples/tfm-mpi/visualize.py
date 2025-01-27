#!/usr/bin/env python3
# %%
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import argparse
from pathlib import Path

# %%
# Parse arguments
parser = argparse.ArgumentParser(description='A tool for visualizing Astaroth output',
epilog='''EXAMPLES:
    ./visualize.py --input *.mesh          # Visualizes all meshes
    ./visualize.py --input *.profile       # Visualize all profiles
    ./visualize.py --input *01000*.profile # Visualize all profiles at step 1000
    
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


parser.add_argument('--dims', type=int, default=[32, 32, 32], nargs=3, help='The dimensions of the computational domain in the order [z y x] (note z first). Should be the same as global_nn defined in mhd.ini')
parser.add_argument('--slice-nz', type=int, default=-1, nargs=1, help='Position on the nz axis to slice on. If set to -1, chooses the middle slice.')
parser.add_argument('--nsteps', type=int, default=100, nargs=1, help='Maximum steps to visualize (defined as AC_simulation_nsteps in mhd.ini)')
parser.add_argument('--inputs', type=str, nargs='+', default=["*.mesh", "*.profile"], help='Input files to visualize (supports globbing)')
parser.add_argument('--output-dir', type=str, nargs='+', default="output", help='The output directory')
#parser.add_argument('--show-individual', action='store_true', help='Controls whether to write individual plots')
#parser.add_argument('--show-grouped', action='store_true', help='Controls whether to write grouped plots')

args = parser.parse_args()

# Set slice_nz to middle if not specified
if args.slice_nz < 0:
    args.slize_nz = int(args.dims[0]/2)

# Make directories
print(f"Creating output directory at {args.output_dir}")
os.makedirs(args.output_dir, exist_ok=True)

# %%
# Read files
files = args.inputs
files = [os.path.join(os.getcwd(), file) for file in files]   # Relative path
files = [glob.glob(file) for file in files]                   # Glob
files = np.concatenate(files)                                 # Flatten
files.sort()
files

# %%
# Visualize individual snapshots
snapshots = [file for file in files if file.endswith(".mesh")] # Filter snapshots
snapshots.sort()

for snapshot in snapshots:
    data = np.fromfile(
        snapshot,
        dtype=np.double,
    )
    data = data.reshape(args.dims)
    
    name = Path(snapshot).stem
    path = os.path.join(args.output_dir, f'{name}.png')
    
    print(f"Writing {path}")
    plt.imshow(data[args.slice_nz, :, :])
    plt.title(name)
    plt.savefig(path)
    plt.close()

# %%
# Visualize snapshots (grouped)
snapshots = [file for file in files if file.endswith(".mesh")] # Filter snapshots
snapshots.sort()

for step in range(args.nsteps):

    stepstr = str(step).zfill(12)
    current_snapshots = [snapshot for snapshot in snapshots if stepstr in snapshot]
    if len(current_snapshots) == 0:
        continue

    fig, axs = plt.subplots(int(np.ceil(len(current_snapshots)/3)), 3, layout="constrained")
    fig.set_figheight(15)
    fig.set_figwidth(20)

    for i, file in enumerate(current_snapshots):
        data = np.fromfile(
            file,
            dtype=np.double,
        )
        data = data.reshape(args.dims)
        
        name = Path(file).stem
        
        col = i % 3
        row = i // 3
        axs[row, col].imshow(data[args.slice_nz, :, :])
        axs[row, col].set_title(name)
    
    path = os.path.join(args.output_dir, f'snapshots-{stepstr}.png')
    print(f"Writing {path}")
    plt.savefig(path)
    plt.close()

# %%
# Visualize individual profiles
profiles = [file for file in files if file.endswith(".profile")] # Filter profiles
profiles.sort()

for profile in profiles:
    data = np.fromfile(
        profile,
        dtype=np.double,
    )
    
    name = Path(profile).stem
    path = os.path.join(args.output_dir, f'{name}.png')
    
    print(f"Writing {path}")
    plt.plot(data)
    plt.title(name)
    plt.savefig(path)
    plt.close()

# %%
# Visualize profiles (grouped)
profiles = [file for file in files if file.endswith(".profile")] # Filter profiles
profiles.sort()

for step in range(args.nsteps):
    
    stepstr = str(step).zfill(12)
    current_profiles = [profile for profile in profiles if stepstr in profile]

    if len(current_profiles) != 9 * 3:
        continue

    fig, axs = plt.subplots(9, 3, layout="constrained")
    fig.set_figheight(15)
    fig.set_figwidth(20)
    for i, file in enumerate(current_profiles):
        arr = np.fromfile(
            file,
            dtype=np.double,
        )
        col = i % 3
        row = i // 3

        name = os.path.basename(file)
        #print(f'Input {name}')
        
        axs[row, col].plot(arr)
        axs[row, col].set_title(name)

    path = os.path.join(args.output_dir, f'profiles-{stepstr}.png')
    
    print(f"Writing {path}")
    plt.savefig(path)
    plt.close()
