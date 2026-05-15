#!/usr/bin/env python3

import os
import re
from itertools import product
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description="Tool to find missing segments of snapshots")
parser.add_argument("--field", type=str, required=True)
parser.add_argument("--snapshot_dir", type=str, required=True)
parser.add_argument("--snapshot_number", type=str, default="0")

args = parser.parse_args()


import re
import numpy as np

x_coords = []
y_coords = []
z_coords = []

with open("missing_segments.txt", "r") as f:
    for line in f:
        line = line.strip()

        # Extract all integers from filename
        nums = re.findall(r"\d+", line)

        # Take first three coordinates after "segment"
        x = int(nums[0])
        y = int(nums[1])
        z = int(nums[2])

        x_coords.append(x)
        y_coords.append(y)
        z_coords.append(z)

x_coords = np.array(x_coords)
y_coords = np.array(y_coords)
z_coords = np.array(z_coords)


pattern = re.compile(
    f"{args.field}-segment-(\d+)-(\d+)-(\d+)-{args.snapshot_number}\.mesh"
)

found = set()

# -------------------------------------------------
# Read files
# -------------------------------------------------

size_bytes = 0
for filename in os.listdir(args.snapshot_dir):
    m = pattern.match(filename)
    if m:
        coords = tuple(map(int, m.groups()))
        file_size = os.path.getsize(args.snapshot_dir +"/" + filename)
        size_bytes = max(file_size,size_bytes)
        #Files which have a corrupted size are not considered to exist
        if(size_bytes == file_size):
          found.add(coords)

if not found:
    print("No matching files found.")
    exit(1)

# -------------------------------------------------
# Discover coordinate ranges
# -------------------------------------------------

x_vals = sorted(set(c[0] for c in found))
y_vals = sorted(set(c[1] for c in found))
z_vals = sorted(set(c[2] for c in found))

nx = x_vals[1]
ny = y_vals[1]
nz = z_vals[1]

for i in range(len(x_coords)):
    x = x_coords[i]
    y = y_coords[i]
    z = z_coords[i]

    #res_filename = f"{args.field}-segment-{x}-{y}-{z}-{args.snapshot_number}.mesh"
    res_filename = args.snapshot_dir + "/" + f"{args.field}-segment-{x}-{y}-{z}-{args.snapshot_number}.mesh"
    print(f"Restoring {res_filename}")

    upper_neighbour_filename = f"{args.field}-segment-{x+nx}-{y}-{z}-{args.snapshot_number}.mesh"
    lower_neighbour_filename = f"{args.field}-segment-{x-nx}-{y}-{z}-{args.snapshot_number}.mesh"
    
    import numpy as np
    
    shape = (nx, ny, nz)
    
    upper_neighbour = np.fromfile(args.snapshot_dir +"/" + upper_neighbour_filename, dtype=np.float64)
    upper_neighbour = upper_neighbour.reshape(shape)
    
    lower_neighbour = np.fromfile(args.snapshot_dir +"/" + lower_neighbour_filename, dtype=np.float64)
    lower_neighbour = lower_neighbour.reshape(shape)
    
    w_forward = np.linspace(0.0, 1.0, nx)
    w_reverse = 1.0 - w_forward
    w_forward = w_forward[:,None,None]
    w_reverse = w_reverse[:,None,None]
    
    lower_neighbour_weighted = lower_neighbour*w_forward
    upper_neighbour_weighted = upper_neighbour*w_reverse
    upper_neighbour_flipped = upper_neighbour[::-1,:,:]
    lower_neighbour_flipped = lower_neighbour[::-1,:,:]
    
    reconstructed_data = lower_neighbour_flipped+upper_neighbour_flipped
    
    reconstructed_data.tofile(res_filename)

