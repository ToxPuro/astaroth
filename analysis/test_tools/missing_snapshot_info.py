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

expected = set(product(x_vals, y_vals, z_vals))
missing = sorted(expected - found)

print(f"\nTotal expected: {len(expected)}")
print(f"Found:          {len(found)}")
print(f"Missing:        {len(missing)}")

# -------------------------------------------------
# Group missing by layer
# -------------------------------------------------

missing_by_x = defaultdict(list)
missing_by_y = defaultdict(list)
missing_by_z = defaultdict(list)

for coords in missing:
    x, y, z = coords

    missing_by_x[x].append(coords)
    missing_by_y[y].append(coords)
    missing_by_z[z].append(coords)

# -------------------------------------------------
# Layer statistics
# -------------------------------------------------

print("\n=== Missing by X layer ===")

layer_size = len(y_vals) * len(z_vals)

for x in x_vals:
    count = len(missing_by_x[x])

    percent = 100.0 * count / layer_size

    print(
        f"X={x:4d} : "
        f"{count:5d} missing "
        f"({percent:5.1f}%)"
    )

    # Detect nearly complete missing layer
    if percent == 100:
        print(">>> ENTIRE LAYER MISSING <<<")
    elif percent > 90:
        print(">>> ALMOST ENTIRE LAYER MISSING <<<")

print("\n=== Missing by Y layer ===")

layer_size = len(x_vals) * len(z_vals)

for y in y_vals:
    count = len(missing_by_y[y])

    percent = 100.0 * count / layer_size

    print(
        f"Y={y:4d} : "
        f"{count:5d} missing "
        f"({percent:5.1f}%)"
    )

    # Detect nearly complete missing layer
    if percent == 100:
        print(">>> ENTIRE LAYER MISSING <<<")
    elif percent > 90:
        print(">>> ALMOST ENTIRE LAYER MISSING <<<")

print("\n=== Missing by Z layer ===")

layer_size = len(x_vals) * len(y_vals)

for z in z_vals:
    count = len(missing_by_z[z])

    percent = 100.0 * count / layer_size

    print(
        f"Z={z:4d} : "
        f"{count:5d} missing "
        f"({percent:5.1f}%)"
    )

    # Detect nearly complete missing layer
    if percent == 100:
        print(">>> ENTIRE LAYER MISSING <<<")
    elif percent > 90:
        print(">>> ALMOST ENTIRE LAYER MISSING <<<")
# -------------------------------------------------
# Print actual missing files
# -------------------------------------------------

print("\n=== Missing files written to missing_segments.txt ===")
with open("missing_segments.txt","w") as f:
  for x, y, z in missing:
      print(
          f"{args.field}-segment-"
          f"{x}-{y}-{z}-{args.snapshot_number}.mesh", file=f
      )
