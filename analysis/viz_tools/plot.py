#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import re
from pathlib import Path

parser = argparse.ArgumentParser(description='A tool for generating images from binary slices')
parser.add_argument('--input', type=str, nargs='+', required=True, help='A list of data files')
parser.add_argument('--output', type=str, default='output-slices-frames', help='Output dir to write images to')
parser.add_argument('--dims', type=int, nargs=2, help='The dimensions of a single data slice')
parser.add_argument('--dtype', type=str, default='double', help='The datatype of a single data element (default: double). Accepted values: numpy dtypes.')
parser.add_argument('--dpi', type=int, default=150, help='Set DPI of the output images')
args = parser.parse_args()

#TODO:mkdir call again, use paths library instead
#os.system(f'mkdir -p {args.output}')

dims_regex = re.compile(r'.*dims_(\d+)_(\d+).*')
for file in args.input:
    dims_match = dims_regex.match(file)
    dims = args.dims
    if dims_match:
        dims = [int(dims_match.group(1)), int(dims_match.group(2))]
        print(f"File {file}, grabbed dims from filename")
    print(f"File {file}, dims:{dims}")
    #Try to get dimensions out of tile
    if dims is not None:
        data = np.fromfile(file, args.dtype)
        data = data.reshape((-1, dims[1], dims[0]))
        for field in range(0, data.shape[0]):
            slice = data[field,:,:]
            plt.imshow(slice, cmap='plasma', interpolation='nearest')
            plt.colorbar()
            plt.savefig(f'{args.output}/{Path(file).stem}-field-{field}.png', dpi=args.dpi)
            plt.clf()
    else:
        print(f"Skipping {file}, no dimensions given")

#print(f'do `convert {args.output}/input_files.png animation.gif` to create an animation"')
