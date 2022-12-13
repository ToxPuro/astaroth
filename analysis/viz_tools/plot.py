#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='A tool for generating images from binary slices')
parser.add_argument('--input', type=str, nargs='+', required=True, help='A list of data files')
parser.add_argument('--output', type=str, default='output-slices-frames', help='Output dir to write images to')
parser.add_argument('--dims', type=int, nargs=2, required=True, help='The dimensions of a single data slice')
parser.add_argument('--dtype', type=str, default='double', help='The datatype of a single data element (default: double). Accepted values: numpy dtypes.')
parser.add_argument('--plot', action='store_true', help='Plot the results instead of writing to disk.')
parser.add_argument('--dpi', type=int, default=150, help='Set DPI of the output images')
args = parser.parse_args()


os.system(f'mkdir -p {args.output}')
for file in args.input:
    data = np.fromfile(file, args.dtype)
    data = data.reshape((-1, args.dims[1], args.dims[0]))
    for field in range(0, data.shape[0]):
        slice = data[field,:,:]
        plt.imshow(slice, cmap='plasma', interpolation='nearest')
        plt.colorbar()
        if args.plot:
            plt.show()
        else:
            plt.savefig(f'{args.output}/{Path(file).stem}-field-{field}.png', dpi=args.dpi)
        plt.clf()

print(f'do `convert {args.output}/input_files.png animation.gif` to create an animation"')
