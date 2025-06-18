#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='A tool for checking whether two .csv files are close within numerical precision')
parser.add_argument('--files', type=str, nargs=2, required=True, help='.csv files to compare')
args = parser.parse_args()

print(args.files)

first = pd.read_csv(args.files[0])
second = pd.read_csv(args.files[1])
diff = first.compare(second).to_numpy()
first_diff = diff[:,0]
second_diff = diff[:,1]

largest_abs_error = abs(first_diff - second_diff).max()


machine_epsilon = np.finfo(np.float64).eps
atol = 5*machine_epsilon
are_close = np.allclose(first_diff, second_diff, rtol=machine_epsilon, atol=atol)
print(f'machine_epsilon: {machine_epsilon}')
print(f'atol: {atol}')
print(f'largest_abs_error: {largest_abs_error}')
print(f'OK? {are_close}')
assert(are_close)