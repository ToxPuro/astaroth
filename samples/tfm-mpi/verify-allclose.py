#!/usr/bin/env python3
# %%
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='A tool for checking whether two .csv files are close within numerical precision')
parser.add_argument('--files', type=str, nargs=2, required=True, help='.csv files to compare')
args = parser.parse_args()

print(args.files)

first = pd.read_csv(args.files[0])
second = pd.read_csv(args.files[1])


for col in first.columns:
    diff = first[col].compare(second[col])
    if len(diff > 0): 
        print(f'Found diffs in column {col}')
        npdiff = diff.to_numpy()
        first_diff = npdiff[:,0]
        second_diff = npdiff[:,1]

        largest_abs_error = abs(first_diff - second_diff).max()
        machine_epsilon = np.finfo(np.float64).eps
        rtol  = 10*machine_epsilon
        atol = 10*machine_epsilon
        are_close = np.allclose(first_diff, second_diff, rtol=rtol, atol=atol)
        print(f'are_close: {are_close}')
        print(f'machine_epsilon: {machine_epsilon}')
        print(f'atol: {atol}')
        print(f'largest_abs_error: {largest_abs_error}')
        # assert(are_close)


# %%
import pandas as pd
import numpy as np
import argparse

#files = ['../../build/timeseries.csv', '../../build/timeseries.csv.nonsoca.turbulence.model']
#files = ['../../build/timeseries-1-0-default.csv', '../../build/timeseries.csv.model']
#files = ['../../build/timeseries-0-0-default.csv', '../../build/timeseries.csv']
files = ['../../build/timeseries.csv', '../../samples/tfm/model/laplace-nonsoca-turbulence/timeseries.csv']
#files = ['../../build/timeseries.csv', '../../samples/tfm/model/laplace-soca-roberts/timeseries.csv']

#files = ['../../build/timeseries.csv', '../../samples/tfm/model/laplace-soca-roberts/timeseries.csv']
#files = ['../../build/timeseries-11706636-0-strong.csv', '../../build/timeseries-11706631-0-strong.csv']
files = ['../../build/timeseries-0-0-default.csv', '../../samples/tfm/model/laplace-nonsoca-turbulence-incl-alfven-dt/timeseries.csv']

candidate = pd.read_csv(files[0])
model = pd.read_csv(files[1])

# Compare the last step
step_to_compare = candidate['step'].unique().max()

candidate = candidate[candidate['step'] == step_to_compare]
model = model[model['step'] == step_to_compare]

candidate.reset_index(drop=True, inplace=True)
model.reset_index(drop=True, inplace=True)

model.compare(candidate).max()