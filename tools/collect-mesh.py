#!/usr/bin/env python3
import argparse
import numpy as np

# Arguments
parser = argparse.ArgumentParser(
    description='A tool for collecting distributed mesh files into a monolithic snapshot',
    epilog='''EXAMPLES:
    # Generate run scripts and build directories
    genbenchmarks.py --task-type preprocess # Generate makefiles and benchmark scripts

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

parser.add_argument('--inputs', type=str, nargs='+', required=True, help='A list if distributed input files')
parser.add_argument('--local-dims', type=np.uint64, required=True, nargs=3, help='The dimensions of the local computational domain')
parser.add_argument('--global-dims', type=np.uint64, required=True, nargs=3, help='The dimensions of the global computational domain')
parser.add_argument('--output', type=str, required=True)
parser.add_argument('--i-have-backed-up-my-input-files', action="store_true", required=True, help="THIS SCRIPT IS NOT YET WELL TESTED. The script may corrupt the files passed as input and output. Pass this flag only if you acknowledge that your input files are safely backed up somewhere and you are running this script in a safe directory.")
args = parser.parse_args()

# Check argument validity
global_nn = np.array(args.global_dims)
local_nn = np.array(args.local_dims)
decomp = global_nn // local_nn
assert np.all(global_nn >= local_nn), "Global dims must be larger than local dims"
assert np.all(decomp * local_nn == global_nn), "Global dims must be divisible by local dims"

expected_input_count = np.prod(global_nn // local_nn)
assert len(args.inputs) == expected_input_count, "Invalid number of input files given for the global and local dims"

# Indexing functions
def to_linear(x, n):
    assert np.all(x >= 0) and np.all(x < n)
    a = np.cumprod(np.append([np.uint64(1)], n)[:-1])
    return np.dot(x, a)


def to_spatial(i, n):
    assert 0 <= i < np.prod(n)
    a = np.cumprod(np.append([np.uint64(1)], n)[:-1])
    return (i // a) % n

offsets = []
for i in range(np.prod(decomp)):
    offset = to_spatial(i, decomp) * local_nn
    offsets.append(offset)
assert len(offsets) == len(args.inputs)

# Generate dummy data
# label = "VTXBUF_UUX"
# modstep = "0"
# for offset in offsets:
#     with open(f'{label}-segment-{offset[0]}-{offset[1]}-{offset[2]}-{modstep}.mesh', 'wb') as fp:
#         #data = curr_block * np.ones(np.prod(local_nn), dtype=np.float64)
#         data = to_linear(offset // local_nn, decomp) * np.ones(np.prod(local_nn), dtype=np.float64)
#         fp.write(data)
# exit(0)


# Create the monolithic file and write the distributed snapshots into it
with open(args.output, 'wb') as output:
    for current_input_idx, input_path in enumerate(args.inputs):
        with open(input_path, 'rb') as input:
            # Parse offset from input
            _, _, x_offset, y_offset, z_offset, _ = input_path.split("-")
            offset = np.array([np.uint64(x_offset), np.uint64(y_offset), np.uint64(z_offset)])

            # Print status
            print(f"Processing input {input_path} ({np.round(current_input_idx / len(args.inputs), 2)}%)")

            # Check that the offset is within the expected offsets
            assert next((True for x in offsets if np.all(x == offset)), False), f"Offset {offset} not in the list of expected offsets"

            for k in range(local_nn[2]):
                for j in range(local_nn[1]):
                    local_offset = np.array([0, j, k], dtype=np.uint64)
                    local_idx = to_linear(local_offset, local_nn)
                    global_idx = to_linear(offset + local_offset, global_nn)

                    assert offset.dtype == np.uint64
                    assert local_nn.dtype == np.uint64
                    assert global_nn.dtype == np.uint64
                    assert local_offset.dtype == np.uint64
                    assert local_idx.dtype == np.uint64
                    assert global_idx.dtype == np.uint64
                    input.seek(np.uint64(8) * local_idx, 0)
                    output.seek(np.uint64(8) * global_idx, 0)

                    output.write(input.read(np.uint64(8)*local_nn[0]))

print(f"Complete (100%). Output written to {args.output}.")

# Check the monolithic file
# data = np.fromfile(args.output, dtype=np.float64).reshape(global_nn)
# for z in range(data.shape[2]):
#     print(data[z,:,:])
#print(data)