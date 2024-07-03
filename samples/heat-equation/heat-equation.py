#!/usr/bin/env python3
# Copyright (C) 2023, Johannes Pekkilä
#
# This file is part of Astaroth.
#
# Astaroth is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Astaroth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.

# %%
import argparse
import numpy as np
import time
import pandas as pd

# %%
parser = argparse.ArgumentParser(description="A tool for generating benchmarks")
parser.add_argument(
    "--dims",
    type=int,
    nargs=3,
    default=[1024, 1024, 1],
    help="Dimensions of the computational domain",
)
parser.add_argument(
    "--device",
    type=str,
    default="gpu",
    choices=["cpu", "gpu"],
    help="The device used for the benchmarks",
)
parser.add_argument("--radius", type=int, default=1, help="Sets the stencil radius")
parser.add_argument(
    "--dtype",
    default="fp32",
    choices=["fp32", "fp64"],
    help="The precision used for the benchmarks",
)
parser.add_argument(
    "--library",
    required=True,
    choices=["pytorch", "tensorflow", "jax"],
    help="The underlying library used for benchmarking",
)
parser.add_argument(
    "--verify", type=int, default=1, help="Verify results with the model solution"
)
parser.add_argument("--jobid", type=int, default=0, help="Set the job id")
# parser.add_argument('--seed', type=int, default=12345, help='Set seed for the random number generator')
parser.add_argument(
    "--salt", type=int, default=12345, help="Set salt for the random number generator"
)
parser.add_argument(
    "--nsamples", type=int, default=100, help="The number of samples to benchmark"
)

jupyter = False
if jupyter:
    args = parser.parse_args(["--library", "tensorflow"])
else:
    args = parser.parse_args()

if args.dtype in "fp64":
    args.dtype = np.float64
else:
    args.dtype = np.float32

# Global variables
seed = int(args.salt + time.time() + args.jobid * time.time()) % (2**32 - 1)

# %%
# Model
import scipy
from scipy import ndimage


def convolve(input, weights):
    return scipy.ndimage.convolve(input, weights, mode="constant", output=args.dtype)


# %%
# Construct the input and weights
def get_input():
    # Grid setup
    box_size = 2 * np.pi
    nx, ny, nz = args.dims
    dx, dy, dz = box_size / np.array(args.dims)
    dt = 1e-3 * min(dx, min(dy, dz))

    np.random.seed(seed)
    r = args.radius
    l = 2 * r + 1
    if r == 0:
        coeffs = np.array([0])
    elif r == 1:
        coeffs = np.array([1, -2, 1])
    elif r == 2:
        coeffs = np.array([-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
    elif r == 3:
        coeffs = np.array([1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])
    elif r == 4:
        coeffs = np.array(
            [
                -1 / 560,
                8 / 315,
                -1 / 5,
                8 / 5,
                -205 / 72,
                8 / 5,
                -1 / 5,
                8 / 315,
                -1 / 560,
            ]
        )
    else:
        # import findiff # Takes a long time
        # coeffs = np.array(findiff.coefficients(deriv=2, acc=2*r, symbolic=True)[
        #                   'center']['coefficients'], dtype=args.dtype)
        # coeffs = np.random.random(l) # Note random coefficients
        # coeffs = np.linspace(-1, 1, l) # Note bogus coefficients
        coeffs = np.zeros(l)

    if nz > 1:
        input = 2 * np.random.random((nz, ny, nx)) - 1
        input = input.astype(args.dtype)

        kronecker = np.zeros((l, l, l), dtype=args.dtype)
        kronecker[r, r, r] = 1

        ddx = np.zeros((l, l, l), dtype=args.dtype)
        ddx[r, r, :] = dt * (1 / dx**2) * coeffs

        ddy = np.zeros((l, l, l), dtype=args.dtype)
        ddy[r, :, r] = dt * (1 / dy**2) * coeffs

        ddz = np.zeros((l, l, l), dtype=args.dtype)
        ddz[:, r, r] = dt * (1 / dz**2) * coeffs

        weights = kronecker + ddx + ddy + ddz
        # print(kronecker)
        # print(ddx)
        # print(ddy)
        # print(ddz)
        # print(weights)
    elif ny > 1:
        input = 2 * np.random.random((ny, nx)) - 1
        input = input.astype(args.dtype)

        kronecker = np.zeros((l, l), dtype=args.dtype)
        kronecker[r, r] = 1

        ddx = np.zeros((l, l), dtype=args.dtype)
        ddx[r, :] = dt * (1 / dx**2) * coeffs

        ddy = np.zeros((l, l), dtype=args.dtype)
        ddy[:, r] = dt * (1 / dy**2) * coeffs

        weights = kronecker + ddx + ddy
        # print(kronecker)
        # print(ddx)
        # print(ddy)
        # print(weights)
    elif nx > 1:
        input = 2 * np.random.random((nx)) - 1
        input = input.astype(args.dtype)

        kronecker = np.zeros((l), dtype=args.dtype)
        kronecker[r] = 1

        ddx = np.zeros((l), dtype=args.dtype)
        ddx[:] = dt * (1 / dx**2) * coeffs

        weights = kronecker + ddx
        # print(kronecker)
        # print(ddx)
        # print(weights)

    return input, weights


class Debug:
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


# %%
# Benchmark output
class Output:
    def __init__(self):
        self.df = pd.DataFrame(
            columns=[
                "kernel",
                "implementation",
                "maxthreadsperblock",
                "nx",
                "ny",
                "nz",
                "radius",
                "milliseconds",
                "tpbx",
                "tpby",
                "tpbz",
                "jobid",
                "seed",
                "iteration",
                "double_precision",
            ]
        )

    def record(self, milliseconds, iteration):
        row = {
            "kernel": "convolve",
            "implementation": args.library,
            "nx": args.dims[0],
            "ny": args.dims[1],
            "nz": args.dims[2],
            "radius": args.radius,
            "milliseconds": milliseconds,
            "jobid": args.jobid,
            "seed": seed,
            "iteration": iteration,
            "double_precision": int(args.dtype == np.float64),
        }
        self.df.loc[len(self.df.index)] = row

    def __del__(self):
        self.df.to_csv(f"heat-equation-python-{args.jobid}-{seed}.csv", index=False)


# %%
# Libraries
lib = None
if args.library in "pytorch":
    import torch
    import torch.utils.benchmark

    print(f"PyTorch version: {torch.__version__}")

    class Pytorch(torch.nn.Module):
        def __init__(self, device, dtype):
            super().__init__()
            self.device = "cpu" if device in "cpu" else "cuda"
            self.dtype = torch.double if dtype == np.float64 else torch.float

            print(f"Using {device} device")

            # Enable autotuning
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark_limit = 0  # Try every available algorithm

            # Disable tensor cores
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cudnn.allow_tf32 = False
            # torch.backends.cudnn.deterministic = True

            # Disable debugging APIs
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile = False
            torch.autograd.gradgradcheck = False

            # Print information
            print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")

            input, weights = get_input()
            self.weights = (
                torch.tensor(weights, dtype=self.dtype, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )

        def get_input(self):
            input, weights = get_input()
            input = (
                torch.tensor(input, dtype=self.dtype, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )  # .to(memory_format=torch.channels_last)
            weights = (
                torch.tensor(weights, dtype=self.dtype, device=self.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            return input, weights

        def pad(self, input):
            ndims = len(input.shape) - 2
            return torch.nn.functional.pad(
                input, (args.radius,) * 2 * ndims, mode="constant"
            )

        # @torch.compile
        # @torch.jit.script
        # @torch.no_grad()
        @torch.inference_mode
        def forward(self, input):
            if len(input.shape) == 5:
                return torch.nn.functional.conv3d(input, self.weights)
            elif len(input.shape) == 4:
                return torch.nn.functional.conv2d(input, self.weights)
            else:
                return torch.nn.functional.conv1d(input, self.weights)

        def benchmark_old(self, num_samples):
            output = Output()

            input, weights = self.get_input()
            for i in range(num_samples):
                input = self.pad(input)

                if self.device == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start = time.time()
                input = self.convolve(input, weights)
                if self.device == "cuda":
                    end.record()
                    torch.cuda.synchronize()
                    milliseconds = start.elapsed_time(end)
                else:
                    milliseconds = 1e3 * (time.time() - start)

                output.record(milliseconds, i)
                if i == num_samples - 1:
                    print(f"{milliseconds} ms")

        def benchmark_cuda(self, num_samples):
            input, weights = self.get_input()
            for i in range(num_samples):
                input = self.pad(input)

                if self.device == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start = time.time()
                input = self.convolve(input, weights)
                if self.device == "cuda":
                    end.record()
                    torch.cuda.synchronize()
                    milliseconds = start.elapsed_time(end)
                else:
                    milliseconds = 1e3 * (time.time() - start)
                if i == num_samples - 1:
                    print(f"{milliseconds} ms")

        def benchmark(self, num_samples):
            output = Output()

            input, weights = self.get_input()
            input = self.pad(input)
            traced = torch.jit.trace(self, input)

            timer = torch.utils.benchmark.Timer(
                stmt="traced.forward(input)",
                setup="from __main__ import Pytorch",
                globals={"input": input, "traced": traced},
            )

            for i in range(num_samples):
                measurement = timer.timeit(1)
                milliseconds = 1e3 * measurement.raw_times[0]
                output.record(milliseconds, i)
                if i >= num_samples - 10:
                    print(f"{milliseconds} ms")

    lib = Pytorch(args.device, args.dtype)
elif args.library in "tensorflow":
    import tensorflow as tf

    print(f"TensorFlow version: {tf.__version__}")

    class Tensorflow(Debug):
        def __init__(self, device, dtype):
            print(tf.sysconfig.get_build_info())
            devices = tf.config.list_physical_devices("GPU")
            # tf.config.set_visible_devices(devices[0], 'GPU') # Limit to one GPU
            print(devices)

            if device in "gpu":
                assert len(devices) > 0
            # tf.debugging.set_log_device_placement(True)

            self.device = "/device:CPU:0" if device in "cpu" else "/GPU:0"
            self.dtype = tf.float64 if dtype == np.float64 else tf.float32

        def get_input(self):
            input, weights = get_input()
            input = tf.constant(input, dtype=self.dtype)
            input = tf.expand_dims(input, 0)
            input = tf.expand_dims(input, -1)

            weights = tf.constant(weights, dtype=self.dtype)
            weights = tf.expand_dims(weights, -1)
            weights = tf.expand_dims(weights, -1)
            return input, weights

        def pad(self, input):
            ndims = len(input.shape) - 2
            paddings = tf.constant(
                [[0, 0]] + [[args.radius, args.radius]] * ndims + [[0, 0]]
            )
            return tf.pad(input, paddings, "CONSTANT")

        @tf.function(jit_compile=True)
        def convolve(self, input, weights):
            return tf.nn.convolution(input, weights)

        def benchmark_old_and_incorrect_sync(self, num_samples):
            output = Output()

            input, weights = self.get_input()
            for i in range(num_samples):
                input = self.pad(input)

                start = time.time()
                input = self.convolve(input, weights)
                milliseconds = 1e3 * (time.time() - start)

                output.record(milliseconds, i)
                if i == num_samples - 1:
                    print(f"{milliseconds} ms")

        def benchmark(self, num_samples):
            benchmark_output = Output()
            with tf.Graph().as_default() as graph:
                # Setup input and weights
                input, weights = self.get_input()
                with tf.compat.v1.Session(graph=graph) as sess:
                    benchmark = tf.test.Benchmark()
                    for i in range(num_samples):
                        measurement = benchmark.run_op_benchmark(
                            sess, self.convolve(input, weights), min_iters=1
                        )
                        benchmark_output.record(1e3 * measurement["wall_time"], i)

    lib = Tensorflow(args.device, args.dtype)
elif args.library in "jax":
    import jax
    from jax import config

    class Jax(Debug):
        def __init__(self, device, dtype):
            self.device = device
            if dtype == np.float64:
                config.update("jax_enable_x64", True)
                self.dtype = jax.numpy.float64
            else:
                self.dtype = jax.numpy.float32

        def get_input(self):
            input, weights = get_input()
            if self.device in "gpu":
                input = jax.device_put(input)
                weights = jax.device_put(weights)

            return input, weights

        def pad(self, input):
            return jax.numpy.pad(input, args.radius, mode="constant")

        # @jax.jit
        def convolve(self, input, weights):
            return jax.scipy.signal.convolve(
                input, weights, mode="valid", method="direct"
            )

        def benchmark(self, num_samples):
            output = Output()

            input, weights = self.get_input()
            convolve_jit = jax.jit(self.convolve)
            for i in range(num_samples):
                input = self.pad(input)

                start = time.time()
                input = convolve_jit(input, weights).block_until_ready()
                milliseconds = 1e3 * (time.time() - start)

                output.record(milliseconds, i)
                if i == num_samples - 10:
                    print(f"{milliseconds} ms")

    lib = Jax(args.device, args.dtype)

print(f"Using library {lib}")


# %%
# Check correctness
if args.verify:
    print("Verifying results...")
    input, weights = get_input()
    model = convolve(input, weights)
    input, weights = lib.get_input()
    if args.library == "jax":
        candidate = lib.convolve(lib.pad(input), weights).squeeze()
    elif args.library == "tensorflow":
        candidate = lib.convolve(lib.pad(input), weights).cpu().numpy().squeeze()
    else:  # Pytorch
        input = lib.pad(input)
        traced = torch.jit.trace(lib, input)
        candidate = traced.forward(input).cpu().numpy().squeeze()
    epsilon = (
        np.finfo(np.float64).eps
        if args.dtype == np.float64
        else np.finfo(np.float32).eps
    )
    epsilon *= 5
    correct = np.allclose(model, candidate, rtol=epsilon, atol=epsilon)
    print(f"Done. Results within rel/abs epsilon {epsilon}: {correct}")
    if not correct:
        diff = np.abs(model - candidate)
        print(f"Largest absolute error: {diff.max()}")
        print(f"Indices: {np.where(diff > epsilon)}")
    assert correct

# %%
# Benchmark
print("Benchmarking")
lib.benchmark(args.nsamples)
