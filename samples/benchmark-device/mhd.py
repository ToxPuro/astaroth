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
import itertools
import scipy
import argparse
import numpy as np
import time
import pandas as pd
import findiff

# %%
parser = argparse.ArgumentParser(
    description='A tool for generating benchmarks')
parser.add_argument('--dims', type=int, nargs=3,
                    default=[128, 128, 128], help='Dimensions of the computational domain')
parser.add_argument('--device', type=str, default='gpu',
                    choices=['cpu', 'gpu'], help='The device used for the benchmarks')
parser.add_argument('--radius', type=int, default=1,
                    help='Sets the stencil radius')
parser.add_argument('--dtype', default='fp64',
                    choices=['fp32', 'fp64'], help='The precision used for the benchmarks')
parser.add_argument('--library', required=True, choices=[
                    'pytorch', 'tensorflow', 'jax'], help='The underlying library used for benchmarking')
parser.add_argument('--verify', type=int, default=1,
                    help='Verify results with the model solution')
parser.add_argument('--jobid', type=int, default=0, help='Set the job id')
# parser.add_argument('--seed', type=int, default=12345, help='Set seed for the random number generator')
parser.add_argument('--salt', type=int, default=12345,
                    help='Set salt for the random number generator')
parser.add_argument('--nsamples', type=int, default=100,
                    help='The number of samples to benchmark')
parser.add_argument('--visualize', action='store_true', help='Visualize the results')


jupyter = False
if jupyter:
    args = parser.parse_args(
        ['--library', 'pytorch', '--device', 'cpu', '--dims', '128', '128', '2', '--nsamples', '3'])
else:
    args = parser.parse_args()

# try: # Do not parse args if running in an interactive shell
#     get_ipython().__class__.__name__
#     args = parser.parse_args(
#         ['--library', 'pytorch', '--device', 'cpu', '--dims', '2', '2', '2', '--nsamples', '3'])
#     print('Running an interactive session. Arguments not parsed.')
# except NameError:
#     args = parser.parse_args()
#     print('Running a non-interactive session. Arguments parsed.')

if args.dtype in 'fp64':
    args.dtype = np.float64
else:
    args.dtype = np.float32

# Global variables
seed = int(args.salt + time.time() + args.jobid * time.time()) % (2**32-1)

random_scale = 1e-3
box_size = 2 * np.pi
nn = np.array(args.dims)
ds = box_size / nn
dt = 1e-1 * min(ds)
## Hydro
cs2_sound = 1.0
nu_visc = 1e-3
zeta = 0.01
## MHD
gamma = 0.5
lnrho0 = 1.3
cp_sound = 1
mu0 = 1.4
eta = 1e-2 #5e-5
lnT0 = 1.2
K_heatcond = 1e-3

field_names = ['lnrho', 'ux', 'uy', 'uz', 'ax', 'ay', 'az', 'ss']
stencil_names = ['value', 'ddx', 'ddy', 'ddz', 'd2dx2',
            'd2dy2', 'd2dz2', 'd2dxdy', 'd2dxdz', 'd2dydz']

# Derived variables
ndims = len(np.empty(args.dims).squeeze().shape)
assert(ndims == 3) # Hardcoded 3D in forward() funtions
field_indices = {field: i for i, field in enumerate(field_names)}
stencil_indices = {stencil: i for i, stencil in enumerate(stencil_names)}


# %%
# Construct the input and weights
def get_coeffs(deriv, axis=0):
    assert(axis < ndims)

    order = 2*args.radius
    # Coefficients
    if deriv == 0:
        coeffs = np.zeros(2*args.radius + 1, dtype=args.dtype)
        coeffs[args.radius] = 1
    else:
        coeffs = np.array(findiff.coefficients(deriv=deriv, acc=order, symbolic=True)[
                          'center']['coefficients'], dtype=args.dtype)
    l = len(coeffs)
    r = int(np.rint((l-1)/2))
    tensor = np.zeros((l, l, l), dtype=args.dtype)

    # Axis
    if axis == 0:
        tensor[r, r, :] = coeffs
    elif axis == 1:
        tensor[r, :, r] = coeffs
    else:
        tensor[:, r, r] = coeffs

    # Dims
    if ndims == 1:
        return tensor[r, r, :]
    elif ndims == 2:
        return tensor[r, :, :]
    else:
        return tensor


def get_mixed_coeffs(deriv, axis0=0, axis1=-1):
    assert(axis0 < ndims)
    assert(axis1 < ndims)
    assert(axis0 != axis1)
    assert(axis0 < axis1)

    order = 2*args.radius
    coeffs = np.array(findiff.coefficients(deriv=deriv, acc=order, symbolic=True)[
                      'center']['coefficients'], dtype=args.dtype)
    l = len(coeffs)
    r = int(np.rint((l-1)/2))
    a = np.zeros((l, l, l), dtype=args.dtype)

    if axis0 == 0:
        if axis1 == 1:
            a[r, :, :] = coeffs
            for i in range(l):
                a[r, i, :] *= coeffs[i]
        else:
            a[:, r, :] = coeffs
            for i in range(l):
                a[i, r, :] *= coeffs[i]
    else:
        a[:, :, r] = coeffs
        for i in range(l):
            a[i, :, r] *= coeffs[i]

    # Dims
    if ndims == 1:
        return a[r, r, :]
    elif ndims == 2:
        return a[r, :, :]
    else:
        return a


def get_input():
    np.random.seed(seed)
    fields = []
    for i in range(len(field_names)):
        fields.append(random_scale * (2 * np.random.random((nn[2], nn[1], nn[0])) - 1).squeeze())

    fields[field_indices['lnrho']] = np.random.normal(loc=0.0, scale=1e-1, size=(nn[2], nn[1], nn[0]))
    fields[field_indices['ux']] = np.random.normal(loc=0.0, scale=1e-12, size=(nn[2], nn[1], nn[0]))
    fields[field_indices['uy']] = np.random.normal(loc=0.0, scale=1e-12, size=(nn[2], nn[1], nn[0]))
    fields[field_indices['uz']] = np.random.normal(loc=0.0, scale=1e-12, size=(nn[2], nn[1], nn[0]))
    fields[field_indices['ax']] = np.random.normal(loc=0.0, scale=1e-8, size=(nn[2], nn[1], nn[0]))
    fields[field_indices['ay']] = np.random.normal(loc=0.0, scale=1e-8, size=(nn[2], nn[1], nn[0]))
    fields[field_indices['az']] = np.random.normal(loc=0.0, scale=1e-8, size=(nn[2], nn[1], nn[0]))
    fields[field_indices['ss']] = np.random.normal(loc=0.0, scale=1e-8, size=(nn[2], nn[1], nn[0]))
    # fields[field_indices['ax']] = np.zeros((nn[2], nn[1], nn[0])) #np.random.normal(loc=0.0, scale=1e-15, size=(nn[2], nn[1], nn[0]))
    # fields[field_indices['ay']] = np.zeros((nn[2], nn[1], nn[0])) #np.random.normal(loc=0.0, scale=1e-15, size=(nn[2], nn[1], nn[0]))
    # fields[field_indices['az']] = np.zeros((nn[2], nn[1], nn[0])) #np.random.normal(loc=0.0, scale=1e-15, size=(nn[2], nn[1], nn[0]))
    # fields[field_indices['ss']] = np.zeros((nn[2], nn[1], nn[0])) #np.random.normal(loc=1.0, scale=1e-8, size=(nn[2], nn[1], nn[0]))


    return np.stack(fields)


def get_weights():

    # Kronecker
    kronecker = get_coeffs(0)
    stencils = [kronecker]

    # First derivatives
    for ax in range(ndims):
        stencils.append((1/ds[ax]) * get_coeffs(1, ax))

    # Second derivatives
    for ax in range(ndims):
        stencils.append((1/ds[ax]**2) * get_coeffs(2, ax))

    # Mixed derivatives
    for ax1 in range(ndims):
        for ax0 in range(ndims):
            if ax0 >= ax1:
                continue
            stencils.append(
                (1/(ds[ax0]*ds[ax1]) * get_mixed_coeffs(1, ax0, ax1)))

    return np.stack(stencils)


def get_kronecker():
    return np.expand_dims(get_coeffs(0), 0)


def get_ddx():
    stencils = []
    for ax in range(ndims):
        stencils.append((1/ds[ax]) * get_coeffs(1, ax))
    return np.stack(stencils)


def get_d2dx2():
    stencils = []
    for ax in range(ndims):
        stencils.append((1/ds[ax]**2) * get_coeffs(2, ax))
    return np.stack(stencils)


def get_d2dxdy():
    stencils = []
    for ax0 in range(ndims):
        for ax1 in range(ndims):
            if ax0 >= ax1:
                continue
            stencils.append(
                (1/(ds[ax0]*ds[ax1]) * get_mixed_coeffs(1, ax0, ax1)))
    return np.stack(stencils)

# %%
# Model
def convolve(input, weights):
    # Flip weights: do cross-correlation like ML libs instead
    # or just use the correlate() directly
    #weights = np.flip(np.flip(np.flip(weights, axis=-1), axis=-2), axis=-3)

    tensor = None
    for field in range(len(input)):
        convolutions = []
        for stencil in range(len(weights)):
            convolutions.append(scipy.ndimage.correlate(
                input[field], weights[stencil], mode='constant', output=args.dtype))
        convolutions = np.expand_dims(np.stack(convolutions), 0)
        if tensor is not None:
            tensor = np.vstack((tensor, convolutions))
        else:
            tensor = convolutions
    return tensor


def dot(a, b):
    return np.sum(a * b, axis=0)

# Expects a tensor of shape (fields, first derivatives, ...) as input
# Assumes that f.ex. a[0,1] is ddy f_x
def curl(a):
    x = a[2,1] - a[1,2]
    y = a[0,2] - a[2,0]
    z = a[1,0] - a[0,1]
    return np.stack([x, y, z])

def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return np.stack([x, y, z])


def forward(input):

    # Inputs
    ## Hydro
    uu = input[[field_indices['ux'],
                        field_indices['uy'], field_indices['uz']]]
    lnrho = input[field_indices['lnrho']:field_indices['lnrho']+1]
    ## MHD
    aa = input[[field_indices['ax'],
                field_indices['ay'], field_indices['az']]]
    ss = input[field_indices['ss']:field_indices['ss']+1]

    # Convolutions
    ## Hydro
    grad_lnrho, hessian_lnrho = np.split(convolve(lnrho, np.concatenate([get_ddx(), get_d2dx2()])), 2, axis=1)
    grad_uu, hessian_uu, mixed_uu = np.split(
        convolve(uu, np.concatenate([get_ddx(), get_d2dx2(), get_d2dxdy()])), 3, axis=1)
    ## MHD
    grad_ss, hessian_ss = np.split(convolve(ss, np.concatenate([get_ddx(), get_d2dx2()])), 2, axis=1)
    grad_aa, hessian_aa, mixed_aa = np.split(
        convolve(aa, np.concatenate([get_ddx(), get_d2dx2(), get_d2dxdy()])), 3, axis=1)

    # Flatten
    ## Hydro
    uu = np.reshape(uu, (3, -1))
    lnrho = np.reshape(lnrho, (-1))
    grad_lnrho = np.reshape(grad_lnrho.squeeze(), (3, -1))
    hessian_lnrho = np.reshape(hessian_lnrho.squeeze(), (3, -1))
    grad_uu = np.reshape(grad_uu, (3, 3, -1))
    hessian_uu = np.reshape(hessian_uu, (3, 3, -1))
    mixed_uu = np.reshape(mixed_uu, (3, 3, -1))

    ## MHD
    aa = np.reshape(aa, (3, -1))
    ss = np.reshape(ss, (-1))
    grad_ss = np.reshape(grad_ss.squeeze(), (3, -1))
    hessian_ss = np.reshape(hessian_ss.squeeze(), (3, -1))
    grad_aa = np.reshape(grad_aa, (3, 3, -1))
    hessian_aa = np.reshape(hessian_aa, (3, 3, -1))
    mixed_aa = np.reshape(mixed_aa, (3, 3, -1))
    cs2 = cs2_sound * np.exp((gamma/cp_sound) * ss + (gamma - 1) * (lnrho - lnrho0))
    #cs2 = np.reshape(cs2, (-1))

    # More complex operations
    ## Hydro
    laplace_ss = np.sum(hessian_ss, axis=0)
    laplace_lnrho = np.sum(hessian_ss, axis=0)
    laplace_uu = np.sum(hessian_uu, axis=1)
    div_uu = grad_uu[0, 0] + grad_uu[1, 1] + grad_uu[2, 2]
    grad_div_ux = hessian_uu[0, 0] + mixed_uu[1, 0] + mixed_uu[2, 1]
    grad_div_uy = mixed_uu[0, 0] + hessian_uu[1, 1] + mixed_uu[2, 2]
    grad_div_uz = mixed_uu[0, 1] + mixed_uu[1, 2] + hessian_uu[2, 2]
    ## Traceless rate-of-strain stress tensor
    S00 = (2/3) * grad_uu[0, 0] - (1/3) * (grad_uu[1,1] + grad_uu[2, 2])
    S01 = (1/2) * (grad_uu[0, 1] + grad_uu[1, 0])
    S02 = (1/2) * (grad_uu[0,2] + grad_uu[2,0])
    S0 = np.stack([S00, S01, S02])
    
    S10 = S01
    S11 = (2/3) * grad_uu[1, 1] - (1/3) * (grad_uu[0,0] + grad_uu[2, 2])
    S12 = (1/2) * (grad_uu[1, 2] + grad_uu[2, 1])
    S1 = np.stack([S10, S11, S12])

    S20 = S02
    S21 = S12
    S22 = (2/3) * grad_uu[2, 2] - (1/3) * (grad_uu[0,0] + grad_uu[1, 1])
    S2 = np.stack([S20, S21, S22])
    
    S = np.concatenate([S0, S1, S2])
    ## MHD
    laplace_aa = np.sum(hessian_aa, axis=1)
    div_aa = grad_aa[0, 0] + grad_aa[1, 1] + grad_aa[2, 2]
    grad_div_ax = hessian_aa[0, 0] + mixed_aa[1, 0] + mixed_aa[2, 1]
    grad_div_ay = mixed_aa[0, 0] + hessian_aa[1, 1] + mixed_aa[2, 2]
    grad_div_az = mixed_aa[0, 1] + mixed_aa[1, 2] + hessian_aa[2, 2]
    grad_div_aa = np.stack([grad_div_ax, grad_div_ay, grad_div_az])
    j = (1/mu0) * (grad_div_aa - laplace_aa)
    B = curl(grad_aa)
    inv_rho = 1 / np.exp(lnrho)
    lnT = lnT0 + (gamma / cp_sound) * ss + (gamma - 1) * (lnrho - lnrho0)
    inv_pT = 1 / (np.exp(lnrho) * np.exp(lnT))
    j_cross_B = cross(j, B)
    entropy_rhs = (0) - (0) + eta * mu0 * dot(j, j)\
     + 2 * np.exp(lnrho) * nu_visc * np.sum(S, axis=0)\
     + zeta * np.exp(lnrho) * div_uu * div_uu
    ## Heat conduction
    grad_ln_chi = -grad_lnrho
    first_term = gamma * (1/cp_sound) * laplace_ss + (gamma - 1) * laplace_lnrho
    second_term = gamma * (1/cp_sound) * grad_ss + (gamma - 1.) * grad_lnrho
    third_term = gamma * ((1/cp_sound) * grad_ss + grad_lnrho) + grad_ln_chi
    chi = K_heatcond / (np.exp(lnrho) * cp_sound)
    heat_conduction = cp_sound * chi * (first_term + dot(second_term, third_term))


    # Equations
    ## Hydro
    continuity = -dot(uu, grad_lnrho) - div_uu
    momx = - dot(uu, grad_uu[0])\
        - cs2 * ((1/cp_sound) * grad_ss[0] + grad_lnrho[0])\
        + nu_visc * (laplace_uu[0] + (1/3) * grad_div_ux + 2 * dot(S0, grad_lnrho))\
        + zeta * grad_div_ux\
        + inv_rho * j_cross_B[0]
    momy = - dot(uu, grad_uu[1])\
        - cs2 * ((1/cp_sound) * grad_ss[1] + grad_lnrho[1])\
        + nu_visc * (laplace_uu[1] + (1/3) * grad_div_uy + 2 * dot(S1, grad_lnrho))\
        + zeta * grad_div_uy\
        + inv_rho * j_cross_B[1]
    momz = - dot(uu, grad_uu[2])\
        - cs2 * ((1/cp_sound) * grad_ss[2] + grad_lnrho[2])\
        + nu_visc * (laplace_uu[2] + (1/3) * grad_div_uz + 2 * dot(S2, grad_lnrho))\
        + zeta * grad_div_uz\
        + inv_rho * j_cross_B[2]
    ## MHD
    ind = cross(uu, B) + eta * laplace_aa
    entropy = - dot(uu, grad_ss) + inv_pT * entropy_rhs + heat_conduction

    return np.reshape(np.stack([continuity, momx, momy, momz, ind[0], ind[1], ind[2], entropy]), (len(field_names), nn[2], nn[1], nn[0]))


# %%
# Debug
class Debug:
    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    
# %%
# Benchmark output


class Output:
    def __init__(self):
        self.df = pd.DataFrame(columns=['kernel',
                                        'implementation',
                                        'maxthreadsperblock',
                                        'nx', 'ny', 'nz', 'radius',
                                        'milliseconds',
                                        'tpbx', 'tpby', 'tpbz',
                                        'jobid', 'seed', 'iteration', 'double_precision'])

    def record(self, milliseconds, iteration):
        row = {'kernel': 'convolve',
               'implementation': args.library,
               'nx': args.dims[0],
               'ny': args.dims[1],
               'nz': args.dims[2],
               'radius': args.radius,
               'milliseconds': milliseconds,
               'jobid': args.jobid,
               'seed': seed,
               'iteration': iteration,
               'double_precision': int(args.dtype == np.float64)}
        self.df.loc[len(self.df.index)] = row

    def __del__(self):
        self.df.to_csv(f'nonlinear-mhd-python-{args.jobid}-{seed}.csv', index=False)


# %%
# Libraries
lib = None
if args.library in 'pytorch':
    import torch
    import torch.utils.benchmark
    print(torch.__version__)

    class Pytorch(Debug):
        def get_weights(self, weights):
            weights = torch.tensor(
                weights, dtype=self.dtype, device=self.device).unsqueeze(1)
            return weights

        def __init__(self, device, dtype):
            self.name = 'pytorch'
            self.device = 'cpu' if device in 'cpu' else 'cuda'
            self.dtype = torch.float64 if dtype == np.float64 else torch.float32
            torch.set_default_dtype(self.dtype)
            #self.dtype = torch.double if dtype == np.float64 else torch.float

            print(f'Using {device} device')

            # Enable autotuning
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.benchmark_limit = 0 # Try every available algorithm

            # Disable tensor cores
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            torch.backends.cudnn.allow_tf32 = False
            #torch.backends.cudnn.deterministic = True

            # Disable debugging APIs
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile = False
            torch.autograd.gradgradcheck = False

            # Print information
            print(f'cuDNN available: {torch.backends.cudnn.is_available()}')
            print(f'cuDNN version: {torch.backends.cudnn.version()}')

            # Setup weights
            self.kronecker = self.get_weights(get_kronecker())
            self.ddx = self.get_weights(get_ddx())
            self.d2dx2 = self.get_weights(get_d2dx2())
            self.d2dxdy = self.get_weights(get_d2dxdy())
            self.weights = self.get_weights(get_weights())

        def get_input(self):
            input = get_input()
            input = torch.tensor(input, dtype=self.dtype,
                                 device=self.device).unsqueeze(1)
            # .to(memory_format=torch.channels_last)
            return input

        def pad(self, input):
            return torch.nn.functional.pad(input, (args.radius,) * 2 * ndims, mode='constant')
            ##return torch.nn.functional.pad(input, (args.radius,) * 2 * ndims, mode='circular') #TODO NOTE DEBUG

        def convolve(self, input, weights):
            if (len(input.shape) == 5):
                return torch.nn.functional.conv3d(input, weights)
            elif (len(input.shape) == 4):
                return torch.nn.functional.conv2d(input, weights)
            else:
                return torch.nn.functional.conv1d(input, weights)

        #def convolve_fft(self, input, weights):
        #    fr_signal = torch.fft.rfftn

        # Element-wise dot product
        def dot(self, a, b):
            return torch.sum(a * b, dim=0)

        # Expects a tensor of shape (fields, first derivatives, ...) as input
        # Assumes that f.ex. a[0,1] is ddy f_x
        def curl(self, a):
            x = a[2,1] - a[1,2]
            y = a[0,2] - a[2,0]
            z = a[1,0] - a[0,1]
            return torch.stack([x, y, z])

        def cross(self, a, b):
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return torch.stack([x, y, z])

        #@torch.compile(options={'max-autotune': True})
        @torch.compile(mode='max-autotune', fullgraph=True)
        @torch.no_grad() # Note: not supported with jit
        #@torch.jit.script
        def forward(self, input):
            # Inputs
            ## Hydro
            uu = input[[field_indices['ux'],
                                field_indices['uy'], field_indices['uz']]]
            lnrho = input[field_indices['lnrho']:field_indices['lnrho']+1]
            ## MHD
            aa = input[[field_indices['ax'],
                        field_indices['ay'], field_indices['az']]]
            ss = input[field_indices['ss']:field_indices['ss']+1]

            # Convolutions
            ## Hydro
            grad_lnrho, hessian_lnrho = torch.split(self.convolve(lnrho, torch.concat([self.ddx, self.d2dx2])), 3, dim=1)
            grad_uu, hessian_uu, mixed_uu = torch.split(
                self.convolve(uu, torch.concat([self.ddx, self.d2dx2, self.d2dxdy])), 3, dim=1)
            ## MHD
            grad_ss, hessian_ss = torch.split(self.convolve(ss, torch.concat([self.ddx, self.d2dx2])), 3, dim=1)
            grad_aa, hessian_aa, mixed_aa = torch.split(
                self.convolve(aa, torch.concat([self.ddx, self.d2dx2, self.d2dxdy])), 3, dim=1)

            # Squeeze input and remove padding
            uu = uu[:, :, args.radius:-args.radius,
                    args.radius:-args.radius, args.radius:-args.radius].squeeze()
            lnrho = lnrho[:, :, args.radius:-args.radius, args.radius:-
                          args.radius, args.radius:-args.radius].squeeze()
            aa = aa[:, :, args.radius:-args.radius,
                    args.radius:-args.radius, args.radius:-args.radius].squeeze()
            ss = ss[:, :, args.radius:-args.radius, args.radius:-
                          args.radius, args.radius:-args.radius].squeeze()

            # Flatten
            ## Hydro
            uu = torch.reshape(uu, (3, -1))
            lnrho = torch.flatten(lnrho)
            grad_lnrho = torch.reshape(grad_lnrho.squeeze(), (3, -1))
            hessian_lnrho = torch.reshape(hessian_lnrho.squeeze(), (3, -1))
            grad_uu = torch.reshape(grad_uu, (3, 3, -1))
            hessian_uu = torch.reshape(hessian_uu, (3, 3, -1))
            mixed_uu = torch.reshape(mixed_uu, (3, 3, -1))

            ## MHD
            aa = torch.reshape(aa, (3, -1))
            ss = torch.flatten(ss)
            grad_ss = torch.reshape(grad_ss.squeeze(), (3, -1))
            hessian_ss = torch.reshape(hessian_ss.squeeze(), (3, -1))
            grad_aa = torch.reshape(grad_aa, (3, 3, -1))
            hessian_aa = torch.reshape(hessian_aa, (3, 3, -1))
            mixed_aa = torch.reshape(mixed_aa, (3, 3, -1))
            cs2 = cs2_sound * torch.exp((gamma/cp_sound) * ss + (gamma - 1) * (lnrho - lnrho0))
            cs2 = torch.flatten(cs2)

            # More complex operations
            ## Hydro
            laplace_ss = torch.sum(hessian_ss, dim=0)
            laplace_lnrho = torch.sum(hessian_ss, dim=0)
            laplace_uu = torch.sum(hessian_uu, dim=1)
            div_uu = grad_uu[0, 0] + grad_uu[1, 1] + grad_uu[2, 2]
            grad_div_ux = hessian_uu[0, 0] + mixed_uu[1, 0] + mixed_uu[2, 1]
            grad_div_uy = mixed_uu[0, 0] + hessian_uu[1, 1] + mixed_uu[2, 2]
            grad_div_uz = mixed_uu[0, 1] + mixed_uu[1, 2] + hessian_uu[2, 2]
            ## Traceless rate-of-strain stress tensor
            S00 = (2/3) * grad_uu[0, 0] - (1/3) * (grad_uu[1,1] + grad_uu[2, 2])
            S01 = (1/2) * (grad_uu[0, 1] + grad_uu[1, 0])
            S02 = (1/2) * (grad_uu[0,2] + grad_uu[2,0])
            S0 = torch.stack([S00, S01, S02])
            
            S10 = S01
            S11 = (2/3) * grad_uu[1, 1] - (1/3) * (grad_uu[0,0] + grad_uu[2, 2])
            S12 = (1/2) * (grad_uu[1, 2] + grad_uu[2, 1])
            S1 = torch.stack([S10, S11, S12])

            S20 = S02
            S21 = S12
            S22 = (2/3) * grad_uu[2, 2] - (1/3) * (grad_uu[0,0] + grad_uu[1, 1])
            S2 = torch.stack([S20, S21, S22])
            
            S = torch.concat([S0, S1, S2])
            ## MHD
            laplace_aa = torch.sum(hessian_aa, dim=1)
            div_aa = grad_aa[0, 0] + grad_aa[1, 1] + grad_aa[2, 2]
            grad_div_ax = hessian_aa[0, 0] + mixed_aa[1, 0] + mixed_aa[2, 1]
            grad_div_ay = mixed_aa[0, 0] + hessian_aa[1, 1] + mixed_aa[2, 2]
            grad_div_az = mixed_aa[0, 1] + mixed_aa[1, 2] + hessian_aa[2, 2]
            grad_div_aa = torch.stack([grad_div_ax, grad_div_ay, grad_div_az])
            j = (1/mu0) * (grad_div_aa - laplace_aa)
            B = self.curl(grad_aa)
            inv_rho = 1 / torch.exp(lnrho)
            lnT = lnT0 + (gamma / cp_sound) * ss + (gamma - 1) * (lnrho - lnrho0)
            inv_pT = 1 / (torch.exp(lnrho) * torch.exp(lnT))
            j_cross_B = self.cross(j, B)
            entropy_rhs = (0) - (0) + eta * mu0 * self.dot(j, j)\
            + 2 * torch.exp(lnrho) * nu_visc * torch.sum(S, dim=0)\
            + zeta * torch.exp(lnrho) * div_uu * div_uu
            ## Heat conduction
            grad_ln_chi = -grad_lnrho
            first_term = gamma * (1/cp_sound) * laplace_ss + (gamma - 1) * laplace_lnrho
            second_term = gamma * (1/cp_sound) * grad_ss + (gamma - 1.) * grad_lnrho
            third_term = gamma * ((1/cp_sound) * grad_ss + grad_lnrho) + grad_ln_chi
            chi = K_heatcond / (torch.exp(lnrho) * cp_sound)
            heat_conduction = cp_sound * chi * (first_term + self.dot(second_term, third_term))


            # Equations
            ## Hydro
            continuity = -self.dot(uu, grad_lnrho) - div_uu
            momx = - self.dot(uu, grad_uu[0])\
                - cs2 * ((1/cp_sound) * grad_ss[0] + grad_lnrho[0])\
                + nu_visc * (laplace_uu[0] + (1/3) * grad_div_ux + 2 * self.dot(S0, grad_lnrho))\
                + zeta * grad_div_ux\
                + inv_rho * j_cross_B[0]
            momy = - self.dot(uu, grad_uu[1])\
                - cs2 * ((1/cp_sound) * grad_ss[1] + grad_lnrho[1])\
                + nu_visc * (laplace_uu[1] + (1/3) * grad_div_uy + 2 * self.dot(S1, grad_lnrho))\
                + zeta * grad_div_uy\
                + inv_rho * j_cross_B[1]
            momz = - self.dot(uu, grad_uu[2])\
                - cs2 * ((1/cp_sound) * grad_ss[2] + grad_lnrho[2])\
                + nu_visc * (laplace_uu[2] + (1/3) * grad_div_uz + 2 * self.dot(S2, grad_lnrho))\
                + zeta * grad_div_uz\
                + inv_rho * j_cross_B[2]
            ## MHD
            ind = self.cross(uu, B) + eta * laplace_aa
            entropy = - self.dot(uu, grad_ss) + inv_pT * entropy_rhs + heat_conduction

            return torch.stack([continuity, momx, momy, momz, ind[0], ind[1], ind[2], entropy]).view(-1, 1, nn[2], nn[1], nn[0])

        def benchmark_cuda(self, num_samples):
            output = Output()

            input = self.get_input()
            weights = self.get_weights(get_weights())
            for i in range(num_samples):
                input = self.pad(input)

                if self.device == 'cuda':
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                else:
                    start = time.time()
                input = self.forward(input, weights)
                if self.device == 'cuda':
                    end.record()
                    torch.cuda.synchronize()
                    milliseconds = start.elapsed_time(end)
                else:
                    milliseconds = 1e3 * (time.time() - start)

                output.record(milliseconds, i)
                if i >= num_samples-10:
                    print(f'{milliseconds} ms')

        def benchmark(self, num_samples):
            output = Output()

            input = self.pad(self.get_input())
            weights = self.get_weights(get_weights())

            timer = torch.utils.benchmark.Timer(
                stmt='self.forward(input)',
                setup='from __main__ import Pytorch',
                globals={'input': input, 'self': self}
            )

            for i in range(num_samples):
                measurement = timer.timeit(1)
                milliseconds = 1e3 * measurement.raw_times[0]
                output.record(milliseconds, i)
                if i >= num_samples-10:
                    print(f'{milliseconds} ms')

    lib = Pytorch(args.device, args.dtype)
elif args.library in 'tensorflow':
    import tensorflow as tf

    class Tensorflow(Debug):

        def get_weights(self, weights):
            weights = tf.constant(weights, dtype=self.dtype)
            weights = tf.transpose(weights, [1, 2, 3, 0])
            weights = tf.expand_dims(weights, ndims)
            return weights

        def __init__(self, device, dtype):
            self.name = 'tensorflow'

            print(tf.sysconfig.get_build_info())
            devices = tf.config.list_physical_devices('GPU')
            # tf.config.set_visible_devices(devices[0], 'GPU') # Limit to one GPU
            print(devices)

            if device in 'gpu':
                assert(len(devices) > 0)
            # tf.debugging.set_log_device_placement(True)

            self.device = '/device:CPU:0' if device in 'cpu' else '/GPU:0'
            self.dtype = tf.float64 if dtype == np.float64 else tf.float32

            # Warning: potential performance degradation due to this
            # Also @functions are not traced in this case
            # tf.config.run_functions_eagerly(True)

            # Setup weights
            self.kronecker = self.get_weights(get_kronecker())
            self.ddx = self.get_weights(get_ddx())
            self.d2dx2 = self.get_weights(get_d2dx2())
            self.d2dxdy = self.get_weights(get_d2dxdy())
            self.weights = self.get_weights(get_weights())

        def get_input(self):
            input = get_input()
            input = tf.constant(input, dtype=self.dtype)
            input = tf.expand_dims(input, -1)
            return input

        def pad(self, input):
            ndims = len(input.shape)-2
            paddings = tf.constant(
                [[0, 0]] + [[args.radius, args.radius]] * ndims + [[0, 0]])
            return tf.pad(input, paddings, 'CONSTANT')

        def convolve(self, input, weights):
            return tf.nn.convolution(input, weights)

        # Element-wise dot product
        def dot(self, a, b):
            return tf.reduce_sum(a * b, axis=0)

        # Expects a tensor of shape (fields, first derivatives, ...) as input
        # Assumes that f.ex. a[0,1] is ddy f_x
        def curl(self, a):
            x = a[2,1] - a[1,2]
            y = a[0,2] - a[2,0]
            z = a[1,0] - a[0,1]
            return tf.stack([x, y, z])

        def cross(self, a, b):
            x = a[1] * b[2] - a[2] * b[1]
            y = a[2] * b[0] - a[0] * b[2]
            z = a[0] * b[1] - a[1] * b[0]
            return tf.stack([x, y, z])

        @tf.function(jit_compile=True)
        def forward(self, input):
            # Inputs
            ## Hydro
            uu = tf.gather(input, indices=[field_indices['ux'],
                                           field_indices['uy'], field_indices['uz']])
            lnrho = input[field_indices['lnrho']:field_indices['lnrho']+1]
            ## MHD
            aa = tf.gather(input, indices=[field_indices['ax'],
                                           field_indices['ay'], field_indices['az']])
            ss = input[field_indices['ss']:field_indices['ss']+1]

            # Convolutions
            ## Hydro
            grad_lnrho, hessian_lnrho = tf.split(self.convolve(lnrho, tf.concat([self.ddx, self.d2dx2], axis=-1)), 2, axis=-1)
            grad_uu, hessian_uu, mixed_uu = tf.split(
                self.convolve(uu, tf.concat([self.ddx, self.d2dx2, self.d2dxdy], axis=-1)), 3, axis=-1)
            ## MHD
            grad_ss, hessian_ss = tf.split(self.convolve(ss, tf.concat([self.ddx, self.d2dx2], axis=-1)), 2, axis=-1)
            grad_aa, hessian_aa, mixed_aa = tf.split(
                self.convolve(aa, tf.concat([self.ddx, self.d2dx2, self.d2dxdy], axis=-1)), 3, axis=-1)

            # Squeeze input and remove padding
            uu = tf.squeeze(uu[:, args.radius:-args.radius,
                               args.radius:-args.radius, args.radius:-args.radius, :])
            lnrho = tf.squeeze(lnrho[:, args.radius:-args.radius, args.radius:-
                                     args.radius, args.radius:-args.radius, :])
            aa = tf.squeeze(aa[:, args.radius:-args.radius,
                               args.radius:-args.radius, args.radius:-args.radius, :])
            ss = tf.squeeze(ss[:, args.radius:-args.radius, args.radius:-
                                     args.radius, args.radius:-args.radius, :])

            # Flatten
            ## Hydro
            uu = tf.reshape(uu, (3, -1))
            lnrho = tf.reshape(lnrho, [-1])
            grad_lnrho = tf.reshape(tf.transpose(grad_lnrho, [0, 4, 1, 2, 3]), (3, -1))
            grad_uu = tf.reshape(tf.transpose(grad_uu, [0, 4, 1, 2, 3]), (3, 3, -1))
            hessian_uu = tf.reshape(tf.transpose(hessian_uu, [0, 4, 1, 2, 3]), (3, 3, -1))
            mixed_uu = tf.reshape(tf.transpose(mixed_uu, [0, 4, 1, 2, 3]), (3, 3, -1))

            ## MHD
            aa = tf.reshape(aa, (3, -1))
            ss = tf.reshape(ss, [-1])
            grad_ss = tf.reshape(tf.transpose(grad_ss, [0, 4, 1, 2, 3]), (3, -1))
            hessian_ss = tf.reshape(tf.transpose(hessian_ss, [0, 4, 1, 2, 3]), (3, -1))
            grad_aa = tf.reshape(tf.transpose(grad_aa, [0, 4, 1, 2, 3]), (3, 3, -1))
            hessian_aa = tf.reshape(tf.transpose(hessian_aa, [0, 4, 1, 2, 3]), (3, 3, -1))
            mixed_aa = tf.reshape(tf.transpose(mixed_aa, [0, 4, 1, 2, 3]), (3, 3, -1))
            cs2 = cs2_sound * tf.exp((gamma/cp_sound) * ss + (gamma - 1) * (lnrho - lnrho0))
            cs2 = tf.reshape(cs2, [-1])

            # More complex operations
            ## Hydro
            laplace_ss = tf.reduce_sum(hessian_ss, axis=0)
            laplace_lnrho = tf.reduce_sum(hessian_ss, axis=0)
            laplace_uu = tf.reduce_sum(hessian_uu, axis=1)
            div_uu = grad_uu[0, 0] + grad_uu[1, 1] + grad_uu[2, 2]
            grad_div_ux = hessian_uu[0, 0] + mixed_uu[1, 0] + mixed_uu[2, 1]
            grad_div_uy = mixed_uu[0, 0] + hessian_uu[1, 1] + mixed_uu[2, 2]
            grad_div_uz = mixed_uu[0, 1] + mixed_uu[1, 2] + hessian_uu[2, 2]
            ## Traceless rate-of-strain stress tensor
            S00 = (2/3) * grad_uu[0, 0] - (1/3) * (grad_uu[1,1] + grad_uu[2, 2])
            S01 = (1/2) * (grad_uu[0, 1] + grad_uu[1, 0])
            S02 = (1/2) * (grad_uu[0,2] + grad_uu[2,0])
            S0 = tf.stack([S00, S01, S02])
            
            S10 = S01
            S11 = (2/3) * grad_uu[1, 1] - (1/3) * (grad_uu[0,0] + grad_uu[2, 2])
            S12 = (1/2) * (grad_uu[1, 2] + grad_uu[2, 1])
            S1 = tf.stack([S10, S11, S12])

            S20 = S02
            S21 = S12
            S22 = (2/3) * grad_uu[2, 2] - (1/3) * (grad_uu[0,0] + grad_uu[1, 1])
            S2 = tf.stack([S20, S21, S22])
            
            S = tf.concat([S0, S1, S2], axis=0)
            ## MHD
            laplace_aa = tf.reduce_sum(hessian_aa, axis=1)
            div_aa = grad_aa[0, 0] + grad_aa[1, 1] + grad_aa[2, 2]
            grad_div_ax = hessian_aa[0, 0] + mixed_aa[1, 0] + mixed_aa[2, 1]
            grad_div_ay = mixed_aa[0, 0] + hessian_aa[1, 1] + mixed_aa[2, 2]
            grad_div_az = mixed_aa[0, 1] + mixed_aa[1, 2] + hessian_aa[2, 2]
            grad_div_aa = tf.stack([grad_div_ax, grad_div_ay, grad_div_az])
            j = (1/mu0) * (grad_div_aa - laplace_aa)
            B = self.curl(grad_aa)
            inv_rho = 1 / tf.exp(lnrho)
            lnT = lnT0 + (gamma / cp_sound) * ss + (gamma - 1) * (lnrho - lnrho0)
            inv_pT = 1 / (tf.exp(lnrho) * tf.exp(lnT))
            j_cross_B = self.cross(j, B)
            entropy_rhs = (0) - (0) + eta * mu0 * self.dot(j, j)\
            + 2 * tf.exp(lnrho) * nu_visc * tf.reduce_sum(S, axis=0)\
            + zeta * tf.exp(lnrho) * div_uu * div_uu
            ## Heat conduction
            grad_ln_chi = -grad_lnrho
            first_term = gamma * (1/cp_sound) * laplace_ss + (gamma - 1) * laplace_lnrho
            second_term = gamma * (1/cp_sound) * grad_ss + (gamma - 1.) * grad_lnrho
            third_term = gamma * ((1/cp_sound) * grad_ss + grad_lnrho) + grad_ln_chi
            chi = K_heatcond / (tf.exp(lnrho) * cp_sound)
            heat_conduction = cp_sound * chi * (first_term + self.dot(second_term, third_term))


            # Equations
            ## Hydro
            continuity = -self.dot(uu, grad_lnrho) - div_uu
            momx = - self.dot(uu, grad_uu[0])\
                - cs2 * ((1/cp_sound) * grad_ss[0] + grad_lnrho[0])\
                + nu_visc * (laplace_uu[0] + (1/3) * grad_div_ux + 2 * self.dot(S0, grad_lnrho))\
                + zeta * grad_div_ux\
                + inv_rho * j_cross_B[0]
            momy = - self.dot(uu, grad_uu[1])\
                - cs2 * ((1/cp_sound) * grad_ss[1] + grad_lnrho[1])\
                + nu_visc * (laplace_uu[1] + (1/3) * grad_div_uy + 2 * self.dot(S1, grad_lnrho))\
                + zeta * grad_div_uy\
                + inv_rho * j_cross_B[1]
            momz = - self.dot(uu, grad_uu[2])\
                - cs2 * ((1/cp_sound) * grad_ss[2] + grad_lnrho[2])\
                + nu_visc * (laplace_uu[2] + (1/3) * grad_div_uz + 2 * self.dot(S2, grad_lnrho))\
                + zeta * grad_div_uz\
                + inv_rho * j_cross_B[2]
            ## MHD
            ind = self.cross(uu, B) + eta * laplace_aa
            entropy = - self.dot(uu, grad_ss) + inv_pT * entropy_rhs + heat_conduction

            return tf.reshape(tf.stack([continuity, momx, momy, momz, ind[0], ind[1], ind[2], entropy]), (len(field_names), nn[2], nn[1], nn[0], 1))

        @tf.function(jit_compile=True)
        def forward_old(self, input):

            # Fields
            uu = tf.gather(input, indices=[field_indices['ux'],
                                           field_indices['uy'], field_indices['uz']])
            lnrho = input[field_indices['lnrho']:field_indices['lnrho']+1]

            # Convolutions
            grad_lnrho = lib.convolve(lnrho, self.ddx)
            grad_uu, hessian_uu, mixed_uu = tf.split(lib.convolve(
                uu, tf.concat([self.ddx, self.d2dx2, self.d2dxdy], axis=-1)), 3, axis=-1)

            # Squeeze input and remove padding
            uu = tf.squeeze(uu[:, args.radius:-args.radius,
                               args.radius:-args.radius, args.radius:-args.radius, :])
            # lnrho = tf.squeeze(lnrho[:, args.radius:-args.radius, args.radius:-
            #                          args.radius, args.radius:-args.radius, :])

            # Flatten
            uu = tf.reshape(uu, (3, -1))
            #lnrho = tf.reshape(lnrho, shape=[-1])
            grad_lnrho = tf.reshape(tf.transpose(
                grad_lnrho, [0, 4, 1, 2, 3]), (3, -1))
            grad_uu = tf.reshape(tf.transpose(
                grad_uu, [0, 4, 1, 2, 3]), (3, 3, -1))
            hessian_uu = tf.reshape(tf.transpose(
                hessian_uu, [0, 4, 1, 2, 3]), (3, 3, -1))
            mixed_uu = tf.reshape(tf.transpose(
                mixed_uu, [0, 4, 1, 2, 3]), (3, 3, -1))

            # Computations
            laplace_uu = tf.reduce_sum(hessian_uu, axis=1)
            div_uu = grad_uu[0, 0] + grad_uu[1, 1] + grad_uu[2, 2]
            grad_div_ux = hessian_uu[0, 0] + mixed_uu[1, 0] + mixed_uu[2, 1]
            grad_div_uy = mixed_uu[0, 0] + hessian_uu[1, 1] + mixed_uu[2, 2]
            grad_div_uz = mixed_uu[0, 1] + mixed_uu[1, 2] + hessian_uu[2, 2]

            ## Traceless rate-of-strain stress tensor
            S00 = (2/3) * grad_uu[0, 0] - (1/3) * (grad_uu[1,1] + grad_uu[2, 2])
            S01 = (1/2) * (grad_uu[0, 1] + grad_uu[1, 0])
            S02 = (1/2) * (grad_uu[0,2] + grad_uu[2,0])
            S0 = tf.stack([S00, S01, S02])
            
            S10 = S01
            S11 = (2/3) * grad_uu[1, 1] - (1/3) * (grad_uu[0,0] + grad_uu[2, 2])
            S12 = (1/2) * (grad_uu[1, 2] + grad_uu[2, 1])
            S1 = tf.stack([S10, S11, S12])
            
            S20 = S02
            S21 = S12
            S22 = (2/3) * grad_uu[2, 2] - (1/3) * (grad_uu[0,0] + grad_uu[1, 1])
            S2 = tf.stack([S20, S21, S22])

            continuity = -self.dot(uu, grad_lnrho) - div_uu
            momx = - self.dot(uu, grad_uu[0])\
                - cs2 * grad_lnrho[0]\
                + nu_visc * (laplace_uu[0] + (1/3) * grad_div_ux + 2 * self.dot(S0, grad_lnrho))\
                + zeta * grad_div_ux
            momy = - self.dot(uu, grad_uu[1])\
                - cs2 * grad_lnrho[1]\
                + nu_visc * (laplace_uu[1] + (1/3) * grad_div_uy + 2 * self.dot(S1, grad_lnrho))\
                + zeta * grad_div_uy
            momz = - self.dot(uu, grad_uu[2])\
                - cs2 * grad_lnrho[2]\
                + nu_visc * (laplace_uu[2] + (1/3) * grad_div_uz + 2 * self.dot(S2, grad_lnrho))\
                + zeta * grad_div_uz

            return tf.reshape(tf.stack([continuity, momx, momy, momz]), (len(field_names), nn[2], nn[1], nn[0], 1))

        def benchmark(self, num_samples):
            benchmark_output = Output()
            with tf.Graph().as_default() as graph:
                # Setup input
                input = self.pad(self.get_input())
                # Setup weights
                self.kronecker = self.get_weights(get_kronecker())
                self.ddx = self.get_weights(get_ddx())
                self.d2dx2 = self.get_weights(get_d2dx2())
                self.d2dxdy = self.get_weights(get_d2dxdy())
                self.weights = self.get_weights(get_weights())
                with tf.compat.v1.Session(graph=graph) as sess:
                    benchmark = tf.test.Benchmark()
                    for i in range(num_samples):
                        measurement = benchmark.run_op_benchmark(
                            sess, self.forward(input), min_iters=1)
                        benchmark_output.record(
                            1e3*measurement['wall_time'], i)

    lib = Tensorflow(args.device, args.dtype)

print(f'Using library {lib.name}')

# %%
# Check correctness
if args.verify:
    print('Verifying results...')
    model = forward(get_input())
    candidate = lib.forward(lib.pad(lib.get_input())).cpu().numpy().squeeze()
    epsilon = np.finfo(
        np.float64).eps if args.dtype == np.float64 else np.finfo(np.float32).eps
    epsilon *= 100
    correct = np.allclose(model, candidate, rtol=epsilon, atol=epsilon)
    print(f'Done. Results within rel/abs epsilon {epsilon}: {correct}')
    if not correct:
        diff = np.abs(model - candidate)
        print(f'Largest absolute error: {diff.max()}')
        print(f'Indices: {np.where(diff > epsilon)}')

    # Check convolutions
    model = convolve(get_input(), get_weights())
    candidate = lib.convolve(lib.pad(lib.get_input()),
                             lib.get_weights(get_weights())).cpu().numpy().squeeze()
    for field in range(len(model)):
        for stencil in range(len(model[0])):
            if args.library in 'tensorflow':
                correct = np.allclose(
                    model[field, stencil], candidate[field, :, :, :, stencil], rtol=epsilon, atol=epsilon)
            elif args.library in 'pytorch':
                correct = np.allclose(
                    model[field, stencil], candidate[field, stencil, :, :, :], rtol=epsilon, atol=epsilon)

            if not correct:
                print(f'Convolution f{field}_s{stencil} correct: {correct}')
    
# %%
# Benchmark
print('Benchmarking')
lib.benchmark(args.nsamples)

# %%
# %%
# Visualize
if args.visualize:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML


    def alpha_step(step, dt, intermediate, rate_of_change):    
        alpha = [0., -5./9., -153. / 128.]
        return alpha[step] * intermediate + rate_of_change * dt

    def beta_step(step, dt, field, intermediate):
        beta = [1. / 3., 15./ 16., 8. / 15.]
        return field + beta[step] * intermediate

    fig, axs = plt.subplots(2, int(len(field_names)/2), figsize=(15,5), dpi=200)
    ims = []
    cbars = []
    for i, ax in enumerate(axs.flat):
        im = ax.imshow(np.zeros((nn[1], nn[0])), cmap='plasma')
        ims.append(im)
        cbars.append(fig.colorbar(im, ax=ax))
        ax.set_title(field_names[i])

    input = lib.get_input()
    winput = lib.get_input()
    def animate(i):
        global lib, input, winput

        input = input + dt * lib.forward(lib.pad(input))
        for step in range(3):
            roc = lib.forward(lib.pad(input))
            winput = alpha_step(step, dt, winput, roc)
            input = beta_step(step, dt, input, winput)

        for i, ax in enumerate(axs.flat):
            field = input[i].cpu().numpy().squeeze()
            field = field[int(field.shape[0]/2),:,:]

            #ax.imshow(field, cmap='plasma', interpolation='none')
            ims[i].set_data(field)
            ims[i].set_clim(np.min(field), np.max(field))
            #fig.colorbar(ims[i])
            # if i == 0:
            #     print(field)

        #if torch.isnan(input).any():
        #    print(input)
        #    print('Found nan')
        return ims,

    ani = FuncAnimation(fig, animate, frames=100, blit=False)
    ani.save(f'nonlinear-mhd-{args.library}.mp4', writer='ffmpeg')
    #plt.show()
    #plt.close()
    #HTML(ani.to_html5_video())

# %%
'''
# Move fields to device
input = torch.from_numpy(input).reshape(1, 1, ny, nx).to(dtype=dtype, device=device)

steps = 0
def animate(i):
    global input, weights, dt
    global start

    start = time.time()
    input = convolve(bc(input), weights)
    print(f'Euler step time elapsed: {1e3*(time.time() - start)} ms')

    # Visualize
    host_input = input.reshape(ny, nx).cpu().numpy()
    print(f'{host_input}')
    axs.set_title(f'Step {i}')
    im.set_data(host_input)
    im.set_clim(np.min(host_input), np.max(host_input))

    global steps
    steps += 1
    if steps > 100:
        exit(0)

ani = FuncAnimation(fig, animate, interval=1)
plt.show()
'''

'''
# %%
# FFT test
import torch
input = lib.pad(lib.get_input())[0:1]
ddx = lib.get_weights(get_ddx()[0:1])
model = torch.nn.functional.conv3d(input, ddx)

input = input.squeeze()
ddx = ddx.squeeze()
kernel_width = 2*args.radius + 1
ddx = torch.nn.functional.pad(ddx, [0, input.shape[-3] - kernel_width, 0, input.shape[-2] - kernel_width, 0, input.shape[-1] - kernel_width], mode='constant')
fr_signal = torch.fft.rfftn(input)
fr_kernel = torch.fft.rfftn(ddx) 
fr_kernel.imag *= -1

hhat = fr_signal * fr_kernel
fprime = torch.fft.irfftn(hhat)
candidate = fprime[:args.dims[2], :args.dims[1], :args.dims[0]]

np.allclose(model, candidate)

# %%
# FFT test
import torch
input = lib.pad(lib.get_input())[0:1]
ddx = lib.get_weights(get_weights())
model = torch.nn.functional.conv3d(input, ddx).squeeze()

input = input.squeeze()
ddx = ddx.squeeze()
kernel_width = 2*args.radius + 1
ddx = torch.nn.functional.pad(ddx, [0, input.shape[-3] - kernel_width, 0, input.shape[-2] - kernel_width, 0, input.shape[-1] - kernel_width], mode='constant')
fr_signal = torch.fft.rfftn(input)
fr_kernel = torch.fft.rfftn(ddx) 
fr_kernel.imag *= -1

hhat = fr_signal * fr_kernel
fprime = torch.fft.irfftn(hhat)
candidate = fprime[:, :args.dims[2], :args.dims[1], :args.dims[0]]
#candidate = candidate[0:1] + candidate[-1:0:-1] # Circular shift property - reorder (pytorch does not support negative strides for slicing)
candidate = torch.cat([candidate[:1], torch.flip(candidate[1:], dims=[0])])

np.allclose(model, candidate)
'''
