/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "kernels.h"

#include "acc_runtime.cu"

static_assert(NUM_VTXBUF_HANDLES > 0, "ERROR: At least one uniform ScalarField must be declared.");

static __global__ void
dummy_kernel(void)
{
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    acComplex a = exp(acComplex(1, 1) * AcReal(1));
    a* a;
}

AcResult
acKernelDummy(void)
{
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    return AC_SUCCESS;
}

// Built-in kernels
#include "boundconds.cuh"
#include "boundconds_miikka_GBC.cuh"
#include "packing.cuh"
#include "reductions.cuh"
#include "volume_copy.cuh"

AcResult
acKernel(const KernelParameters params, VertexBufferArray vba)
{
#ifdef AC_INTEGRATION_ENABLED
    // TODO: Why is AC_step_number loaded here??
    acLoadIntUniform(params.stream, AC_step_number, params.step_number);
    acLaunchKernel(params.kernel, params.stream, params.start, params.end, vba);
    return AC_SUCCESS;
#else
    (void)params; // Unused
    (void)vba;    // Unused
    ERROR("acKernel() called but AC_step_number not defined!");
    return AC_FAILURE;
#endif
}
