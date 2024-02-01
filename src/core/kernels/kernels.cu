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

static __global__ void
dummy_kernel(void)
{
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    // Commented out until issues on lumi sorted
    // acComplex a = exp(acComplex(1, 1) * AcReal(1));
    AcReal3 a = (AcReal)2.0 * (AcReal3){1, 2, 3};
    (void)a;
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

AcResult
acLaunchKernelDebug(Kernel kernel, const cudaStream_t stream, const int3 vba_start,
                 const int3 vba_end,VertexBufferArray vba)
{
    const dim3 tpb(8, 8, 8);
    const int3 dims = {vba_end.x-vba_start.x,vba_end.y-vba_start.y,vba_end.z-vba_start.z};
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel<<<bpg, tpb, 0, stream>>>(vba_start, vba_end, vba);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}
