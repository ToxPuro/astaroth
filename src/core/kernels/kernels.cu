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
#include "ac_fft.h"
#include "acc_runtime.cu"
#include "device_headers.h"
#include "host_datatypes.h"
#include "kernels.h"
#include "packing.cuh"

// clang-format off
#include "user_defines.h"
// clang-format on

static __global__ void
dummy_kernel(void)
{
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    // Commented out until issues on lumi sorted
    // acComplex a = exp(acComplex(1, 1) * AcReal(1));
    //AcReal3 a = (AcReal)2.0 * (AcReal3){1, 2, 3};
    //(void)a;
}

AcResult
acKernelDummy(void)
{
    #if AC_CPU_BUILD
    dummy_kernel();
    #else
    ERRCHK_CUDA_KERNEL_ALWAYS();
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    #endif
    return AC_SUCCESS;
}

AcResult
acKernelsClean()
{
    	acFFTQuit();
	return acReduceClean();
}
