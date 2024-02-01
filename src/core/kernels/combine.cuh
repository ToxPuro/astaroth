
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
#pragma once
__global__ void
kernel_combine(const int3 vba_start, const int3 vba_end, const VertexBufferArray vba)
{
    constexpr AcReal coef = (1.0/3.0);
    const int i = threadIdx.x + blockIdx.x*blockDim.x + vba_start.x;
    const int j = threadIdx.y + blockIdx.y*blockDim.y + vba_start.y;
    const int k = threadIdx.z + blockIdx.z*blockDim.z + vba_start.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i >= vba_end.x || //
        j >= vba_end.y || //
        k >= vba_end.z) {
        return;
    }


    const int idx = DEVICE_VTXBUF_IDX(i,j,k);
    for(int i=0;i<NUM_VTXBUF_HANDLES; ++i)
      vba.out[i][idx] += coef*vba.in[i][idx];
}




AcResult
acKernelCombine(const cudaStream_t stream, const VertexBufferArray vba, const int3 vba_start,
                 const int3 vba_end)
{
    const dim3 tpb(8, 8, 8);
    const int3 dims = {vba_end.x-vba_start.x,vba_end.y-vba_start.y,vba_end.z-vba_start.z};
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_combine<<<bpg, tpb, 0, stream>>>(vba_start, vba_end, vba);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}
