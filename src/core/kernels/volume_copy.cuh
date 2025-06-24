/*
    Copyright (C) 2014-2022, Johannes Pekkila, Miikka Vaisala.

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

/**
static __global__ void
kernel_volume_copy(const AcReal* in, const int3 in_offset, const int3 in_volume, //
                   AcReal* out, const int3 out_offset, const int3 out_volume)
{
    const int3 idx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };
    if (idx.x >= min(in_volume.x, out_volume.x) || //
        idx.y >= min(in_volume.y, out_volume.y) || //
        idx.z >= min(in_volume.z, out_volume.z))
        return;

    const int3 in_pos  = idx + in_offset;
    const int3 out_pos = idx + out_offset;

    const size_t in_idx = in_pos.x +               //
                          in_pos.y * in_volume.x + //
                          in_pos.z * in_volume.x * in_volume.y;
    const size_t out_idx = out_pos.x +                //
                           out_pos.y * out_volume.x + //
                           out_pos.z * out_volume.x * out_volume.y;

    out[out_idx] = in[in_idx];
}
**/

AcResult
acKernelVolumeCopy(const cudaStream_t stream,                                    //
                   const AcReal* in, const Volume in_offset, const Volume in_volume, //
                   AcReal* out, const Volume out_offset, const Volume out_volume)
{
    VertexBufferArray vba{};
    vba.on_device.out[0] = out;
    acLoadKernelParams(vba.on_device.kernel_input_params,AC_VOLUME_COPY,(AcReal*)in,in_offset,in_volume,out,out_offset,out_volume); 
    const Volume start = {0,0,0};
    const Volume nn = to_volume(min(to_int3(in_volume), to_int3(out_volume)));
    acLaunchKernel(AC_VOLUME_COPY,stream,start,nn,vba);
    /**
    const Volume tpb {512 < nn.x ? 512 : nn.x , 1, 1};
    const Volume bpg(ceil_div(nn,tpb));
    kernel_volume_copy<<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(in, to_int3(in_offset), to_int3(in_volume), //
                                                out, to_int3(out_offset), to_int3(out_volume));
    **/
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}
