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

// Function pointer definitions
typedef AcReal (*MapFn)(const AcReal&);
typedef AcReal (*ReduceFn)(const AcReal&, const AcReal&);

// Map functions
static __device__ inline AcReal
map_value(const AcReal& a)
{
    return AcReal(a);
}

// Reduce functions
static __device__ inline AcReal
reduce_max(const AcReal& a, const AcReal& b)
{
    return a > b ? a : b;
}

static __device__ inline AcReal
reduce_min(const AcReal& a, const AcReal& b)
{
    return a < b ? a : b;
}

/** Map data from a 3D array into a 1D array */
template <MapFn map_fn>
__global__ void
map(const AcReal* in, const int3 start, const int3 end, AcReal* out)
{
    assert((start >= (int3){0, 0, 0}));
    assert((end <= (int3){DCONST(AC_mx), DCONST(AC_my), DCONST(AC_mz)}));

    const int3 tid = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    const int3 in_idx3d      = start + tid;
    const bool out_of_bounds = in_idx3d.x >= end.x || in_idx3d.y >= end.y || in_idx3d.z >= end.z;

    const size_t in_idx  = IDX(in_idx3d);
    const size_t out_idx = tid.x + tid.y * blockDim.x + tid.z * blockDim.x * blockDim.y;

    if (out_of_bounds)
      out[out_idx] = AC_REAL_INVALID_VALUE;
    else
      out[out_idx] = map_fn(in[in_idx]);
}

template <ReduceFn reduce_fn>
__global__ void
reduce(AcReal* arr, const size_t count)
{
    const int curr = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ AcReal smem[];
    if (curr < count)
        smem[threadIdx.x] = arr[curr];
    else
        smem[threadIdx.x] = AC_REAL_INVALID_VALUE;

    __syncthreads();

    int offset = blockDim.x / 2;
    while (offset > 0) {
        if (threadIdx.x < offset) {
            const AcReal a = smem[threadIdx.x];
            const AcReal b = smem[threadIdx.x + offset];
            if (b != AC_REAL_INVALID_VALUE)
              smem[threadIdx.x] = reduce_fn(a, b);
            else
              smem[threadIdx.x] = a;
        }

        offset /= 2;
        __syncthreads();
    }

    if (!threadIdx.x)
        arr[blockIdx.x] = smem[threadIdx.x];
}

AcReal
acKernelReduceScal(const cudaStream_t stream, const ReductionType rtype, const AcReal* vtxbuf,
                   const int3 start, const int3 end, AcReal* scratchpad,
                   const size_t scratchpad_size)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    cudaDeviceSynchronize();

    /*
    // Determine the map and reduction type
    MapFn map_fn       = map_value;
    ReduceFn reduce_fn = reduce_max;
    switch (rtype) {
    case RTYPE_MAX:
        map_fn    = map_value;
        reduce_fn = reduce_max;
        break;
    case RTYPE_MIN:
        map_fn    = map_value;
        reduce_fn = reduce_min;
        break;
    default:
        WARNING("Invalid reduction type in acKernelReduceScal");
        return AC_FAILURE;
    };
    */

    // Set thread block dimensions
    const int3 dims  = end - start;
    const Volume tpb = (Volume){32, 32, 1};
    const Volume bpg = (Volume){
        as_size_t(int(ceil(double(dims.x) / tpb.x))),
        as_size_t(int(ceil(double(dims.y) / tpb.y))),
        as_size_t(int(ceil(double(dims.z) / tpb.z))),
    };
    const size_t initial_count = tpb.x * bpg.x * tpb.y * bpg.y * tpb.z * bpg.z;
    ERRCHK_ALWAYS(initial_count <= scratchpad_size);

    // Map
    // map<map_fn><<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf, start, end, scratchpad);
    switch (rtype) {
    case RTYPE_MAX: /* Fallthrough */
    case RTYPE_MIN:
        map<map_value><<<to_dim3(bpg), to_dim3(tpb), 0, stream>>>(vtxbuf, start, end, scratchpad);
        break;
    default:
        WARNING("Invalid reduction type in acKernelReduceScal");
        return AC_FAILURE;
    };

    // Reduce
    size_t count = initial_count;
    do {
        const size_t tpb  = 128;
        const size_t bpg  = as_size_t(ceil(double(count) / tpb));
        const size_t smem = tpb * sizeof(scratchpad[0]);
        // reduce<reduce_fn><<<bpg, tpb, smem, stream>>>(scratchpad, count);
        switch (rtype) {
        case RTYPE_MAX:
            reduce<reduce_max><<<bpg, tpb, smem, stream>>>(scratchpad, count);
            break;
        case RTYPE_MIN:
            reduce<reduce_min><<<bpg, tpb, smem, stream>>>(scratchpad, count);
            break;
        default:
            WARNING("Invalid reduction type in acKernelReduceScal");
            return AC_FAILURE;
        };

        ERRCHK_CUDA_KERNEL();

        count = bpg;
    } while (count > 1);

    // Copy the result back to host
    AcReal result;
    cudaMemcpyAsync(&result, scratchpad, sizeof(scratchpad[0]), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    cudaDeviceSynchronize();
    return result;
}

AcReal
acKernelReduceVec(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                  const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                  const AcReal* vtxbuf2, AcReal* scratchpad, AcReal* reduce_result)
{
    return 0;
}

AcReal
acKernelReduceVecScal(const cudaStream_t stream, const ReductionType rtype, const int3 start,
                      const int3 end, const AcReal* vtxbuf0, const AcReal* vtxbuf1,
                      const AcReal* vtxbuf2, const AcReal* vtxbuf3, AcReal* scratchpad,
                      AcReal* reduce_result)
{
    return 0;
}