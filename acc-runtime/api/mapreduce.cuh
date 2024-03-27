/*
    Copyright (C) 2024, Johannes Pekkila.

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
#include <thrust/device_vector.h>
#include <thrust/functional.h>
// #include <thrust/transform_reduce.h>
#include <thrust/reduce.h>

typedef struct {
  AcReal *x, *y, *z;
} SOAVector;

typedef struct {
  SOAVector A[1];
  size_t A_count;
  SOAVector B[4];
  size_t B_count;
  SOAVector outputs[4]; // Same count as B_count
} CrossProductInputs;

__global__ void
map_cross_product(const CrossProductInputs inputs, const int3 start,
                  const int3 end)
{
  assert((start >= (int3){0, 0, 0}));
  assert((end <= (int3){DCONST(AC_mx), DCONST(AC_my), DCONST(AC_mz)}));

  const int3 tid = (int3){
      threadIdx.x + blockIdx.x * blockDim.x,
      threadIdx.y + blockIdx.y * blockDim.y,
      threadIdx.z + blockIdx.z * blockDim.z,
  };

  const int3 in_idx3d = start + tid;
  const size_t in_idx = IDX(in_idx3d);

  const int3 dims      = end - start;
  const size_t out_idx = tid.x + tid.y * dims.x + tid.z * dims.x * dims.y;

  const bool within_bounds = in_idx3d.x < end.x && in_idx3d.y < end.y &&
                             in_idx3d.z < end.z;
  if (within_bounds) {
    for (size_t i = 0; i < inputs.A_count; ++i) {
      const AcReal3 a = (AcReal3){
          inputs.A[i].x[in_idx],
          inputs.A[i].y[in_idx],
          inputs.A[i].z[in_idx],
      };
      for (size_t j = 0; j < inputs.B_count; ++j) {
        const AcReal3 b = (AcReal3){
            inputs.B[j].x[in_idx],
            inputs.B[j].y[in_idx],
            inputs.B[j].z[in_idx],
        };
        const AcReal3 res            = cross(a, b);
        inputs.outputs[j].x[out_idx] = res.x;
        inputs.outputs[j].y[out_idx] = res.y;
        inputs.outputs[j].z[out_idx] = res.z;
      }
    }
  }
}

AcBufferArray
acBufferArrayCreate(const size_t num_buffers, const size_t count)
{
  AcBufferArray ba = {
      .buffers     = (AcReal**)malloc(sizeof(AcReal*) * num_buffers),
      .num_buffers = num_buffers,
      .count       = count,
  };
  ERRCHK_ALWAYS(ba.buffers);

  const size_t bytes = sizeof(ba.buffers[0][0]) * ba.count;
  for (size_t i = 0; i < ba.num_buffers; ++i)
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&ba.buffers[i], bytes));

  return ba;
}

void
acBufferArrayDestroy(AcBufferArray* ba)
{
  for (size_t i = 0; i < ba->num_buffers; ++i) {
    cudaFree(ba->buffers[i]);
    ba->buffers[i] = NULL;
  }
  ba->count = 0;

  free(ba->buffers);
  ba->buffers     = NULL;
  ba->num_buffers = 0;
}

size_t
acMapCross(const VertexBufferArray vba, const cudaStream_t stream,
           const int3 start, const int3 end, AcBufferArray scratchpad)
{
  const int3 dims    = end - start;
  const size_t count = as_size_t(dims.x) * as_size_t(dims.y);
  ERRCHK_ALWAYS(scratchpad.count >= count);
  ERRCHK_ALWAYS(scratchpad.num_buffers >= 12);

  const SOAVector uu = {
      .x = vba.in[VTXBUF_UUX],
      .y = vba.in[VTXBUF_UUY],
      .z = vba.in[VTXBUF_UUZ],
  };
  const SOAVector bb11 = {
      .x = vba.in[TF_b11_x],
      .y = vba.in[TF_b11_y],
      .z = vba.in[TF_b11_z],
  };
  const SOAVector bb12 = {
      .x = vba.in[TF_b12_x],
      .y = vba.in[TF_b12_y],
      .z = vba.in[TF_b12_z],
  };
  const SOAVector bb21 = {
      .x = vba.in[TF_b21_x],
      .y = vba.in[TF_b21_y],
      .z = vba.in[TF_b21_z],
  };
  const SOAVector bb22 = {
      .x = vba.in[TF_b22_x],
      .y = vba.in[TF_b22_y],
      .z = vba.in[TF_b22_z],
  };
  const SOAVector out_bb11 = {
      .x = scratchpad.buffers[0],
      .y = scratchpad.buffers[1],
      .z = scratchpad.buffers[2],
  };
  const SOAVector out_bb12 = {
      .x = scratchpad.buffers[3],
      .y = scratchpad.buffers[4],
      .z = scratchpad.buffers[5],
  };
  const SOAVector out_bb21 = {
      .x = scratchpad.buffers[6],
      .y = scratchpad.buffers[7],
      .z = scratchpad.buffers[8],
  };
  const SOAVector out_bb22 = {
      .x = scratchpad.buffers[9],
      .y = scratchpad.buffers[10],
      .z = scratchpad.buffers[11],
  };

  const CrossProductInputs inputs = {
      .A       = {uu},
      .A_count = 1,
      .B       = {bb11, bb12, bb21, bb22},
      .B_count = 4,
      .outputs = {out_bb11, out_bb12, out_bb21, out_bb22},
  };

  const dim3 tpb = (dim3){64, 8, 1};
  // Integer round-up division
  const dim3 bpg = (dim3){
      (dims.x + tpb.x - 1) / tpb.x,
      (dims.y + tpb.y - 1) / tpb.y,
      (dims.z + tpb.z - 1) / tpb.z,
  };
  map_cross_product<<<bpg, tpb, 0, stream>>>(inputs, start, end);
  return count;
}

// struct CrossProduct {
//   /*
//   __host__ __device__ void operator()(                      //
//       const AcReal& a1, const AcReal& a2, const AcReal& a3, //
//       const AcReal& b1, const AcReal& b2, const AcReal& b3, //
//       AcReal& c1, AcReal& c2, AcReal& c3)
//   {
//     c1 = a2 * b3 - a3 * b2;
//     c2 = a3 * b1 - a1 * b3;
//     c3 = a1 * b2 - a2 * b1;
//   }
//   */
//   template <typename Tuple> __host__ __device__ void operator()(Tuple t)
//   {
//     thrust::get<2>(t) = thrust::get<0>(t);
//   }
// };

// #include <thrust/for_each.h>
// #include <thrust/iterator/zip_iterator.h>
// #include <thrust/tuple.h>
// void
// acThrustMapCross(const AcReal* a0, const AcReal* a1, const AcReal* a2,
//                  const AcReal* b0, const AcReal* b1, const AcReal* b2,
//                  const size_t count, AcReal* c0, AcReal* c1, AcReal* c2)
// {
//   thrust::for_each(thrust::make_zip_iterator(
//                        thrust::make_tuple(a0, a1, a2, b0, b1, b2, c0, c1,
//                        c2)),
//                    thrust::make_zip_iterator(
//                        thrust::make_tuple(a0 + count, a1 + count, a2 + count,
//                                           b0 + count, b1 + count, b2 + count,
//                                           c0 + count, c1 + count, c2 +
//                                           count)),
//                    CrossProduct);
// }

__global__ void
greduce(const AcReal* in, const size_t count, AcReal* out, AcReal* output)
{
  // Note: possible integer overflow when GPU memory becomes large enough
  const size_t curr = threadIdx.x + blockIdx.x * blockDim.x;

  extern __shared__ AcReal smem[];
  if (curr < count)
    smem[threadIdx.x] = in[curr];
  else
    smem[threadIdx.x] = AC_REAL_INVALID_VALUE;

  __syncthreads();

  size_t offset = blockDim.x / 2;
  while (offset > 0) {
    if (threadIdx.x < offset) {
      const AcReal a = smem[threadIdx.x];
      const AcReal b = smem[threadIdx.x + offset];

      // If the mesh dimensions are not divisible by mapping tbdims, and mapping
      // tb dims are not divisible by reduction tb dims, then it is possible for
      // `a` to be invalid but `b` to be valid
      if (a != AC_REAL_INVALID_VALUE && b != AC_REAL_INVALID_VALUE)
        smem[threadIdx.x] = a + b;
      else if (a != AC_REAL_INVALID_VALUE)
        smem[threadIdx.x] = a;
      else
        smem[threadIdx.x] = b;
    }

    offset /= 2;
    __syncthreads();
  }

  if (!threadIdx.x) {
    out[blockIdx.x] = smem[threadIdx.x];
    if (!blockIdx.x)
      output[0] = smem[threadIdx.x];
  }
}

static void
swap_pointers(AcReal** a, AcReal** b)
{
  AcReal* tmp = *a;
  *a          = *b;
  *b          = tmp;
}

AcReal*
acKernelReduceScalToOutput(const cudaStream_t stream, const AcReal* input,
                           const size_t initial_count, AcReal* scratchpad0,
                           AcReal* scratchpad1, AcReal* output)
{
  AcReal* in  = scratchpad0;
  AcReal* out = scratchpad1;

  // Reduce
  size_t count = initial_count;
  do {
    const size_t tpb  = 128;
    const size_t bpg  = as_size_t(ceil(double(count) / tpb));
    const size_t smem = tpb * sizeof(in[0]);

    swap_pointers(&in, &out);
    if (count == initial_count)
      greduce<<<bpg, tpb, smem, stream>>>(input, count, out, output);
    else
      greduce<<<bpg, tpb, smem, stream>>>(in, count, out, output);
    ERRCHK_CUDA_KERNEL();

    count = bpg;
  } while (count > 1);
  return out;
}

// AcReal*
// acKernelReduceScal(const cudaStream_t stream, const AcReal* input,
//                    const size_t initial_count, AcReal* scratchpad0,
//                    AcReal* scratchpad1)
// {
//   AcReal* in  = scratchpad0;
//   AcReal* out = scratchpad1;
//   cudaMemcpy(in, input, sizeof(input[0]) * initial_count,
//              cudaMemcpyDeviceToDevice);

//   // Reduce
//   size_t count = initial_count;
//   do {
//     const size_t tpb  = 128;
//     const size_t bpg  = as_size_t(ceil(double(count) / tpb));
//     const size_t smem = tpb * sizeof(in[0]);

//     greduce<<<bpg, tpb, smem, stream>>>(in, count, out);
//     ERRCHK_CUDA_KERNEL();
//     swap_pointers(&in, &out);

//     count = bpg;
//   } while (count > 1);
//   return in;
// }

void
acMapCrossReduce(const VertexBufferArray vba, const cudaStream_t stream,
                 AcBufferArray scratchpad, ProfileBufferArray pba)
{
  ERRCHK_ALWAYS(vba.mz == pba.count);

  //   auto ucrossb11mean_x = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb11mean_x]);
  //   auto ucrossb11mean_y = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb11mean_y]);
  //   auto ucrossb11mean_z = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb11mean_z]);

  //   auto ucrossb12mean_x = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb12mean_x]);
  //   auto ucrossb12mean_y = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb12mean_y]);
  //   auto ucrossb12mean_z = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb12mean_z]);

  //   auto ucrossb21mean_x = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb21mean_x]);
  //   auto ucrossb21mean_y = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb21mean_y]);
  //   auto ucrossb21mean_z = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb21mean_z]);

  //   auto ucrossb22mean_x = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb22mean_x]);
  //   auto ucrossb22mean_y = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb22mean_y]);
  //   auto ucrossb22mean_z = thrust::device_pointer_cast(
  //       pba.in[PROFILE_ucrossb22mean_z]);
  ERRCHK_ALWAYS(scratchpad.num_buffers >= 12 + 2);
  const size_t radius = STENCIL_ORDER / 2;
  for (size_t k = 0; k < vba.mz; ++k) {
    const int3 start   = (int3){radius, radius, 0};
    const int3 end     = (int3){vba.mx - radius, vba.my - radius, 1};
    const size_t count = acMapCross(vba, stream, start, end, scratchpad);
    ERRCHK_ALWAYS(count == (vba.mx - 2 * radius) * (vba.my - 2 * radius));

    for (size_t i = 0; i < 12; ++i) {
      acKernelReduceScalToOutput(stream, scratchpad.buffers[i], count,
                                 scratchpad.buffers[12], scratchpad.buffers[13],
                                 &pba.in[PROFILE_ucrossb11mean_x + i][k]);
      // cudaMemcpyAsync(&pba.in[PROFILE_ucrossb11mean_x + i][k],
      //                 acKernelReduceScal(stream, scratchpad.buffers[i],
      //                 count,
      //                                    scratchpad.buffers[12],
      //                                    scratchpad.buffers[13]),
      //                 sizeof(AcReal), cudaMemcpyDeviceToDevice, stream);
      //   acKernelReduceScal(stream, scratchpad.buffers[i], count,
      //                      scratchpad.buffers[12], scratchpad.buffers[13]);

      // AcReal ac_result;
      // cudaMemcpyAsync(&ac_result, &pba.in[PROFILE_ucrossb11mean_x + i][k],
      //                 sizeof(AcReal), cudaMemcpyHostToDevice, stream);
      // cudaStreamSynchronize(stream);
      // AcReal thrust_result = thrust::reduce(
      //     thrust::cuda::par.on(stream),
      //     thrust::device_pointer_cast(scratchpad.buffers[0]),
      //     thrust::device_pointer_cast(scratchpad.buffers[0]) + count);
      // printf("%zu results: %g (ac) and %g (thrust)\n", i, ac_result,
      //        thrust_result);
    }
    // ucrossb11mean_x[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[0]),
    //     thrust::device_pointer_cast(scratchpad.buffers[0]) + count);
    // ucrossb11mean_y[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[1]),
    //     thrust::device_pointer_cast(scratchpad.buffers[1]) + count);
    // ucrossb11mean_z[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[2]),
    //     thrust::device_pointer_cast(scratchpad.buffers[2]) + count);

    // ucrossb12mean_x[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[3]),
    //     thrust::device_pointer_cast(scratchpad.buffers[3]) + count);
    // ucrossb12mean_y[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[4]),
    //     thrust::device_pointer_cast(scratchpad.buffers[4]) + count);
    // ucrossb12mean_z[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[5]),
    //     thrust::device_pointer_cast(scratchpad.buffers[5]) + count);

    // ucrossb21mean_x[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[6]),
    //     thrust::device_pointer_cast(scratchpad.buffers[6]) + count);
    // ucrossb21mean_y[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[7]),
    //     thrust::device_pointer_cast(scratchpad.buffers[7]) + count);
    // ucrossb21mean_z[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[8]),
    //     thrust::device_pointer_cast(scratchpad.buffers[8]) + count);

    // ucrossb22mean_x[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[9]),
    //     thrust::device_pointer_cast(scratchpad.buffers[9]) + count);
    // ucrossb22mean_y[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[10]),
    //     thrust::device_pointer_cast(scratchpad.buffers[10]) + count);
    // ucrossb22mean_z[k] = thrust::reduce(
    //     thrust::cuda::par.on(stream),
    //     thrust::device_pointer_cast(scratchpad.buffers[11]),
    //     thrust::device_pointer_cast(scratchpad.buffers[11]) + count);
  }
  // cudaStreamSynchronize(stream);
}
