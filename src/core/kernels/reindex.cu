#include "host_datatypes.h"
#include "device_headers.h"
#include "device_details.h"
#include "ac_helpers.h"
#include "errchk.h"
#include "reindex.h"

__host__ __device__ constexpr bool
acOutOfBounds(const AcIndex& index, const AcShape& shape)
{
  return (index.x >= shape.x) || //
         (index.y >= shape.y) || //
         (index.z >= shape.z) || //
         (index.w >= shape.w);
}

static __host__ __device__ constexpr AcIndex __attribute__((unused))
min(const AcIndex& a, const AcIndex& b)
{
  return (AcIndex){
      a.x < b.x ? a.x : b.x,
      a.y < b.y ? a.y : b.y,
      a.z < b.z ? a.z : b.z,
      a.w < b.w ? a.w : b.w,
  };
}

__host__ __device__ constexpr AcIndex
operator+(const AcIndex& a, const AcIndex& b)
{
  return (AcIndex){
      a.x + b.x,
      a.y + b.y,
      a.z + b.z,
      a.w + b.w,
  };
}

static __host__ __device__ constexpr AcIndex __attribute__((unused))
operator-(const AcIndex& a, const AcIndex& b) 
{
  return (AcIndex){
      a.x - b.x,
      a.y - b.y,
      a.z - b.z,
      a.w - b.w,
  };
}

__host__ __device__ constexpr AcIndex
to_spatial(const size_t i, const AcShape& shape)
{
  return (AcIndex){
      .x = i % shape.x,
      .y = (i / shape.x) % shape.y,
      .z = (i / (shape.x * shape.y)) % shape.z,
      .w = i / (shape.x * shape.y * shape.z),
  };
}

__host__ __device__ constexpr size_t
to_linear(const AcIndex& index, const AcShape& shape)
{
  return index.x +           //
         index.y * shape.x + //
         index.z * shape.x * shape.y + index.w * shape.x * shape.y * shape.z;
}

#if 0
__global__ void
map_cross_product(const CrossProductInputs inputs, const AcIndex start,
                  const AcIndex end)
{

  const AcIndex tid = {
      .x = threadIdx.x + blockIdx.x * blockDim.x,
      .y = threadIdx.y + blockIdx.y * blockDim.y,
      .z = threadIdx.z + blockIdx.z * blockDim.z,
      .w = 0,
  };

  const AcIndex in_idx3d = start + tid;
  const size_t in_idx = DEVICE_VTXBUF_IDX(in_idx3d.x, in_idx3d.y, in_idx3d.z);

  const AcShape dims   = end - start;
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
#endif

#ifdef AC_TFM_ENABLED

static __global__ void
reindex(const AcReal* in, const AcIndex in_offset, const AcShape in_shape,
        AcReal* out, const AcIndex out_offset, const AcShape out_shape,
        const AcShape block_shape)
{
  const size_t i    = (size_t)threadIdx.x + blockIdx.x * blockDim.x;
  const AcIndex idx = to_spatial(i, block_shape);

  const AcIndex in_pos  = idx + in_offset;
  const AcIndex out_pos = idx + out_offset;

  if (acOutOfBounds(idx, block_shape) || //
      acOutOfBounds(in_pos, in_shape) || //
      acOutOfBounds(out_pos, out_shape))
    return;

  const size_t in_idx  = to_linear(in_pos, in_shape);
  const size_t out_idx = to_linear(out_pos, out_shape);

  out[out_idx] = in[in_idx];
}

AcResult
acReindex(const cudaStream_t stream, //
          const AcReal* in, const AcIndex in_offset, const AcShape in_shape,
          AcReal* out, const AcIndex out_offset, const AcShape out_shape,
          const AcShape block_shape)
{
  const size_t count = acShapeSize(block_shape);
  const size_t tpb   = min(256ul, count);
  const size_t bpg   = (count + tpb - 1) / tpb;

  KERNEL_LAUNCH(reindex,bpg, tpb, 0, stream)(in, in_offset, in_shape, //
                                   out, out_offset, out_shape, block_shape);
  ERRCHK_CUDA_KERNEL();

  return AC_SUCCESS;
}

typedef struct {
  AcReal *x, *y, *z;
} SOAVector;

typedef struct {
  // Input vectors
  SOAVector A[1];
  size_t A_count;
  SOAVector B[4];
  size_t B_count;
  // Note: more efficient with A_count < B_count

  // Output vectors
  SOAVector C[1 * 4];
  // C count = A_count*B_count
} CrossProductArrays;


static __global__ void UNUSED
reindex_cross(const CrossProductArrays arrays, const AcIndex in_offset,
              const AcShape in_shape, const AcIndex out_offset,
              const AcShape out_shape, const AcShape block_shape)
{
  const AcIndex idx = to_spatial(
      static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x, block_shape);

  const AcIndex in_pos  = idx + in_offset;
  const AcIndex out_pos = idx + out_offset;

  if (acOutOfBounds(idx, block_shape) || //
      acOutOfBounds(in_pos, in_shape) || //
      acOutOfBounds(out_pos, out_shape))
    return;

  const size_t in_idx  = to_linear(in_pos, in_shape);
  const size_t out_idx = to_linear(out_pos, out_shape);

  for (size_t j = 0; j < arrays.A_count; ++j) {
    const AcReal3 a = {
        arrays.A[j].x[in_idx],
        arrays.A[j].y[in_idx],
        arrays.A[j].z[in_idx],
    };
    for (size_t i = 0; i < arrays.B_count; ++i) {
      const AcReal3 b = {
          arrays.B[i].x[in_idx],
          arrays.B[i].y[in_idx],
          arrays.B[i].z[in_idx],
      };
      const AcReal3 res                           = AC_cross(a, b);
      arrays.C[i + j * arrays.B_count].x[out_idx] = res.x;
      arrays.C[i + j * arrays.B_count].y[out_idx] = res.y;
      arrays.C[i + j * arrays.B_count].z[out_idx] = res.z;
    }
  }
}
AcResult
acReindexCross(const cudaStream_t stream, //
               const VertexBufferArray vba, const AcIndex in_offset,
               const AcShape in_shape, //
               AcReal* out, const AcIndex out_offset, const AcShape out_shape,
               const AcShape block_shape)
{
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

  const size_t block_offset = out_shape.x * out_shape.y * out_shape.z;
  const SOAVector out_bb11  = {
       .x = &out[3 * block_offset],
       .y = &out[4 * block_offset],
       .z = &out[5 * block_offset],
  };
  const SOAVector out_bb12 = {
      .x = &out[6 * block_offset],
      .y = &out[7 * block_offset],
      .z = &out[8 * block_offset],
  };
  const SOAVector out_bb21 = {
      .x = &out[9 * block_offset],
      .y = &out[10 * block_offset],
      .z = &out[11 * block_offset],
  };
  const SOAVector out_bb22 = {
      .x = &out[12 * block_offset],
      .y = &out[13 * block_offset],
      .z = &out[14 * block_offset],
  };

  const CrossProductArrays arrays = {
      .A       = {uu},
      .A_count = 1,
      .B       = {bb11, bb12, bb21, bb22},
      .B_count = 4,
      .C       = {out_bb11, out_bb12, out_bb21, out_bb22},
  };

  const size_t count = acShapeSize(block_shape);
  const size_t tpb   = min(256ul, count);
  const size_t bpg   = (count + tpb - 1) / tpb;

  KERNEL_LAUNCH(reindex_cross,bpg, tpb, 0, stream)(arrays, in_offset, in_shape,
                                         out_offset, out_shape, block_shape);
  return AC_SUCCESS;
}
#else
/**
AcResult
acReindexCross(const cudaStream_t , //
               const VertexBufferArray , const AcIndex ,
               const AcShape , //
               AcReal* , const AcIndex , const AcShape ,
               const AcShape )
{
  ERROR("acReindexCross called but AC_TFM_ENABLED was false");
  return AC_FAILURE;
}
**/

AcResult
acReindex(const cudaStream_t, //
          const AcReal*, const AcIndex, const AcShape,
          AcReal*, const AcIndex, const AcShape,
          const AcShape)
{
  ERROR("acReindex called but AC_TFM_ENABLED was false");
  return AC_FAILURE;
}
#endif
