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
#include <assert.h>

//<<<<<<< Updated upstream
//
//=======
//// Function pointer definitions
//typedef AcReal (*MapFn)(const AcReal&);
//typedef AcReal (*MapVecFn)(const AcReal&, const AcReal&, const AcReal&);
////typedef AcReal (*MapScalFacScalFn)(const AcReal&, const AcReal&, const AcReal&);
//typedef AcReal (*MapVecScalFn)(const AcReal&, const AcReal&, const AcReal&, const AcReal&);
//typedef AcReal (*ReduceFn)(const AcReal&, const AcReal&);
//typedef int    (*ReduceFnInt)(const int&, const int&);
//typedef AcReal (*CoordFn)(const AcReal3&);
//typedef void (*GridLocFn)(AcReal*, AcReal*, AcReal*, const int3&);
//
//// Map functions
//static __device__ inline AcReal
//map_value(const AcReal& a)
//{
//    return a;
//}
//
//static __device__ inline AcReal
//map_square(const AcReal& a)
//{
//    return a * a;
//}
//
//static __device__ inline AcReal
//map_exp_square(const AcReal& a)
//{
//    return exp(2.*a);
//}
//
//static __device__ inline AcReal
//map_exp_fac(const AcReal& a, const AcReal& b, const AcReal & fac)
//{
//    return exp(a + fac*b);
//}
//
//static __device__ inline AcReal
//map_length_vec(const AcReal& a, const AcReal& b, const AcReal& c)
//{
//    return sqrt(a * a + b * b + c * c);
//}
//
//static __device__ inline AcReal
//map_square_vec(const AcReal& a, const AcReal& b, const AcReal& c)
//{
//    return map_square(a) + map_square(b) + map_square(c);
//}
//
//static __device__ inline AcReal
//map_exp_square_vec(const AcReal& a, const AcReal& b, const AcReal& c)
//{
//    return map_exp_square(a) + map_exp_square(b) + map_exp_square(c);
//}
//
//static __device__ inline AcReal
//map_length_alf(const AcReal& a, const AcReal& b, const AcReal& c, const AcReal& d)
//{
//    return sqrt(a * a + b * b + c * c) / sqrt(exp(d));
//}
//
//static __device__ inline AcReal
//map_square_alf(const AcReal& a, const AcReal& b, const AcReal& c, const AcReal& d)
//{
//    return (map_square(a) + map_square(b) + map_square(c)) / (exp(d));
//}
//
//// Coordinate based functions
//
//// Here physical coordinate in the grid is calculating by assuming that
//// coordinate (0.0, 0.0, 0.0) corresresponds to index (0, 0, 0)
//// with distance between grid points being AC_dsx, AC_dsy, AC_dsz
//// respectively.
////
//
//#ifdef AC_INTEGRATION_ENABLED
//static __device__ inline void
//cartesian_grid_location(AcReal* coord_x1, AcReal* coord_y1, AcReal* coord_z1,
//                        const int3& globalVertexIdx)
//{
//    *coord_x1 = AcReal(globalVertexIdx.x - STENCIL_ORDER/2)*VAL(AC_dsx);
//    *coord_y1 = AcReal(globalVertexIdx.y - STENCIL_ORDER/2)*VAL(AC_dsy);
//#if TWO_D == 0
//    *coord_z1 = AcReal(globalVertexIdx.z - STENCIL_ORDER/2)*VAL(AC_dsz);
//#else
//    *coord_z1 = AcReal(0.0);
//#endif
//}
//
//static __device__ inline AcReal
//distance(const AcReal coord_x1, const AcReal coord_y1, const AcReal coord_z1, const AcReal coord_x2,
//         const AcReal coord_y2, const AcReal coord_z2)
//{
//    return sqrt((coord_x1 - coord_x2) * (coord_x1 - coord_x2) +
//                (coord_y1 - coord_y2) * (coord_y1 - coord_y2) +
//                (coord_z1 - coord_z2) * (coord_z1 - coord_z2));
//}
//
//static __device__ inline AcReal
//radial_window(const AcReal3& coordinate)
//{
//    AcReal loc_weight = 0.0;
//
//    const AcReal radius = distance(coordinate.x, coordinate.y,  coordinate.z,
//                                   VAL(AC_center_x), VAL(AC_center_y), 
//                                   VAL(AC_center_z)); 
//
//    if (radius <= VAL(AC_window_radius)) loc_weight = 1.0;  
//    //if (radius <= DCONST(AC_window_radius)) printf("Condition met radial_window \n");  OKOK
//
//    return loc_weight;
//}
//
//static __device__ inline AcReal
//gaussian_window(const AcReal3& coordinate)
//{
//    const AcReal radius = distance(coordinate.x, coordinate.y,  coordinate.z,
//                                   VAL(AC_center_x), VAL(AC_center_y), 
//                                   VAL(AC_center_z)); 
//    const AcReal rscale = VAL(AC_window_radius);
//
//    // if (radius <= DCONST(AC_window_radius)) printf("Condition met gaussian_window \n");  OKOK
//
//    // printf("radius %e, rscale %e, radius/rscale %e, exp((radius/rscale))^2 %e \n",
//    //         radius, rscale, radius/rscale, exp(-(radius/rscale)*(radius/rscale)));
//    return exp(-(radius / rscale) * (radius / rscale));
//}
//#else
//static __device__ inline void
//cartesian_grid_location(AcReal* coord_x1, AcReal* coord_y1, AcReal* coord_z1,
//                        const int3&)
//{
//    // Produce nan to halt the code
//    *coord_x1 = 0.0 / 0.0;
//    *coord_y1 = 0.0 / 0.0;
//    *coord_z1 = 0.0 / 0.0;
//}
//
//static __device__ inline AcReal
//distance(const AcReal , const AcReal , const AcReal , const AcReal ,
//         const AcReal , const AcReal )
//{
//    // Produce nan to halt the code
//    return 0.0 / 0.0;
//}
//
//static __device__ inline AcReal
//radial_window(const AcReal3& )
//{
//    // Produce nan to halt the code
//    return 0.0 / 0.0;
//}
//
//static __device__ inline AcReal
//gaussian_window(const AcReal3& )
//{
//    // Produce nan to halt the code
//    return 0.0 / 0.0;
//}
//#endif
//
//// Reduce functions
//static __device__ inline AcReal
//reduce_max(const AcReal& a, const AcReal& b)
//{
//    return a > b ? a : b;
//}
//
//static __device__ inline AcReal
//reduce_min(const AcReal& a, const AcReal& b)
//{
//    return a < b ? a : b;
//}
//
//static __device__ inline AcReal
//reduce_sum(const AcReal& a, const AcReal& b)
//{
//    return a + b;
//}
//static __device__ inline int
//reduce_max_int(const int& a, const int& b)
//{
//    return a > b ? a : b;
//}
//
//static __device__ inline int
//reduce_min_int(const int& a, const int& b)
//{
//    return a > b ? a : b;
//}
//
//static __device__ inline int
//reduce_sum_int(const int& a, const int& b)
//{
//    return a + b;
//}
//
//bool __device__
//bound_check(const int3 end)
//{
//#if TWO_D == 0
//    return ((end <= (int3){VAL(AC_mx), VAL(AC_my), VAL(AC_mz)}));
//#else
//    return ((end <= (int3){VAL(AC_mx), VAL(AC_my), 1}));
//#endif
//}
//
///** Map data from a 3D array into a 1D array */
//template <MapFn map_fn>
//__global__ void
//map(const AcReal* in, const int3 start, const int3 end, AcReal* out)
//{
//    assert((start >= (int3){0, 0, 0}));
//    assert(bound_check(end));
//
//    const int3 tid = (int3){
//        threadIdx.x + blockIdx.x * blockDim.x,
//        threadIdx.y + blockIdx.y * blockDim.y,
//        threadIdx.z + blockIdx.z * blockDim.z,
//    };
//
//    const int3 in_idx3d = start + tid;
//    const size_t in_idx = IDX(in_idx3d);
//
//    const int3 dims      = end - start;
//    const size_t out_idx = tid.x + tid.y * dims.x + tid.z * dims.x * dims.y;
//
//    const bool within_bounds = in_idx3d.x < end.x && in_idx3d.y < end.y && in_idx3d.z < end.z;
//    if (within_bounds)
//        out[out_idx] = map_fn(in[in_idx]);
//}
//
//template <MapVecFn map_fn>
//__global__ void
//map_vec(const AcReal* in0, const AcReal* in1, const AcReal* in2, const int3 start, const int3 end,
//        AcReal* out)
//{
//    assert((start >= (int3){0, 0, 0}));
//    assert(bound_check(end));
//
//    const int3 tid = (int3){
//        threadIdx.x + blockIdx.x * blockDim.x,
//        threadIdx.y + blockIdx.y * blockDim.y,
//        threadIdx.z + blockIdx.z * blockDim.z,
//    };
//
//    const int3 in_idx3d = start + tid;
//    const size_t in_idx = IDX(in_idx3d);
//
//    const int3 dims      = end - start;
//    const size_t out_idx = tid.x + tid.y * dims.x + tid.z * dims.x * dims.y;
//
//    const bool within_bounds = in_idx3d.x < end.x && in_idx3d.y < end.y && in_idx3d.z < end.z;
//    if (within_bounds)
//        out[out_idx] = map_fn(in0[in_idx], in1[in_idx], in2[in_idx]);
//}
///*
//template <MapScalFacScalFn map_fn>
//__global__ void
//map_scal_scal(const AcReal* in0, const AcReal* in1, const AcReal* fac, const int3 start, const int3 end, AcReal* out)
//{
//    assert((start >= (int3){0, 0, 0}));
//    assert((end <= (int3){DCONST(AC_mx), DCONST(AC_my), DCONST(AC_mz)}));
//
//    const int3 tid = (int3){
//        threadIdx.x + blockIdx.x * blockDim.x,
//        threadIdx.y + blockIdx.y * blockDim.y,
//        threadIdx.z + blockIdx.z * blockDim.z,
//    };
//
//    const int3 in_idx3d = start + tid;
//    const size_t in_idx = IDX(in_idx3d);
//
//    const int3 dims      = end - start;
//    const size_t out_idx = tid.x + tid.y * dims.x + tid.z * dims.x * dims.y;
//
//    const bool within_bounds = in_idx3d.x < end.x && in_idx3d.y < end.y && in_idx3d.z < end.z;
//    if (within_bounds)
//        out[out_idx] = map_fn(in0[in_idx], in1[in_idx], *fac);
//}
//*/
//template <MapVecScalFn map_fn>
//__global__ void
//map_vec_scal(const AcReal* in0, const AcReal* in1, const AcReal* in2, const AcReal* in3,
//             const int3 start, const int3 end, AcReal* out)
//{
//    assert((start >= (int3){0, 0, 0}));
//    assert(bound_check(end));
//>>>>>>> Stashed changes


template <typename T>
static void
swap_ptrs(T** a, T** b)
{
    T* tmp = *a;
    *a          = *b;
    *b          = tmp;
}

static Volume
get_map_tpb(void)
{
    return (Volume){32, 4, 1};
}

static Volume
get_map_bpg(const int3 dims, const Volume tpb)
{
    return (Volume){
        as_size_t(int(ceil(double(dims.x) / tpb.x))),
        as_size_t(int(ceil(double(dims.y) / tpb.y))),
        as_size_t(int(ceil(double(dims.z) / tpb.z))),
    };
}

size_t
acKernelReduceGetMinimumScratchpadSize(const int3 max_dims)
{
    const Volume tpb   = get_map_tpb();
    const Volume bpg   = get_map_bpg(max_dims, tpb);
    const size_t count = tpb.x * bpg.x * tpb.y * bpg.y * tpb.z * bpg.z;
    return count;
}

size_t
acKernelReduceGetMinimumScratchpadSizeBytes(const int3 max_dims)
{
    return sizeof(AcReal) * acKernelReduceGetMinimumScratchpadSize(max_dims);
}

AcReal
acKernelReduceScal(const cudaStream_t stream, const AcReduction reduction, const VertexBufferHandle vtxbuf,
                   const Volume start, const Volume end, const int scratchpad_index,
                   VertexBufferArray vba)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());


    // Compute block dimensions
    const Volume dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;

    const AcKernel map_kernel = reduction.map_vtxbuf_single;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceScal");
      return AC_FAILURE;
    }
    resize_scratchpad_real(scratchpad_index, initial_count*sizeof(AcReal), reduction.reduce_op);
    AcReal* in  = *(vba.reduce_buffer_real[scratchpad_index].src);
    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vtxbuf,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_buffer_real[scratchpad_index].res;
    acReduce(stream,reduction.reduce_op, vba.reduce_buffer_real[scratchpad_index],initial_count);
    AcReal result;
    ERRCHK_CUDA(cudaMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(cudaStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());
    return result;
}

AcReal
acKernelReduceVec(const cudaStream_t stream, const AcReduction reduction, const Volume start,
                  const Volume end, const Field3 vector, VertexBufferArray vba, const int scratchpad_index
                  )
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());


    // Set thread block dimensions
    const Volume dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;

    const AcKernel map_kernel = reduction.map_vtxbuf_vec;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceVec");
      return AC_FAILURE;
    }
    resize_scratchpad_real(scratchpad_index, initial_count*sizeof(AcReal), reduction.reduce_op);
    AcReal* in  = *(vba.reduce_buffer_real[scratchpad_index].src);
    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vector,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_buffer_real[scratchpad_index].res;
    acReduce(stream,reduction.reduce_op, vba.reduce_buffer_real[scratchpad_index],initial_count);
    AcReal result;
    ERRCHK_CUDA(cudaMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(cudaStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());
    return result;
}

AcReal
acKernelReduceVecScal(const cudaStream_t stream, const AcReduction reduction, const Volume start,
                      const Volume end, const Field4 vtxbufs,VertexBufferArray vba,
                      const int scratchpad_index)
{
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());


    // Set thread block dimensions
    const Volume dims            = end - start;
    const size_t initial_count = dims.x * dims.y * dims.z;
    const AcKernel map_kernel = reduction.map_vtxbuf_vec_scal;
    if(map_kernel == AC_NULL_KERNEL)
    {
      ERROR("Invalid reduction type in acKernelReduceVecScal");
      return AC_FAILURE;
    }

    resize_scratchpad_real(scratchpad_index, initial_count*sizeof(AcReal), reduction.reduce_op);
    AcReal* in  = *(vba.reduce_buffer_real[scratchpad_index].src);
    acLoadKernelParams(vba.on_device.kernel_input_params,map_kernel,vtxbufs,in); 
    acLaunchKernel(map_kernel,stream,start,end,vba);

    AcReal* out = vba.reduce_buffer_real[scratchpad_index].res;
    acReduce(stream,reduction.reduce_op, vba.reduce_buffer_real[scratchpad_index],initial_count);
    AcReal result;
    ERRCHK_CUDA(cudaMemcpyAsync(&result, out, sizeof(out[0]), cudaMemcpyDeviceToHost, stream));
    ERRCHK_CUDA_ALWAYS(cudaStreamSynchronize(stream));
    
    // NOTE synchronization here: we have only one scratchpad at the moment and multiple reductions
    // cannot be parallelized due to race conditions to this scratchpad Communication/memcopies
    // could be done in parallel, but allowing that also exposes the users to potential bugs with
    // race conditions
    ERRCHK_CUDA_ALWAYS(cudaDeviceSynchronize());
    return result;
}
