#pragma once
//#include "user_defines.h"

extern "C" {

/**************************
 *                        *
 *  Symmetric boundconds  *
 *                        *
 **************************/

static __global__ void
kernel_symmetric_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{
    // The only reason a lot of these are const and not constexpr is DCONST(AC_n*) is not a compile
    // time expression
    const int start_x = (region_id.x == 1    ? NGHOST + DCONST(AC_nx)
                         : region_id.x == -1 ? 0
                                             : NGHOST);
    const int start_y = (region_id.y == 1    ? NGHOST + DCONST(AC_ny)
                         : region_id.y == -1 ? 0
                                             : NGHOST);
    const int start_z = (region_id.z == 1    ? NGHOST + DCONST(AC_nz)
                         : region_id.z == -1 ? 0
                                             : NGHOST);

    const int i_dst = start_x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start_y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start_z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= start_x + dims.x || j_dst >= start_y + dims.y || k_dst >= start_z + dims.z)
        return;

    const int mirror_x = normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1 : normal.x == -1 ? NGHOST : -1;
    const int mirror_y = normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1 : normal.y == -1 ? NGHOST : -1;
    const int mirror_z = normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1 : normal.z == -1 ? NGHOST : -1;

    const bool mask_x = (normal.x == 0);
    const bool mask_y = (normal.y == 0);
    const bool mask_z = (normal.z == 0);

    const int i_src = mask_x ? i_dst : mirror_x * 2 - i_dst;
    const int j_src = mask_y ? j_dst : mirror_y * 2 - j_dst;
    const int k_src = mask_z ? k_dst : mirror_z * 2 - k_dst;

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);

    vtxbuf[dst_idx] = vtxbuf[src_idx];
    // For antisymmetric boundconds:
    // vtxbuf[dst_idx]   = sign * vtxbuf[src_idx];
}

AcResult
acKernelSymmetricBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                            const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_symmetric_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}

/************************
 *                      *
 *  Entropy boundconds  *
 *                      *
 ************************/


static __global__ void
kernel_entropy_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            VertexBufferArray vba)
{
    // The only reason a lot of these are const and not constexpr is DCONST(AC_n*) is not a compile
    // time expression
    const int start_x = (region_id.x == 1    ? NGHOST + DCONST(AC_nx)
                         : region_id.x == -1 ? 0
                                             : NGHOST);
    const int start_y = (region_id.y == 1    ? NGHOST + DCONST(AC_ny)
                         : region_id.y == -1 ? 0
                                             : NGHOST);
    const int start_z = (region_id.z == 1    ? NGHOST + DCONST(AC_nz)
                         : region_id.z == -1 ? 0
                                             : NGHOST);

    const int i_dst = start_x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start_y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start_z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= start_x + dims.x || j_dst >= start_y + dims.y || k_dst >= start_z + dims.z)
        return;

    const int mirror_x = normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1 : normal.x == -1 ? NGHOST : -1;
    const int mirror_y = normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1 : normal.y == -1 ? NGHOST : -1;
    const int mirror_z = normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1 : normal.z == -1 ? NGHOST : -1;

    const bool mask_x = (normal.x == 0);
    const bool mask_y = (normal.y == 0);
    const bool mask_z = (normal.z == 0);

    const int i_src = mask_x ? i_dst : mirror_x * 2 - i_dst;
    const int j_src = mask_y ? j_dst : mirror_y * 2 - j_dst;
    const int k_src = mask_z ? k_dst : mirror_z * 2 - k_dst;

    const int src_idx    = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx    = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);
    const int mirror_idx = DEVICE_VTXBUF_IDX(mirror_x, mirror_y, mirror_z);


    //Same as lnT(), except we are reading the values from the boundary (mirror_idx)
    AcReal lnT_boundary = DCONST(AC_lnT0)+DCONST(AC_gamma)*vba.in[VTXBUF_ENTROPY][mirror_idx]/DCONST(AC_cp_sound)+(DCONST(AC_gamma)-AcReal(1.))*(vba.in[VTXBUF_LNRHO][mirror_idx]-DCONST(AC_lnrho0));

    vba.in[VTXBUF_ENTROPY][dst_idx] = - vba.in[VTXBUF_ENTROPY][src_idx]
                      + 2 * DCONST(AC_cv_sound)
                          * (lnT_boundary - DCONST(AC_lnT0))
                      - (DCONST(AC_cp_sound) -  DCONST(AC_cv_sound))
                          * (vba.in[VTXBUF_LNRHO][src_idx] + vba.in[VTXBUF_LNRHO][dst_idx] - 2*DCONST(AC_lnrho0));
}

AcResult
acKernelEntropyBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                            const int3 dims, VertexBufferArray vba)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_entropy_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vba);
    return AC_SUCCESS;
}


/************************
 *                      *
 *  Dummy test kernels  *
 *                      *
 ************************/

#pragma once
static __global__ void
kernel_add_one_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{
    // The only reason a lot of these are const and not constexpr is DCONST(AC_n*) is not a compile
    // time expression
    const int start_x = (region_id.x == 1    ? NGHOST + DCONST(AC_nx)
                         : region_id.x == -1 ? 0
                                             : NGHOST);
    const int start_y = (region_id.y == 1    ? NGHOST + DCONST(AC_ny)
                         : region_id.y == -1 ? 0
                                             : NGHOST);
    const int start_z = (region_id.z == 1    ? NGHOST + DCONST(AC_nz)
                         : region_id.z == -1 ? 0
                                             : NGHOST);

    const int i_dst = start_x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start_y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start_z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= start_x + dims.x || j_dst >= start_y + dims.y || k_dst >= start_z + dims.z)
        return;

    const int mirror_x = normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1 : normal.x == -1 ? NGHOST : -1;
    const int mirror_y = normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1 : normal.y == -1 ? NGHOST : -1;
    const int mirror_z = normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1 : normal.z == -1 ? NGHOST : -1;

    const bool mask_x = (normal.x == 0);
    const bool mask_y = (normal.y == 0);
    const bool mask_z = (normal.z == 0);

    const int i_src = mask_x ? i_dst : mirror_x * 2 - i_dst;
    const int j_src = mask_y ? j_dst : mirror_y * 2 - j_dst;
    const int k_src = mask_z ? k_dst : mirror_z * 2 - k_dst;

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);

    vtxbuf[dst_idx] = vtxbuf[src_idx] + 1.0f;
}

AcResult
acKernelAddOneBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                            const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_add_one_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}
//Dummy test kernel
#pragma once
static __global__ void
kernel_add_two_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{
    // The only reason a lot of these are const and not constexpr is DCONST(AC_n*) is not a compile
    // time expression
    const int start_x = (region_id.x == 1    ? NGHOST + DCONST(AC_nx)
                         : region_id.x == -1 ? 0
                                             : NGHOST);
    const int start_y = (region_id.y == 1    ? NGHOST + DCONST(AC_ny)
                         : region_id.y == -1 ? 0
                                             : NGHOST);
    const int start_z = (region_id.z == 1    ? NGHOST + DCONST(AC_nz)
                         : region_id.z == -1 ? 0
                                             : NGHOST);

    const int i_dst = start_x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start_y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start_z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= start_x + dims.x || j_dst >= start_y + dims.y || k_dst >= start_z + dims.z)
        return;

    const int mirror_x = normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1 : normal.x == -1 ? NGHOST : -1;
    const int mirror_y = normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1 : normal.y == -1 ? NGHOST : -1;
    const int mirror_z = normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1 : normal.z == -1 ? NGHOST : -1;

    const bool mask_x = (normal.x == 0);
    const bool mask_y = (normal.y == 0);
    const bool mask_z = (normal.z == 0);

    const int i_src = mask_x ? i_dst : mirror_x * 2 - i_dst;
    const int j_src = mask_y ? j_dst : mirror_y * 2 - j_dst;
    const int k_src = mask_z ? k_dst : mirror_z * 2 - k_dst;

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);

    vtxbuf[dst_idx] = vtxbuf[src_idx] + 2.0f;
}

AcResult
acKernelAddTwoBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                            const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_add_two_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}
//Dummy test kernel
#pragma once
static __global__ void
kernel_add_four_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{
    // The only reason a lot of these are const and not constexpr is DCONST(AC_n*) is not a compile
    // time expression
    const int start_x = (region_id.x == 1    ? NGHOST + DCONST(AC_nx)
                         : region_id.x == -1 ? 0
                                             : NGHOST);
    const int start_y = (region_id.y == 1    ? NGHOST + DCONST(AC_ny)
                         : region_id.y == -1 ? 0
                                             : NGHOST);
    const int start_z = (region_id.z == 1    ? NGHOST + DCONST(AC_nz)
                         : region_id.z == -1 ? 0
                                             : NGHOST);

    const int i_dst = start_x + threadIdx.x + blockIdx.x * blockDim.x;
    const int j_dst = start_y + threadIdx.y + blockIdx.y * blockDim.y;
    const int k_dst = start_z + threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_dst >= start_x + dims.x || j_dst >= start_y + dims.y || k_dst >= start_z + dims.z)
        return;

    const int mirror_x = normal.x == 1 ? NGHOST + DCONST(AC_nx) - 1 : normal.x == -1 ? NGHOST : -1;
    const int mirror_y = normal.y == 1 ? NGHOST + DCONST(AC_ny) - 1 : normal.y == -1 ? NGHOST : -1;
    const int mirror_z = normal.z == 1 ? NGHOST + DCONST(AC_nz) - 1 : normal.z == -1 ? NGHOST : -1;

    const bool mask_x = (normal.x == 0);
    const bool mask_y = (normal.y == 0);
    const bool mask_z = (normal.z == 0);

    const int i_src = mask_x ? i_dst : mirror_x * 2 - i_dst;
    const int j_src = mask_y ? j_dst : mirror_y * 2 - j_dst;
    const int k_src = mask_z ? k_dst : mirror_z * 2 - k_dst;

    const int src_idx = DEVICE_VTXBUF_IDX(i_src, j_src, k_src);
    const int dst_idx = DEVICE_VTXBUF_IDX(i_dst, j_dst, k_dst);

    vtxbuf[dst_idx] = vtxbuf[src_idx] + 4.0f;
}

AcResult
acKernelAddFourBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                            const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_add_four_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}

}//extern "C"
