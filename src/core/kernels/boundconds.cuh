#pragma once
static __global__ void
kernel_symmetric_boundconds(const int3 normal, const int3 dims, AcReal* vtxbuf)
{
    // The only reason a lot of these are const and not constexpr is DCONST(AC_n*) is not a compile
    // time expression
    const int start_x = (normal.x == 1 ? NGHOST + DCONST(AC_nx) : normal.x == -1 ? 0 : NGHOST);
    const int start_y = (normal.y == 1 ? NGHOST + DCONST(AC_ny) : normal.y == -1 ? 0 : NGHOST);
    const int start_z = (normal.z == 1 ? NGHOST + DCONST(AC_nz) : normal.z == -1 ? 0 : NGHOST);

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
acKernelSymmetricBoundconds(const cudaStream_t stream, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil((dims.x) / (float)tpb.x),
                   (unsigned int)ceil((dims.y) / (float)tpb.y),
                   (unsigned int)ceil((dims.z) / (float)tpb.z));

    kernel_symmetric_boundconds<<<bpg, tpb, 0, stream>>>(normal, dims, vtxbuf);
    return AC_SUCCESS;
}
