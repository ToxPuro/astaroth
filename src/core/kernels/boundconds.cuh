#pragma once

// TODO remove clang-format on/off
// clang-format off


extern "C" {
/**************************
 *                        *
 *   Generic boundconds   *
 *      (Any vtxbuf)      *
 *                        *
 **************************/



static __global__ void
kernel_symmetric_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{
    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + VAL(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + VAL(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
#if TWO_D == 0
                              (region_id.z == 1 ? NGHOST + VAL(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};
#else
    			       0};
#endif

    const int3 boundary = int3{normal.x == 1 ? NGHOST + VAL(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + VAL(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
#if TWO_D == 0
                               normal.z == 1 ? NGHOST + VAL(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};
#else
				0};
#endif

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vtxbuf[ghost_idx] = vtxbuf[domain_idx];
    }
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

static __global__ void
kernel_antisymmetric_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                                AcReal* vtxbuf)
{
    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + VAL(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + VAL(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
#if TWO_D == 0
                              (region_id.z == 1 ? NGHOST + VAL(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};
#else
    			       0};
#endif

    const int3 boundary = int3{normal.x == 1 ? NGHOST + VAL(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + VAL(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
#if TWO_D == 0
                               normal.z == 1 ? NGHOST + VAL(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};
#else
				0};
#endif

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vtxbuf[ghost_idx] = -vtxbuf[domain_idx];
    }
}

AcResult
acKernelAntiSymmetricBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                                const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_antisymmetric_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}

// Boundcond "a2"
// Does not set the boundary value itself, mainly used for density

static __global__ void
kernel_a2_boundconds(const int3 region_id, const int3 normal, const int3 dims, AcReal* vtxbuf)
{

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + VAL(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + VAL(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
#if TWO_D == 0
                              (region_id.z == 1 ? NGHOST + VAL(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};
#else
    			       0};
#endif

    const int3 boundary = int3{normal.x == 1 ? NGHOST + VAL(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + VAL(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
#if TWO_D == 0
                               normal.z == 1 ? NGHOST + VAL(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};
#else
				0};
#endif

    const int boundary_idx = DEVICE_VTXBUF_IDX(boundary.x, boundary.y, boundary.z);

    AcReal boundary_val = vtxbuf[boundary_idx];

    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vtxbuf[ghost_idx] = 2 * boundary_val - vtxbuf[domain_idx];
    }
}

AcResult
acKernelA2Boundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                     const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_a2_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}

static __global__ void
kernel_const_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                        AcReal* vtxbuf, AcRealParam const_value)
{
    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + VAL(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + VAL(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
#if TWO_D == 0
                              (region_id.z == 1 ? NGHOST + VAL(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};
#else
    			       0};
#endif

    const int3 boundary = int3{normal.x == 1 ? NGHOST + VAL(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + VAL(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
#if TWO_D == 0
                               normal.z == 1 ? NGHOST + VAL(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};
#else
				0};
#endif

    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        ghost  = ghost + normal;

        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vtxbuf[ghost_idx] = DCONST(const_value);
    }
}

AcResult
acKernelConstBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                        const int3 dims, AcReal* vtxbuf, AcRealParam const_value)
{
    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_const_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf, const_value);
    return AC_SUCCESS;
}

#ifdef AC_INTEGRATION_ENABLED
// Constant derivative at boundary
// Sets the normal derivative at the boundary to a value

static __global__ void
kernel_prescribed_derivative_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                                        AcReal* vtxbuf, AcRealParam der_val_param)
{

    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + VAL(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + VAL(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
#if TWO_D == 0
                              (region_id.z == 1 ? NGHOST + VAL(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};
#else
    			       0};
#endif

    const int3 boundary = int3{normal.x == 1 ? NGHOST + VAL(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + VAL(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
#if TWO_D == 0
                               normal.z == 1 ? NGHOST + VAL(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};
#else
				0};
#endif

    int3 domain = boundary;
    int3 ghost  = boundary;

    const AcReal d = normal.x != 0 ? VAL(AC_dsx) :
	   	     normal.y != 0 ? VAL(AC_dsy) :
#if TWO_D == 0
		     normal.z != 0 ? VAL(AC_dsz) :
#endif
		     0.0;
    const AcReal direction = normal.x != 0 ? normal.x : 
    			     normal.y != 0 ? normal.y :
			     normal.z != 0 ? normal.z : 0.0;
    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        AcReal distance = AcReal(2 * (i + 1)) * d;
        // Otherwise resulting derivatives are of different sign and opposite edges.
        if (direction < 0.0) {
            distance = -distance;
        }

        vtxbuf[ghost_idx] = vtxbuf[domain_idx] + distance * VAL(der_val_param);
    }
}

AcResult
acKernelPrescribedDerivativeBoundconds(const cudaStream_t stream, const int3 region_id,
                                       const int3 normal, const int3 dims, AcReal* vtxbuf,
                                       AcRealParam der_val_param)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_prescribed_derivative_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims,
                                                                     vtxbuf, der_val_param);
    return AC_SUCCESS;
}
#endif

/*************************
 *                       *
 *  Velocity boundconds  *
 *                       *
 *************************/

static __global__ void
kernel_outflow_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{
    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + VAL(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + VAL(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
#if TWO_D == 0
                              (region_id.z == 1 ? NGHOST + VAL(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};
#else
    			       0};
#endif

    const int3 boundary = int3{normal.x == 1 ? NGHOST + VAL(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + VAL(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
#if TWO_D == 0
                               normal.z == 1 ? NGHOST + VAL(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};
#else
				0};
#endif

    int3 domain = boundary;
    int3 ghost  = boundary;
    const int boundary_idx = DEVICE_VTXBUF_IDX(boundary.x, boundary.y, boundary.z);
    const AcReal uudir = normal.x != 0 ?  vtxbuf[boundary_idx]*normal.x :
                   normal.y != 0 ?  vtxbuf[boundary_idx]*normal.y :
                   normal.z != 0 ?  vtxbuf[boundary_idx]*normal.z : 0.0;
    const AcReal sign = uudir >= 0.0 ?  1.0 : -1.0;
    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);
        

        vtxbuf[ghost_idx] = sign*vtxbuf[domain_idx];
    }
}

AcResult
acKernelOutflowBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                          const int3 dims, AcReal* vtxbuf)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_outflow_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}

static __global__ void
kernel_inflow_boundconds(const int3 region_id, const int3 normal, const int3 dims,
                            AcReal* vtxbuf)
{
    const int3 vertexIdx = (int3){
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z,
    };

    if (vertexIdx.x >= dims.x || vertexIdx.y >= dims.y || vertexIdx.z >= dims.z) {
        return;
    }

    const int3 start = (int3){(region_id.x == 1 ? NGHOST + VAL(AC_nx)
                                                : region_id.x == -1 ? 0 : NGHOST),
                              (region_id.y == 1 ? NGHOST + VAL(AC_ny)
                                                : region_id.y == -1 ? 0 : NGHOST),
#if TWO_D == 0
                              (region_id.z == 1 ? NGHOST + VAL(AC_nz)
                                                : region_id.z == -1 ? 0 : NGHOST)};
#else
    			       0};
#endif

    const int3 boundary = int3{normal.x == 1 ? NGHOST + VAL(AC_nx) - 1
                                             : normal.x == -1 ? NGHOST : start.x + vertexIdx.x,
                               normal.y == 1 ? NGHOST + VAL(AC_ny) - 1
                                             : normal.y == -1 ? NGHOST : start.y + vertexIdx.y,
#if TWO_D == 0
                               normal.z == 1 ? NGHOST + VAL(AC_nz) - 1
                                             : normal.z == -1 ? NGHOST : start.z + vertexIdx.z};
#else
				0};
#endif

    int3 domain = boundary;
    int3 ghost  = boundary;
    const int boundary_idx = DEVICE_VTXBUF_IDX(boundary.x, boundary.y, boundary.z);
    const AcReal uudir = normal.x != 0 ?  vtxbuf[boundary_idx]*normal.x :
                   normal.y != 0 ?  vtxbuf[boundary_idx]*normal.y :
                   normal.z != 0 ?  vtxbuf[boundary_idx]*normal.z : 0.0;
    const AcReal sign = uudir >= 0.0 ?  -1.0 : 1.0;
    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);
         

        vtxbuf[ghost_idx] = sign*vtxbuf[domain_idx];
    }
}

AcResult
acKernelInflowBoundconds(const cudaStream_t stream, const int3 region_id, const int3 normal,
                          const int3 dims, AcReal* vtxbuf)
{
 
    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_inflow_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims, vtxbuf);
    return AC_SUCCESS;
}
} // extern "C"

// clang-format on
