#pragma once

// TODO remove clang-format on/off
// clang-format off

#define DEVICE_INLINE __device__ __forceinline__

DEVICE_INLINE AcReal
VAL(const AcRealParam& param)
{
	return DCONST(param);
}

DEVICE_INLINE AcReal
VAL(const AcReal& val)
{
	return val;
}

DEVICE_INLINE int
VAL(const AcIntParam& param)
{
	return DCONST(param);
}

DEVICE_INLINE int
VAL(const int& val)
{
	return val;
}

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

        vtxbuf[ghost_idx] = const_value;
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

#ifdef AC_INTEGRATION_ENABLED
/************************
 *                      *
 *  Entropy boundconds  *
 *                      *
 ************************/

#if LENTROPY
static __global__ void
kernel_entropy_const_temperature_boundconds(const int3 region_id, const int3 normal,
                                            const int3 dims, VertexBufferArray vba)
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

    const AcReal lnrho_diff   = vba.in[VTXBUF_LNRHO][boundary_idx] - VAL(AC_lnrho0);
    const AcReal gas_constant = VAL(AC_cp_sound) - VAL(AC_cv_sound);

    // Same as lnT(), except we are reading the values from the boundary
    const AcReal lnT_boundary = VAL(AC_lnT0) +
                          VAL(AC_gamma) * vba.in[VTXBUF_ENTROPY][boundary_idx] /
                              VAL(AC_cp_sound) +
                          (VAL(AC_gamma) - AcReal(1.)) * lnrho_diff;

    const AcReal tmp = AcReal(2.0) * VAL(AC_cv_sound) * (lnT_boundary - VAL(AC_lnT0));

    vba.in[VTXBUF_ENTROPY][boundary_idx] = AcReal(0.5) * tmp - gas_constant * lnrho_diff;

    // Set the values in the halo
    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        vba.in[VTXBUF_ENTROPY][ghost_idx] = -vba.in[VTXBUF_ENTROPY][domain_idx] + tmp -
                                            gas_constant * (vba.in[VTXBUF_LNRHO][domain_idx] +
                                                            vba.in[VTXBUF_LNRHO][ghost_idx] -
                                                            2 * VAL(AC_lnrho0));
    }
}

AcResult
acKernelEntropyConstantTemperatureBoundconds(const cudaStream_t stream, const int3 region_id,
                                             const int3 normal, const int3 dims,
                                             VertexBufferArray vba)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_entropy_const_temperature_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims,
                                                                         vba);
    return AC_SUCCESS;
}

static __global__ void
kernel_entropy_blackbody_radiation_kramer_conductivity_boundconds(const int3 region_id,
                                                                  const int3 normal,
                                                                  const int3 dims,
                                                                  VertexBufferArray vba)
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

    const AcReal rho_boundary = exp(vba.in[VTXBUF_LNRHO][boundary_idx]);

    const AcReal gamma_m1 = VAL(AC_gamma) - AcReal(1.0);
    const AcReal cv1      = VAL(AC_gamma) / VAL(AC_cp_sound);

    // cs20*exp(gamma_m1*(f(l1,:,:,ilnrho)-lnrho0)+cv1*f(l1,:,:,iss))/(gamma_m1*cp)
    const AcReal T_boundary = VAL(AC_cs2_sound) *
                        exp(gamma_m1 * (vba.in[VTXBUF_LNRHO][boundary_idx] - VAL(AC_lnrho0)) +
                            cv1 * vba.in[VTXBUF_ENTROPY][boundary_idx]) /
                        gamma_m1 * VAL(AC_cp_sound);

    // dlnrhodx_yz= coeffs_1_x(1)*(f(l1+1,:,:,ilnrho)-f(l1-1,:,:,ilnrho)) &
    //            +coeffs_1_x(2)*(f(l1+2,:,:,ilnrho)-f(l1-2,:,:,ilnrho)) &
    //            +coeffs_1_x(3)*(f(l1+3,:,:,ilnrho)-f(l1-3,:,:,ilnrho))

    const AcReal c[3] = {(AcReal(1.) / (AcReal(0.04908738521))) * (AcReal(3.) / AcReal(4.)),
                   (AcReal(1.) / (AcReal(0.04908738521))) * (-AcReal(3.) / AcReal(20.)),
                   (AcReal(1.) / (AcReal(0.04908738521))) * (AcReal(1.) / AcReal(60.))};

    AcReal der_lnrho_boundary = 0;

    int3 left       = boundary;
    int3 right      = boundary;
    int3 abs_normal = int3{abs(normal.x), abs(normal.y), abs(normal.z)};

    for (int i = 0; i < 3; i++) {
        left          = left - abs_normal;
        right         = right - abs_normal;
        int left_idx  = DEVICE_VTXBUF_IDX(left.x, left.y, left.z);
        int right_idx = DEVICE_VTXBUF_IDX(right.x, right.y, right.z);
        der_lnrho_boundary += c[i] *
                              (vba.in[VTXBUF_LNRHO][right_idx] - vba.in[VTXBUF_LNRHO][left_idx]);
    }

    // dsdx_yz=-cv*((sigmaSBt/hcond0_kramers)*TT_yz**(3-6.5*nkramers)*rho_yz**(2.*nkramers) &
    //        +gamma_m1*dlnrhodx_yz)

    const AcReal der_ss_boundary = -VAL(AC_cv_sound) *
                                 (VAL(AC_sigma_SBt) / VAL(AC_hcond0_kramers)) *
                                 pow(T_boundary, AcReal(3.0) - AcReal(6.5) * VAL(AC_n_kramers)) *
                                 pow(rho_boundary, AcReal(2.0) * VAL(AC_n_kramers)) +
                             gamma_m1 * der_lnrho_boundary;

    const AcReal d = normal.x != 0 ? VAL(AC_dsx) :
	   	     normal.y != 0 ? VAL(AC_dsy) :
#if TWO_D == 0
		      normal.z != 0 ? VAL(AC_dsz) :
#endif
		     0.0;
    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        const AcReal distance = AcReal(2 * (i + 1)) * d;

        vba.in[VTXBUF_ENTROPY][ghost_idx] = vba.in[VTXBUF_ENTROPY][domain_idx] -
                                            distance * der_ss_boundary;
    }
}

AcResult
acKernelEntropyBlackbodyRadiationKramerConductivityBoundconds(const cudaStream_t stream,
                                                              const int3 region_id,
                                                              const int3 normal, const int3 dims,
                                                              VertexBufferArray vba)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_entropy_blackbody_radiation_kramer_conductivity_boundconds<<<bpg, tpb, 0,
                                                                        stream>>>(region_id, normal,
                                                                                  dims, vba);
    return AC_SUCCESS;
}

// Prescribed heat flux

static __global__ void
kernel_entropy_prescribed_heat_flux_boundconds(const int3 region_id, const int3 normal,
                                               const int3 dims, VertexBufferArray vba,
                                               AcRealParam F_param)
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

#if (L_HEAT_CONDUCTION_CHICONST) || (L_HEAT_CONDUCTION_KRAMERS)
    AcReal rho_boundary = exp(vba.in[VTXBUF_LNRHO][boundary_idx]);
#endif

    AcReal cp = VAL(AC_cp_sound);
    AcReal cv = VAL(AC_cv_sound);

    AcReal gamma_m1 = VAL(AC_gamma) - AcReal(1.0);
    AcReal cv1      = VAL(AC_gamma) / cp;

    // cs20*exp(gamma_m1*(f(l1,:,:,ilnrho)-lnrho0)+cv1*f(l1,:,:,iss))
    AcReal cs2_boundary = VAL(AC_cs2_sound) *
                          exp(gamma_m1 * (vba.in[VTXBUF_LNRHO][boundary_idx] - VAL(AC_lnrho0)) +
                              cv1 * vba.in[VTXBUF_ENTROPY][boundary_idx]);

    AcReal F_boundary = VAL(F_param);
#if (L_HEAT_CONDUCTION_CHICONST)
    // TODO: use chi in the calculation
    AcReal chi = VAL(AC_chi);
    AcReal tmp = F_boundary / (rho_boundary * chi * cs2_boundary);
#elif (L_HEAT_CONDUCTION_KRAMERS)
    AcReal n_kramers      = VAL(AC_n_kramers);
    AcReal hcond0_kramers = VAL(AC_hcond0_kramers);
    AcReal tmp            = F_boundary * pow(rho_boundary, AcReal(2.0) * n_kramers) *
                 pow(cp * gamma_m1, AcReal(6.5) * n_kramers) /
                 (hcond0_kramers * pow(cs2_boundary, AcReal(6.5) * n_kramers + AcReal(1.0)));
#else
    // NOTE: FbotKbot, FtopKtop, ... = F_param, just like Fbot, Ftop, ... = F_param
    // If both are needed, it would be preferable if they were separate boundary conditions
    // and that the switch would be between them in the main program that creates the task graph
    AcReal tmp            = F_boundary / cs2_boundary;
#endif

    const AcReal d = normal.x != 0 ? VAL(AC_dsx) :
	   	     normal.y != 0 ? VAL(AC_dsy) :
#if TWO_D == 0
		      normal.z != 0 ? VAL(AC_dsz) :
#endif
		     0.0;
    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        AcReal distance = AcReal(2 * (i + 1)) * d;

        AcReal rho_diff = vba.in[VTXBUF_LNRHO][ghost_idx] - vba.in[VTXBUF_LNRHO][domain_idx];
        vba.in[VTXBUF_ENTROPY][ghost_idx] = vba.in[VTXBUF_ENTROPY][domain_idx] +
                                            cp * (cp - cv) * (rho_diff + distance * tmp);


    }
}

AcResult
acKernelEntropyPrescribedHeatFluxBoundconds(const cudaStream_t stream, const int3 region_id,
                                            const int3 normal, const int3 dims,
                                            VertexBufferArray vba, AcRealParam F_param)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));
 
    //printf("ENTROPY BOUDNARY asdasasdas");
    kernel_entropy_prescribed_heat_flux_boundconds<<<bpg, tpb, 0, stream>>>(region_id, normal, dims,
                                                                            vba, F_param);
    return AC_SUCCESS;
}

// Prescribed normal + turbulent heat flux

static __global__ void
kernel_entropy_prescribed_normal_and_turbulent_heat_flux_boundconds(
    const int3 region_id, const int3 normal, const int3 dims, VertexBufferArray vba,
    AcRealParam hcond_param, AcRealParam F_param)
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

    const AcReal gamma_m1 = VAL(AC_gamma) - AcReal(1.0);
    const AcReal cv1      = VAL(AC_gamma) / VAL(AC_cp_sound);

    // cs20*exp(gamma_m1*(f(l1,:,:,ilnrho)-lnrho0)+cv1*f(l1,:,:,iss))/(gamma_m1*cp)
    const AcReal T_boundary = VAL(AC_cs2_sound) *
                        exp(gamma_m1 * (vba.in[VTXBUF_LNRHO][boundary_idx] - VAL(AC_lnrho0)) +
                            cv1 * vba.in[VTXBUF_ENTROPY][boundary_idx]) /
                        gamma_m1 * VAL(AC_cp_sound);

    AcReal rho_boundary = exp(vba.in[VTXBUF_LNRHO][boundary_idx]);
#if (L_HEAT_CONDUCTION_CHICONST) || (L_HEAT_CONDUCTION_KRAMERS)
    const AcReal cv           = VAL(AC_cv_sound);
#endif

#if (L_HEAT_CONDUCTION_CHICONST)
    // TODO: use chi in the calculation
    const AcReal chi = VAL(AC_chi);
    const AcReal K   = chi * rho_boundary * cv;
#elif (L_HEAT_CONDUCTION_KRAMERS)
    const AcReal n_kramers      = VAL(AC_n_kramers);
    const AcReal hcond0_kramers = VAL(AC_hcond0_kramers);
    const AcReal K              = hcond0_kramers * pow(T_boundary, AcReal(6.5) * n_kramers) /
               pow(rho_boundary, AcReal(2.0) * n_kramers);
#else
    const AcReal hcond_boundary = VAL(hcond_param);
    const AcReal K              = hcond_boundary;
#endif

    const AcReal F_boundary  = VAL(F_param);
    const AcReal chi_t_prof1 = VAL(AC_chi_t_prof1);
    const AcReal chi_t       = VAL(AC_chi_t);

    const AcReal der_s_boundary = (F_boundary / T_boundary) /
                            (chi_t_prof1 * chi_t * rho_boundary + K * cv1);


    const AcReal d = normal.x != 0 ? VAL(AC_dsx) :
	   	     normal.y != 0 ? VAL(AC_dsy)
#if TWO_D == 0
		     : normal.z != 0 ? VAL(AC_dsz) : 0.0
#endif
		     ;
    int3 domain = boundary;
    int3 ghost  = boundary;

    for (size_t i = 0; i < NGHOST; i++) {
        domain = domain - normal;
        ghost  = ghost + normal;

        const int domain_idx = DEVICE_VTXBUF_IDX(domain.x, domain.y, domain.z);
        const int ghost_idx  = DEVICE_VTXBUF_IDX(ghost.x, ghost.y, ghost.z);

        const AcReal der_lnrho = vba.in[VTXBUF_LNRHO][domain_idx] - vba.in[VTXBUF_LNRHO][ghost_idx];

        const AcReal distance = AcReal(2 * (i + 1)) * d;

        vba.in[VTXBUF_ENTROPY][ghost_idx] = vba.in[VTXBUF_ENTROPY][domain_idx] +
                                            K * gamma_m1 * der_lnrho /
                                                (K * cv1 + chi_t_prof1 * chi_t * rho_boundary) +
                                            distance * der_s_boundary;
    }
}

AcResult
acKernelEntropyPrescribedNormalAndTurbulentHeatFluxBoundconds(
    const cudaStream_t stream, const int3 region_id, const int3 normal, const int3 dims,
    VertexBufferArray vba, AcRealParam hcond_param, AcRealParam F_param)
{

    const dim3 tpb(8, 8, 8);
    const dim3 bpg((unsigned int)ceil(dims.x / (double)tpb.x),
                   (unsigned int)ceil(dims.y / (double)tpb.y),
                   (unsigned int)ceil(dims.z / (double)tpb.z));

    kernel_entropy_prescribed_normal_and_turbulent_heat_flux_boundconds<<<
        bpg, tpb, 0, stream>>>(region_id, normal, dims, vba, hcond_param, F_param);
    return AC_SUCCESS;
}
#endif

#else
AcResult
acKernelPrescribedDerivativeBoundconds(const cudaStream_t stream, const int3 region_id,
                                       const int3 normal, const int3 dims, AcReal* vtxbuf,
                                       AcRealParam der_val_param)
{
    fprintf(stderr, "acKernelPrescribedDerivativeBoundconds() called but AC_INTEGRATION_ENABLED "
                    "was false\n");
    return AC_FAILURE;
}

#endif // AC_INTEGRATION_ENABLED
} // extern "C"

// clang-format on
