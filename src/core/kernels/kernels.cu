#include "kernels.h"

#include "acc_runtime.cu"

static_assert(NUM_VTXBUF_HANDLES > 0, "ERROR: At least one uniform ScalarField must be declared.");

static __global__ void
dummy_kernel(void)
{
    DCONST((AcIntParam)0);
    DCONST((AcInt3Param)0);
    DCONST((AcRealParam)0);
    DCONST((AcReal3Param)0);
    acComplex a = exp(acComplex(1, 1) * AcReal(1));
    a* a;
}

AcResult
acKernelDummy(void)
{
    dummy_kernel<<<1, 1>>>();
    ERRCHK_CUDA_KERNEL_ALWAYS();
    return AC_SUCCESS;
}

AcResult
acDeviceLoadScalarUniform(const Device device, const Stream stream, const AcRealParam param,
                          const AcReal value)
{
    cudaSetDevice(device->id);
    if (param < 0 || param >= NUM_REAL_PARAMS) {
        fprintf(stderr, "WARNING: invalid AcRealParam %d.\n", param);
        return AC_FAILURE;
    }

    if (!is_valid(value)) {
        fprintf(stderr,
                "WARNING: Passed an invalid value %g to device constant %s. "
                "Skipping.\n",
                (double)value, realparam_names[param]);
        return AC_FAILURE;
    }

    const size_t offset = (size_t)&d_mesh_info.real_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadVectorUniform(const Device device, const Stream stream, const AcReal3Param param,
                          const AcReal3 value)
{
    cudaSetDevice(device->id);
    if (param < 0 || param >= NUM_REAL3_PARAMS) {
        fprintf(stderr, "WARNING: invalid AcReal3Param %d\n", param);
        return AC_FAILURE;
    }

    if (!is_valid(value)) {
        fprintf(stderr,
                "WARNING: Passed an invalid value (%g, %g, %g) to device constant "
                "%s. Skipping.\n",
                (double)value.x, (double)value.y, (double)value.z, real3param_names[param]);
        return AC_FAILURE;
    }

    const size_t offset = (size_t)&d_mesh_info.real3_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acLoadIntUniform(const cudaStream_t stream, const AcIntParam param, const int value)
{
    if (param < 0 || param >= NUM_INT_PARAMS) {
        fprintf(stderr, "WARNING: invalid AcIntParam %d\n", param);
        return AC_FAILURE;
    }

    if (!is_valid(value)) {
        fprintf(stderr,
                "WARNING: Passed an invalid value %d to device constant %s. "
                "Skipping.\n",
                value, intparam_names[param]);
        return AC_FAILURE;
    }

    const size_t offset = (size_t)&d_mesh_info.int_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, stream));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadIntUniform(const Device device, const Stream stream, const AcIntParam param,
                       const int value)
{
    cudaSetDevice(device->id);
    return acLoadIntUniform(device->streams[stream], param, value);
}

AcResult
acDeviceLoadInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                        const int3 value)
{
    cudaSetDevice(device->id);
    if (param < 0 || param >= NUM_INT3_PARAMS) {
        fprintf(stderr, "WARNING: invalid AcInt3Param %d\n", param);
        return AC_FAILURE;
    }

    if (!is_valid(value.x) || !is_valid(value.y) || !is_valid(value.z)) {
        fprintf(stderr,
                "WARNING: Passed an invalid value (%d, %d, %def) to device "
                "constant %s. "
                "Skipping.\n",
                value.x, value.y, value.z, int3param_names[param]);
        return AC_FAILURE;
    }

    const size_t offset = (size_t)&d_mesh_info.int3_params[param] - (size_t)&d_mesh_info;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(d_mesh_info, &value, sizeof(value), offset,
                                        cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceLoadMeshInfo(const Device device, const AcMeshInfo config)
{
    cudaSetDevice(device->id);

    AcMeshInfo device_config = config;
    acHostUpdateBuiltinParams(&device_config);

    ERRCHK_ALWAYS(device_config.int_params[AC_nx] == device->local_config.int_params[AC_nx]);
    ERRCHK_ALWAYS(device_config.int_params[AC_ny] == device->local_config.int_params[AC_ny]);
    ERRCHK_ALWAYS(device_config.int_params[AC_nz] == device->local_config.int_params[AC_nz]);
    ERRCHK_ALWAYS(device_config.int_params[AC_multigpu_offset] ==
                  device->local_config.int_params[AC_multigpu_offset]);

    for (int i = 0; i < NUM_INT_PARAMS; ++i)
        acDeviceLoadIntUniform(device, STREAM_DEFAULT, (AcIntParam)i, device_config.int_params[i]);

    for (int i = 0; i < NUM_INT3_PARAMS; ++i)
        acDeviceLoadInt3Uniform(device, STREAM_DEFAULT, (AcInt3Param)i,
                                device_config.int3_params[i]);

    for (int i = 0; i < NUM_REAL_PARAMS; ++i)
        acDeviceLoadScalarUniform(device, STREAM_DEFAULT, (AcRealParam)i,
                                  device_config.real_params[i]);

    for (int i = 0; i < NUM_REAL3_PARAMS; ++i)
        acDeviceLoadVectorUniform(device, STREAM_DEFAULT, (AcReal3Param)i,
                                  device_config.real3_params[i]);

    return AC_SUCCESS;
}

AcResult
acDeviceLoadStencils(const Device device, const Stream stream,
                     AcReal stencil[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    cudaSetDevice(device->id);

    const size_t bytes = sizeof(stencil[0][0][0][0]) * NUM_STENCILS * STENCIL_DEPTH *
                         STENCIL_HEIGHT * STENCIL_WIDTH;
    ERRCHK_CUDA(cudaMemcpyToSymbolAsync(stencils, stencil, bytes, 0, cudaMemcpyHostToDevice,
                                        device->streams[stream]));

    return AC_FAILURE;
}

// Built-in kernels
#include "boundconds.cuh"
#include "boundconds_miikka_GBC.cuh"
#include "packing.cuh"
#include "reductions.cuh"

AcResult
acKernel(const KernelParameters params, VertexBufferArray vba)
{
#ifndef AC_step_number
    (void)params; // Unused
    (void)vba;    // Unused
    ERROR("acKernel() called but AC_step_number not defined!");
    return AC_FAILURE;
#else

    // TODO: Why is AC_step_number loaded here??
    acLoadIntUniform(params.stream, AC_step_number, params.step_number);
    acLaunchKernel(params.kernel, params.stream, params.start, params.end, vba);
    return AC_SUCCESS;
#endif
}
