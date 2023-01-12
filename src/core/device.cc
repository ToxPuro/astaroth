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
#include "astaroth.h"

#include "kernels/kernels.h"

#define GEN_DEVICE_FUNC_HOOK(ID)                                                                   \
    AcResult acDevice_##ID(const Device device, const Stream stream, const int3 start,             \
                           const int3 end)                                                         \
    {                                                                                              \
        cudaSetDevice(device->id);                                                                 \
        return acKernel_##ID(KernelParameters{device->streams[stream], 0, start, end},             \
                             device->vba);                                                         \
    }

AcResult
acDevicePrintInfo(const Device device)
{
    cudaSetDevice(device->id);
    const int device_id = device->id;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    printf("--------------------------------------------------\n");
    printf("Device Number: %d\n", device_id);
    const size_t bus_id_max_len = 128;
    char bus_id[bus_id_max_len];
    cudaDeviceGetPCIBusId(bus_id, bus_id_max_len, device_id);
    printf("  PCI bus ID: %s\n", bus_id);
    printf("    Device name: %s\n", props.name);
    printf("    Compute capability: %d.%d\n", props.major, props.minor);

    // Compute
    printf("  Compute\n");
    printf("    Clock rate (GHz): %g\n", props.clockRate / 1e6); // KHz -> GHz
    printf("    Stream processors: %d\n", props.multiProcessorCount);
#if !AC_USE_HIP
    printf("    SP to DP flops performance ratio: %d:1\n", props.singleToDoublePrecisionPerfRatio);
#endif
    printf(
        "    Compute mode: %d\n",
        (int)props
            .computeMode); // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g7eb25f5413a962faad0956d92bae10d0
    // Memory
    printf("  Global memory\n");
    printf("    Memory Clock Rate (MHz): %d\n", props.memoryClockRate / (1000));
    printf("    Memory Bus Width (bits): %d\n", props.memoryBusWidth);
    printf("    Peak Memory Bandwidth (GiB/s): %f\n",
           2 * (props.memoryClockRate * 1e3) * props.memoryBusWidth / (8. * 1024. * 1024. * 1024.));
    printf("    ECC enabled: %d\n", props.ECCEnabled);

    // Memory usage
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const size_t used_bytes = total_bytes - free_bytes;
    printf("    Total global mem: %.2f GiB\n", props.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("    Gmem used (GiB): %.2f\n", used_bytes / (1024.0 * 1024 * 1024));
    printf("    Gmem memory free (GiB): %.2f\n", free_bytes / (1024.0 * 1024 * 1024));
    printf("    Gmem memory total (GiB): %.2f\n", total_bytes / (1024.0 * 1024 * 1024));
    printf("  Caches\n");
#if !AC_USE_HIP
    printf("    Local L1 cache supported: %d\n", props.localL1CacheSupported);
    printf("    Global L1 cache supported: %d\n", props.globalL1CacheSupported);
#endif
    printf("    L2 size: %d KiB\n", props.l2CacheSize / (1024));
    // MV: props.totalConstMem and props.sharedMemPerBlock cause assembler error
    // MV: while compiling in TIARA gp cluster. Therefore commeted out.
    //!!    printf("    Total const mem: %ld KiB\n", props.totalConstMem / (1024));
    //!!    printf("    Shared mem per block: %ld KiB\n", props.sharedMemPerBlock / (1024));
    printf("  Other\n");
    printf("    Warp size: %d\n", props.warpSize);
    // printf("    Single to double perf. ratio: %dx\n",
    // props.singleToDoublePrecisionPerfRatio); //Not supported with older CUDA
    // versions
#if !AC_USE_HIP
    printf("    Stream priorities supported: %d\n", props.streamPrioritiesSupported);
#endif
    printf("    AcReal precision: %lu bits\n", 8 * sizeof(AcReal));
    printf("--------------------------------------------------\n");

    return AC_SUCCESS;
}

AcResult
acDeviceLoadScalarUniform(const Device device, const Stream stream, const AcRealParam param,
                          const AcReal value)
{
    cudaSetDevice(device->id);
    return acLoadRealUniform(device->streams[stream], param, value);
}

AcResult
acDeviceLoadVectorUniform(const Device device, const Stream stream, const AcReal3Param param,
                          const AcReal3 value)
{
    cudaSetDevice(device->id);
    return acLoadReal3Uniform(device->streams[stream], param, value);
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
    return acLoadInt3Uniform(device->streams[stream], param, value);
}

AcResult
acDeviceStoreScalarUniform(const Device device, const Stream stream, const AcRealParam param,
                           AcReal* value)
{
    cudaSetDevice(device->id);
    return acStoreRealUniform(device->streams[stream], param, value);
}

AcResult
acDeviceStoreVectorUniform(const Device device, const Stream stream, const AcReal3Param param,
                           AcReal3* value)
{
    cudaSetDevice(device->id);
    return acStoreReal3Uniform(device->streams[stream], param, value);
}

AcResult
acDeviceStoreIntUniform(const Device device, const Stream stream, const AcIntParam param,
                        int* value)
{
    cudaSetDevice(device->id);
    return acStoreIntUniform(device->streams[stream], param, value);
}

AcResult
acDeviceStoreInt3Uniform(const Device device, const Stream stream, const AcInt3Param param,
                         int3* value)
{
    cudaSetDevice(device->id);
    return acStoreInt3Uniform(device->streams[stream], param, value);
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

    // OL: added this assignment to make sure that whenever we load a new config,
    // it's updated on both the host Device structure, and the GPU
    device->local_config = device_config;
    return AC_SUCCESS;
}

AcResult
acDeviceSynchronizeStream(const Device device, const Stream stream)
{
    cudaSetDevice(device->id);
    if (stream == STREAM_ALL) {
        cudaDeviceSynchronize();
    }
    else {
        cudaStreamSynchronize(device->streams[stream]);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceCreate(const int id, const AcMeshInfo device_config, Device* device_handle)
{
    // Check
    int count;
    cudaGetDeviceCount(&count);
    ERRCHK_ALWAYS(id < count);

    cudaSetDevice(id);
// cudaDeviceReset(); // Would be good for safety, but messes stuff up if we want to emulate
// multiple devices with a single GPU
#if AC_DOUBLE_PRECISION
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
    // cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
    // cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    // Create Device
    struct device_s* device = (struct device_s*)malloc(sizeof(*device));
    ERRCHK_ALWAYS(device);

    device->id           = id;
    device->local_config = device_config;

#if AC_VERBOSE
    acDevicePrintInfo(device);
    printf("Trying to run a dummy kernel. If this fails, make sure that your\n"
           "device supports the GPU architecture you are compiling for.\n");

    // Check that the code was compiled for the proper GPU architecture

    printf("Running a test kernel... ");
    fflush(stdout);
#endif

    acKernelDummy();
#if AC_VERBOSE
    printf("\x1B[32m%s\x1B[0m\n", "OK!");
    fflush(stdout);
#endif

    // Concurrency
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithPriority(&device->streams[i], cudaStreamNonBlocking, i);
    }

    // Memory
    // VBA in/out
    device->vba = acVBACreate(acVertexBufferSize(device_config));
    /*
    // VBA Profiles
    const size_t profile_size_bytes = sizeof(AcReal) * max(device_config.int_params[AC_mx],
                                                           max(device_config.int_params[AC_my],
                                                               device_config.int_params[AC_mz]));
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->vba.profiles[i], profile_size_bytes));
        ERRCHK_CUDA_ALWAYS(cudaMemset((void*)device->vba.profiles[i], 0, profile_size_bytes));
    }
    */

    // Reductions
    const int3 max_dims = acConstructInt3Param(AC_mx, AC_my, AC_mz, device_config);
    const size_t scratchpad_size       = acKernelReduceGetMinimumScratchpadSize(max_dims);
    const size_t scratchpad_size_bytes = acKernelReduceGetMinimumScratchpadSizeBytes(max_dims);
    for (size_t i = 0; i < NUM_REDUCE_SCRATCHPADS; ++i) {
        ERRCHK_CUDA_ALWAYS(
            cudaMalloc((void**)&device->reduce_scratchpads[i], scratchpad_size_bytes));
    }
    device->scratchpad_size = scratchpad_size;

    // Device constants
    // acDeviceLoadDefaultUniforms(device); // TODO recheck
    acDeviceLoadMeshInfo(device, device_config);

#if AC_VERBOSE
    printf("Created device %d (%p)\n", device->id, device);
#endif
    *device_handle = device;

    acDeviceSynchronizeStream(device, STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acDeviceDestroy(Device device)
{
    cudaSetDevice(device->id);
#if AC_VERBOSE
    printf("Destroying device %d (%p)\n", device->id, device);
#endif
    acDeviceSynchronizeStream(device, STREAM_ALL);

    // Memory
    acVBADestroy(&device->vba);
    /*
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i) {
        cudaFree(device->vba.profiles[i]);
    }
    */

    for (size_t i = 0; i < NUM_REDUCE_SCRATCHPADS; ++i)
        cudaFree(device->reduce_scratchpads[i]);

    // Concurrency
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(device->streams[i]);
    }

    // Destroy Device
    free(device);
    return AC_SUCCESS;
}

AcResult
acDeviceSwapBuffer(const Device device, const VertexBufferHandle handle)
{
    cudaSetDevice(device->id);

    AcReal* tmp             = device->vba.in[handle];
    device->vba.in[handle]  = device->vba.out[handle];
    device->vba.out[handle] = tmp;

    return AC_SUCCESS;
}

AcResult
acDeviceSwapBuffers(const Device device)
{
    cudaSetDevice(device->id);

    int retval = AC_SUCCESS;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        retval |= acDeviceSwapBuffer(device, (VertexBufferHandle)i);

    return (AcResult)retval;
}

/*
AcResult
acDeviceLoadScalarArray(const Device device, const Stream stream, const ScalarArrayHandle handle,
                        const size_t start, const AcReal* data, const size_t num)
{
    cudaSetDevice(device->id);

    if (handle >= NUM_SCALARARRAY_HANDLES || !NUM_SCALARARRAY_HANDLES)
        return AC_FAILURE;

    ERRCHK((int)(start + num) <= max(device->local_config.int_params[AC_mx],
                                     max(device->local_config.int_params[AC_my],
                                         device->local_config.int_params[AC_mz])));
    ERRCHK_ALWAYS(handle < NUM_SCALARARRAY_HANDLES);
    ERRCHK_CUDA(cudaMemcpyAsync(&device->vba.profiles[handle][start], data, sizeof(data[0]) * num,
                                cudaMemcpyHostToDevice, device->streams[stream]));
    return AC_SUCCESS;
}
*/

AcResult
acDeviceLoadVertexBufferWithOffset(const Device device, const Stream stream, const AcMesh host_mesh,
                                   const VertexBufferHandle vtxbuf_handle, const int3 src,
                                   const int3 dst, const int num_vertices)
{
    cudaSetDevice(device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, host_mesh.info);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, device->local_config);

    const AcReal* src_ptr = &host_mesh.vertex_buffer[vtxbuf_handle][src_idx];
    AcReal* dst_ptr       = &device->vba.in[vtxbuf_handle][dst_idx];
    const size_t bytes    = num_vertices * sizeof(src_ptr[0]);

    ERRCHK_CUDA(                                                                                  //
        cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, device->streams[stream]) //
    );

    return AC_SUCCESS;
}

AcResult
acDeviceLoadMeshWithOffset(const Device device, const Stream stream, const AcMesh host_mesh,
                           const int3 src, const int3 dst, const int num_vertices)
{
    WARNING("This function is deprecated");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceLoadVertexBufferWithOffset(device, stream, host_mesh, (VertexBufferHandle)i, src,
                                           dst, num_vertices);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceLoadVertexBuffer(const Device device, const Stream stream, const AcMesh host_mesh,
                         const VertexBufferHandle vtxbuf_handle)
{
    const int3 src            = (int3){0, 0, 0};
    const int3 dst            = src;
    const size_t num_vertices = acVertexBufferSize(device->local_config);
    acDeviceLoadVertexBufferWithOffset(device, stream, host_mesh, vtxbuf_handle, src, dst,
                                       num_vertices);

    return AC_SUCCESS;
}

AcResult
acDeviceLoadMesh(const Device device, const Stream stream, const AcMesh host_mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceLoadVertexBuffer(device, stream, host_mesh, (VertexBufferHandle)i);
    }

    return AC_SUCCESS;
}

AcResult
acDeviceSetVertexBuffer(const Device device, const Stream stream, const VertexBufferHandle handle,
                        const AcReal value)
{
    cudaSetDevice(device->id);

    const size_t count = acVertexBufferSize(device->local_config);
    AcReal* data       = (AcReal*)calloc(count, sizeof(AcReal));
    ERRCHK_ALWAYS(data);

    for (size_t i = 0; i < count; ++i)
        data[i] = value;

    // Set both in and out for safety (not strictly needed)
    ERRCHK_CUDA_ALWAYS(cudaMemcpyAsync(device->vba.in[handle], data, sizeof(data[0]) * count,
                                       cudaMemcpyHostToDevice, device->streams[stream]));
    ERRCHK_CUDA_ALWAYS(cudaMemcpyAsync(device->vba.out[handle], data, sizeof(data[0]) * count,
                                       cudaMemcpyHostToDevice, device->streams[stream]));

    acDeviceSynchronizeStream(device, stream); // Need to synchronize before free
    free(data);
    return AC_SUCCESS;
}

AcResult
acDeviceStoreVertexBufferWithOffset(const Device device, const Stream stream,
                                    const VertexBufferHandle vtxbuf_handle, const int3 src,
                                    const int3 dst, const int num_vertices, AcMesh* host_mesh)
{
    cudaSetDevice(device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, device->local_config);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, host_mesh->info);

    const AcReal* src_ptr = &device->vba.in[vtxbuf_handle][src_idx];
    AcReal* dst_ptr       = &host_mesh->vertex_buffer[vtxbuf_handle][dst_idx];
    const size_t bytes    = num_vertices * sizeof(src_ptr[0]);

    ERRCHK_CUDA(                                                                                  //
        cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, device->streams[stream]) //
    );

    return AC_SUCCESS;
}

AcResult
acDeviceStoreMeshWithOffset(const Device device, const Stream stream, const int3 src,
                            const int3 dst, const int num_vertices, AcMesh* host_mesh)
{
    WARNING("This function is deprecated");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceStoreVertexBufferWithOffset(device, stream, (VertexBufferHandle)i, src, dst,
                                            num_vertices, host_mesh);
    }

    return AC_SUCCESS;
}

AcResult
acDeviceStoreVertexBuffer(const Device device, const Stream stream,
                          const VertexBufferHandle vtxbuf_handle, AcMesh* host_mesh)
{
    int3 src                  = (int3){0, 0, 0};
    int3 dst                  = src;
    const size_t num_vertices = acVertexBufferSize(device->local_config);

    acDeviceStoreVertexBufferWithOffset(device, stream, vtxbuf_handle, src, dst, num_vertices,
                                        host_mesh);

    return AC_SUCCESS;
}

AcResult
acDeviceStoreMesh(const Device device, const Stream stream, AcMesh* host_mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceStoreVertexBuffer(device, stream, (VertexBufferHandle)i, host_mesh);
    }

    return AC_SUCCESS;
}

AcResult
acDeviceTransferVertexBufferWithOffset(const Device src_device, const Stream stream,
                                       const VertexBufferHandle vtxbuf_handle, const int3 src,
                                       const int3 dst, const int num_vertices, Device dst_device)
{
    cudaSetDevice(src_device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, src_device->local_config);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, dst_device->local_config);

    const AcReal* src_ptr = &src_device->vba.in[vtxbuf_handle][src_idx];
    AcReal* dst_ptr       = &dst_device->vba.in[vtxbuf_handle][dst_idx];
    const size_t bytes    = num_vertices * sizeof(src_ptr[0]);

    ERRCHK_CUDA(cudaMemcpyPeerAsync(dst_ptr, dst_device->id, src_ptr, src_device->id, bytes,
                                    src_device->streams[stream]));
    return AC_SUCCESS;
}

AcResult
acDeviceTransferMeshWithOffset(const Device src_device, const Stream stream, const int3 src,
                               const int3 dst, const int num_vertices, Device dst_device)
{
    WARNING("This function is deprecated");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceTransferVertexBufferWithOffset(src_device, stream, (VertexBufferHandle)i, src, dst,
                                               num_vertices, dst_device);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceTransferVertexBuffer(const Device src_device, const Stream stream,
                             const VertexBufferHandle vtxbuf_handle, Device dst_device)
{
    int3 src                  = (int3){0, 0, 0};
    int3 dst                  = src;
    const size_t num_vertices = acVertexBufferSize(src_device->local_config);

    acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                           num_vertices, dst_device);
    return AC_SUCCESS;
}

AcResult
acDeviceTransferMesh(const Device src_device, const Stream stream, Device dst_device)
{
    WARNING("This function is deprecated");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceTransferVertexBuffer(src_device, stream, (VertexBufferHandle)i, dst_device);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceLaunchKernel(const Device device, const Stream stream, const Kernel kernel,
                     const int3 start, const int3 end)
{
    cudaSetDevice(device->id);
    return acLaunchKernel(kernel, device->streams[stream], start, end, device->vba);
}

AcResult
acDeviceLoadStencil(const Device device, const Stream stream, const Stencil stencil,
                    const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    cudaSetDevice(device->id);
    return acLoadStencil(stencil, device->streams[stream], data);
}

/** */
AcResult
acDeviceStoreStencil(const Device device, const Stream stream, const Stencil stencil,
                     AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    cudaSetDevice(device->id);
    return acStoreStencil(stencil, device->streams[stream], data);
}

AcResult
acDeviceIntegrateSubstep(const Device device, const Stream stream, const int step_number,
                         const int3 start, const int3 end, const AcReal dt)
{
#ifdef AC_INTEGRATION_ENABLED
    cudaSetDevice(device->id);

    acDeviceLoadScalarUniform(device, stream, AC_dt, dt);
    acDeviceLoadIntUniform(device, stream, AC_step_number, step_number);
#ifdef AC_SINGLEPASS_INTEGRATION
    return acLaunchKernel(singlepass_solve, device->streams[stream], start, end, device->vba);
#else
    const AcResult res = acLaunchKernel(twopass_solve_intermediate, device->streams[stream], start,
                                        end, device->vba);
    if (res != AC_SUCCESS)
        return res;

    acDeviceSwapBuffers(device);
    return acLaunchKernel(twopass_solve_final, device->streams[stream], start, end, device->vba);
#endif
#else
    (void)device;      // Unused
    (void)stream;      // Unused
    (void)step_number; // Unused
    (void)start;       // Unused
    (void)end;         // Unused
    (void)dt;          // Unused
    ERROR("acDeviceIntegrateSubstep() called but AC_dt not defined!");
    return AC_FAILURE;
#endif
}

AcResult
acDevicePeriodicBoundcondStep(const Device device, const Stream stream,
                              const VertexBufferHandle vtxbuf_handle, const int3 start,
                              const int3 end)
{
    cudaSetDevice(device->id);
    return acKernelPeriodicBoundconds(device->streams[stream], start, end,
                                      device->vba.in[vtxbuf_handle]);
}

AcResult
acDevicePeriodicBoundconds(const Device device, const Stream stream, const int3 start,
                           const int3 end)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDevicePeriodicBoundcondStep(device, stream, (VertexBufferHandle)i, start, end);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceGeneralBoundcondStep(const Device device, const Stream stream,
                             const VertexBufferHandle vtxbuf_handle, const int3 start,
                             const int3 end, const AcMeshInfo config, const int3 bindex)
{
    cudaSetDevice(device->id);
    return acKernelGeneralBoundconds(device->streams[stream], start, end,
                                     device->vba.in[vtxbuf_handle], vtxbuf_handle, config, bindex);
}

AcResult
acDeviceGeneralBoundconds(const Device device, const Stream stream, const int3 start,
                          const int3 end, const AcMeshInfo config, const int3 bindex)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceGeneralBoundcondStep(device, stream, (VertexBufferHandle)i, start, end, config,
                                     bindex);
    }
    return AC_SUCCESS;
}

static int3
constructInt3Param(const Device device, const AcIntParam a, const AcIntParam b, const AcIntParam c)
{
    return (int3){
        device->local_config.int_params[a],
        device->local_config.int_params[b],
        device->local_config.int_params[c],
    };
}

AcResult
acDeviceReduceScalNotAveraged(const Device device, const Stream stream, const ReductionType rtype,
                              const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = constructInt3Param(device, AC_nx_min, AC_ny_min, AC_nz_min);
    const int3 end   = constructInt3Param(device, AC_nx_max, AC_ny_max, AC_nz_max);

    *result = acKernelReduceScal(device->streams[stream], rtype, device->vba.in[vtxbuf_handle],
                                 start, end, device->reduce_scratchpads, device->scratchpad_size);
    return AC_SUCCESS;
}

AcResult
acDeviceReduceScal(const Device device, const Stream stream, const ReductionType rtype,
                   const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    acDeviceReduceScalNotAveraged(device, stream, rtype, vtxbuf_handle, result);

    switch (rtype) {
    case RTYPE_RMS:     /* Fallthrough */
    case RTYPE_RMS_EXP: /* Fallthrough */
    case RTYPE_ALFVEN_RMS: {
        const int3 nn      = constructInt3Param(device, AC_nx, AC_ny, AC_nz);
        const AcReal inv_n = AcReal(1.) / (nn.x * nn.y * nn.z);
        *result            = sqrt(inv_n * *result);
        break;
    }
    default: /* Do nothing */
        break;
    };

    return AC_SUCCESS;
}

AcResult
acDeviceReduceVecNotAveraged(const Device device, const Stream stream, const ReductionType rtype,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = constructInt3Param(device, AC_nx_min, AC_ny_min, AC_nz_min);
    const int3 end   = constructInt3Param(device, AC_nx_max, AC_ny_max, AC_nz_max);

    *result = acKernelReduceVec(device->streams[stream], rtype, start, end, device->vba.in[vtxbuf0],
                                device->vba.in[vtxbuf1], device->vba.in[vtxbuf2],
                                device->reduce_scratchpads, device->scratchpad_size);
    return AC_SUCCESS;
}

AcResult
acDeviceReduceVec(const Device device, const Stream stream, const ReductionType rtype,
                  const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                  const VertexBufferHandle vtxbuf2, AcReal* result)
{
    acDeviceReduceVecNotAveraged(device, stream, rtype, vtxbuf0, vtxbuf1, vtxbuf2, result);

    switch (rtype) {
    case RTYPE_RMS:     /* Fallthrough */
    case RTYPE_RMS_EXP: /* Fallthrough */
    case RTYPE_ALFVEN_RMS: {
        const int3 nn      = constructInt3Param(device, AC_nx, AC_ny, AC_nz);
        const AcReal inv_n = AcReal(1.) / (nn.x * nn.y * nn.z);
        *result            = sqrt(inv_n * *result);
        break;
    }
    default: /* Do nothing */
        break;
    };

    return AC_SUCCESS;
}

AcResult
acDeviceReduceVecScalNotAveraged(const Device device, const Stream stream,
                                 const ReductionType rtype, const VertexBufferHandle vtxbuf0,
                                 const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2,
                                 const VertexBufferHandle vtxbuf3, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = constructInt3Param(device, AC_nx_min, AC_ny_min, AC_nz_min);
    const int3 end   = constructInt3Param(device, AC_nx_max, AC_ny_max, AC_nz_max);

    *result = acKernelReduceVecScal(device->streams[stream], rtype, start, end,
                                    device->vba.in[vtxbuf0], device->vba.in[vtxbuf1],
                                    device->vba.in[vtxbuf2], device->vba.in[vtxbuf3],
                                    device->reduce_scratchpads, device->scratchpad_size);
    return AC_SUCCESS;
}

AcResult
acDeviceReduceVecScal(const Device device, const Stream stream, const ReductionType rtype,
                      const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                      const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                      AcReal* result)
{
    acDeviceReduceVecScalNotAveraged(device, stream, rtype, vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3,
                                     result);

    switch (rtype) {
    case RTYPE_RMS:     /* Fallthrough */
    case RTYPE_RMS_EXP: /* Fallthrough */
    case RTYPE_ALFVEN_RMS: {
        const int3 nn      = constructInt3Param(device, AC_nx, AC_ny, AC_nz);
        const AcReal inv_n = AcReal(1.) / (nn.x * nn.y * nn.z);
        *result            = sqrt(inv_n * *result);
        break;
    }
    default: /* Do nothing */
        break;
    };

    return AC_SUCCESS;
}

AcResult
acDeviceVolumeCopy(const Device device, const Stream stream,                     //
                   const AcReal* in, const int3 in_offset, const int3 in_volume, //
                   AcReal* out, const int3 out_offset, const int3 out_volume)
{
    cudaSetDevice(device->id);
    return acKernelVolumeCopy(device->streams[stream], in, in_offset, in_volume, out, out_offset,
                              out_volume);
}
