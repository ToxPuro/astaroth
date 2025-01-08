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

#include "errchk.h"
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
    printf("    Max registers per block: %d\n", props.regsPerBlock);
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
acDeviceGetLocalConfig(const Device device, AcMeshInfo* info)
{
    *info = device->local_config;
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

    // Check that AC_global_grid_n and AC_multigpu_offset are valid
    // Replace if not and give a warning otherwise
    if (device->local_config.int3_params[AC_global_grid_n].x <= 0 ||
        device->local_config.int3_params[AC_global_grid_n].y <= 0 ||
        device->local_config.int3_params[AC_global_grid_n].z <= 0 ||
        device->local_config.int3_params[AC_multigpu_offset].x < 0 ||
        device->local_config.int3_params[AC_multigpu_offset].y < 0 ||
        device->local_config.int3_params[AC_multigpu_offset].z < 0) {
        WARNING("Invalid AC_global_grid_n or AC_multigpu_offset passed in device_config to "
                "acDeviceCreate. Replacing with AC_global_grid_n = local grid size and "
                "AC_multigpu_offset = (int3){0,0,0}.");
        device->local_config
            .int3_params[AC_global_grid_n] = (int3){device_config.int_params[AC_nx],
                                                    device_config.int_params[AC_ny],
                                                    device_config.int_params[AC_nz]};
        device->local_config.int3_params[AC_multigpu_offset] = (int3){0, 0, 0};
    }

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
    const int3 mm = acConstructInt3Param(AC_mx, AC_my, AC_mz, device->local_config);
    device->vba   = acVBACreate(mm.x, mm.y, mm.z);
    /*
    // VBA Profiles
    const size_t profile_size_bytes = sizeof(AcReal) * max(device->local_config.int_params[AC_mx],
                                                           max(device->local_config.int_params[AC_my],
                                                               device->local_config.int_params[AC_mz]));
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->vba.profiles[i], profile_size_bytes));
        ERRCHK_CUDA_ALWAYS(cudaMemset((void*)device->vba.profiles[i], 0, profile_size_bytes));
    }
    */

    // Reductions
    const int3 max_dims          = acConstructInt3Param(AC_mx, AC_my, AC_mz, device->local_config);
    const size_t scratchpad_size = acKernelReduceGetMinimumScratchpadSize(max_dims);
    const size_t scratchpad_size_bytes = acKernelReduceGetMinimumScratchpadSizeBytes(max_dims);
    for (size_t i = 0; i < NUM_REDUCE_SCRATCHPADS; ++i) {
        ERRCHK_CUDA_ALWAYS(
            cudaMalloc((void**)&device->reduce_scratchpads[i], scratchpad_size_bytes));
    }
    device->scratchpad_size = scratchpad_size;

    // Device constants
    // acDeviceLoadDefaultUniforms(device); // TODO recheck
    acDeviceLoadMeshInfo(device, device->local_config);

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
acDeviceFlushOutputBuffers(const Device device, const Stream stream)
{
    cudaSetDevice(device->id);
    const size_t count = acVertexBufferSize(device->local_config);

    int retval = 0;
    for (size_t i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        retval |= acKernelFlush(device->streams[stream], device->vba.out[i], count, (AcReal)0.0);

    return (AcResult)retval;
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
acDeviceBenchmarkKernel(const Device device, const Kernel kernel, const int3 start, const int3 end)
{
    cudaSetDevice(device->id);
    return acBenchmarkKernel(kernel, start, end, device->vba);
}

AcResult
acDeviceLoadStencil(const Device device, const Stream stream, const Stencil stencil,
                    const AcReal data[STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    cudaSetDevice(device->id);
    return acLoadStencil(stencil, device->streams[stream], data);
}

AcResult
acDeviceLoadStencils(const Device device, const Stream stream,
                     const AcReal data[NUM_STENCILS][STENCIL_DEPTH][STENCIL_HEIGHT][STENCIL_WIDTH])
{
    int retval = 0;
    for (size_t i = 0; i < NUM_STENCILS; ++i)
        retval |= acDeviceLoadStencil(device, stream, (Stencil)i, data[i]);
    return (AcResult)retval;
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
    // Two-pass integration with acDeviceIntegrateSubstep works currently
    // only when integrating the whole subdomain
    // Consider the case:
    // 1) A half of the domain has been updated after the initial call, and the result of step s+1
    // resides in the output buffer.
    //
    // 2) Integration is called again, this time the intermediate w values are incorrectly used for
    // calculating the stencil operations, or, if the buffers have been swapped again, then values
    // from both steps s+0 and s+1 are used to compute the stencils, which is incorrect
    AcMeshDims dims = acGetMeshDims(device->local_config);
    // ERRCHK_ALWAYS(start == dims.n0); // Overload not working for some reason on some compilers
    // ERRCHK_ALWAYS(end == dims.n1); // TODO fix someday
    ERRCHK_ALWAYS(start.x == dims.n0.x); // tmp workaround
    ERRCHK_ALWAYS(start.y == dims.n0.y);
    ERRCHK_ALWAYS(start.z == dims.n0.z);
    ERRCHK_ALWAYS(end.x == dims.n1.x);
    ERRCHK_ALWAYS(end.y == dims.n1.y);
    ERRCHK_ALWAYS(end.z == dims.n1.z);

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
    case RTYPE_RMS:                      /* Fallthrough */
    case RTYPE_RMS_EXP:                  /* Fallthrough */
    case RTYPE_ALFVEN_RADIAL_WINDOW_RMS: /* Fallthrough */
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
    case RTYPE_RMS:                      /* Fallthrough */
    case RTYPE_RMS_EXP:                  /* Fallthrough */
    case RTYPE_ALFVEN_RADIAL_WINDOW_RMS: /* Fallthrough */
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
    case RTYPE_RMS:                      /* Fallthrough */
    case RTYPE_RMS_EXP:                  /* Fallthrough */
    case RTYPE_ALFVEN_RADIAL_WINDOW_RMS: /* Fallthrough */
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

/** XY averages */
AcResult
acDeviceReduceXYAverage(const Device device, const Stream stream, const Field field,
                        const Profile profile)
{
    cudaSetDevice(device->id);
    acDeviceSynchronizeStream(device, stream);

    const AcMeshDims dims = acGetMeshDims(device->local_config);

    for (size_t k = 0; k < dims.m1.z; ++k) {
        const int3 start    = (int3){dims.n0.x, dims.n0.y, k};
        const int3 end      = (int3){dims.n1.x, dims.n1.y, k + 1};
        const size_t nxy    = (end.x - start.x) * (end.y - start.y);
        const AcReal result = (1. / nxy) * acKernelReduceScal(device->streams[stream], RTYPE_SUM,
                                                              device->vba.in[field], start, end,
                                                              device->reduce_scratchpads,
                                                              device->scratchpad_size);
        // printf("%zu Profile: %g\n", k, result);
        // Could be optimized by performing the reduction completely in
        // device memory without the redundant device-host-device transfer
        cudaMemcpy(&device->vba.profiles.in[profile][k], &result, sizeof(result),
                   cudaMemcpyHostToDevice);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceSwapProfileBuffer(const Device device, const Profile handle)
{
    cudaSetDevice(device->id);

    AcReal* tmp                      = device->vba.profiles.in[handle];
    device->vba.profiles.in[handle]  = device->vba.profiles.out[handle];
    device->vba.profiles.out[handle] = tmp;

    return AC_SUCCESS;
}

AcResult
acDeviceSwapProfileBuffers(const Device device, const Profile* profiles, const size_t num_profiles)
{
    int retval = AC_SUCCESS;
    for (size_t i = 0; i < num_profiles; ++i)
        retval |= acDeviceSwapProfileBuffer(device, profiles[i]);

    return (AcResult)retval;
}

AcResult
acDeviceSwapAllProfileBuffers(const Device device)
{
    int retval = AC_SUCCESS;
    for (size_t i = 0; i < NUM_PROFILES; ++i)
        retval |= acDeviceSwapProfileBuffer(device, (Profile)i);

    return (AcResult)retval;
}

AcResult
acDeviceLoadProfile(const Device device, const AcReal* hostprofile, const size_t hostprofile_count,
                    const Profile profile)
{
    cudaSetDevice(device->id);
    ERRCHK_ALWAYS(hostprofile_count == device->vba.profiles.count);
    ERRCHK_CUDA(cudaMemcpy(device->vba.profiles.in[profile], hostprofile,
                           sizeof(device->vba.profiles.in[profile][0]) * device->vba.profiles.count,
                           cudaMemcpyHostToDevice));
    return AC_SUCCESS;
}

AcResult
acDeviceStoreProfile(const Device device, const Profile profile, AcReal* hostprofile,
                     const size_t hostprofile_count)
{
    cudaSetDevice(device->id);
    ERRCHK_ALWAYS(hostprofile_count == device->vba.profiles.count);
    ERRCHK_CUDA(cudaMemcpy(hostprofile, device->vba.profiles.in[profile],
                           sizeof(device->vba.profiles.in[profile][0]) * device->vba.profiles.count,
                           cudaMemcpyDeviceToHost));
    return AC_SUCCESS;
}

AcResult
acDevicePrintProfiles(const Device device)
{
    // int3 multigpu_offset;
    // acStoreInt3Uniform(device->streams[STREAM_DEFAULT], AC_multigpu_offset, &multigpu_offset);
    // printf("%d, %d, %d\n", multigpu_offset.x, multigpu_offset.y, multigpu_offset.z);
    // printf("Num profiles: %zu\n", NUM_PROFILES);
    for (size_t i = 0; i < NUM_PROFILES; ++i) {
        const size_t count = device->vba.profiles.count;
        AcReal host_profile[count];
        cudaMemcpy(host_profile, device->vba.profiles.in[i], sizeof(AcReal) * count,
                   cudaMemcpyDeviceToHost);
        printf("Profile %s (%zu)-----------------\n", profile_names[i], i);
        for (size_t j = 0; j < count; ++j) {
            printf("%g (%zu), ", host_profile[j], j);
        }
        printf("\n");
    }
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

AcResult
acDeviceResetMesh(const Device device, const Stream stream)
{
    cudaSetDevice(device->id);
    acDeviceSynchronizeStream(device, stream);
    return acVBAReset(device->streams[stream], &device->vba);
}

//--------------------------------------

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

#if 0
void
acDeviceTest(const Device device)
{
    AcMeshDims dims = acGetMeshDims(device->local_config);

    ///-------- TESTING START
    AcMeshInfo info = device->local_config;
    AcMesh model;
    acHostMeshCreate(info, &model); // remember to remove or free
#if 0
    for (size_t field = 0; field < NUM_FIELDS; ++field) {
        for (size_t k = dims.m0.z; k < as_size_t(dims.m1.z); ++k) {
            for (size_t j = dims.n0.y; j < as_size_t(dims.n1.y); ++j) {
                for (size_t i = dims.n0.x; i < as_size_t(dims.n1.x); ++i) {
                    const size_t si = (i - dims.n0.x) + (j - dims.n0.y) * dims.n1.x;
                    const int salt  = 2 * (si % 2) - 1; // Generates -1,1,-1,1,...
                    // Nice mathematical feature: nxy is always even for nx, ny > 1
                    model.vertex_buffer[field][i + j * dims.m1.x +
                                               k * dims.m1.x * dims.m1.y] = (int)k + salt;
                }
            }
            // If one of the dimensions is 1 and the other one is odd
            if ((dims.nn.x * dims.nn.y) % 2) //
                ++model.vertex_buffer[field][dims.n0.x + dims.n0.y * dims.m1.x +
                                             k * dims.m1.x * dims.m1.y];
        }
    }
#elif 0 // unique spatial
    for (size_t field = 0; field < NUM_FIELDS; ++field) {
        for (size_t i = 0; i < dims.m1.x * dims.m1.y * dims.m1.z; ++i)
            model.vertex_buffer[field][i] = i;
    }
#elif 0 // unique all
    for (size_t field = 0; field < NUM_FIELDS; ++field) {
        for (size_t i = 0; i < dims.m1.x * dims.m1.y * dims.m1.z; ++i)
            model.vertex_buffer[field][i] = i + field * dims.m1.x * dims.m1.y * dims.m1.z;
    }
#else
    for (size_t i = 0; i < dims.m1.x * dims.m1.y * dims.m1.z; ++i) {
        model.vertex_buffer[VTXBUF_UUX][i] = 0.5;
        model.vertex_buffer[VTXBUF_UUY][i] = 0.2;
        model.vertex_buffer[VTXBUF_UUZ][i] = 0.8;
        model.vertex_buffer[TF_a11_x][i]   = 0.2;
        model.vertex_buffer[TF_a11_y][i]   = 0.3;
        model.vertex_buffer[TF_a11_z][i]   = -0.6;
    }
#endif
    acDeviceLoadMesh(device, STREAM_DEFAULT, model);
    // acDevicePeriodicBoundconds(device, STREAM_DEFAULT, dims.m0, dims.m1); // note: messes up
    // small grids
    cudaDeviceSynchronize();

    printf("---Model---\n");
    const size_t field = 0;
    for (size_t i = 0; i < dims.m1.x * dims.m1.y * dims.m1.z; ++i) {
        printf("%-4g ", i, model.vertex_buffer[field][i]);

        if (!((i + 1) % dims.m1.x))
            printf("\n");
        if (!((i + 1) % dims.m1.x) && !(((i + 1) / dims.m1.x) % dims.m1.y))
            printf("\n---\n");
    }
    printf("\n");
    ///-------- TESTING END

    const size_t num_blocks  = 3 + 3 * 4;
    const AcShape out_volume = {
        .x = dims.nn.x,
        .y = dims.nn.y,
        .z = dims.m1.z,
        .w = num_blocks,
    };
    const size_t count = acShapeSize(out_volume);
    AcBuffer buffer    = acBufferCreate(count, true);

    const AcIndex in_offset = {
        .x = dims.n0.x,
        .y = dims.n0.y,
        .z = 0,
        .w = 0,
    };
    const AcShape in_volume = {
        .x = dims.m1.x,
        .y = dims.m1.y,
        .z = dims.m1.z,
        .w = 1,
    };
    const AcShape block_volume = {
        .x = out_volume.x,
        .y = out_volume.y,
        .z = out_volume.z,
        .w = 1,
    };

    const Field basic_fields[] = {VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ};
    for (size_t w = 0; w < ARRAY_SIZE(basic_fields); ++w) {
        const AcIndex out_offset = {
            .x = 0,
            .y = 0,
            .z = 0,
            .w = w,
        };

        acReindex(device->streams[STREAM_DEFAULT], device->vba.in[basic_fields[w]], in_offset,
                  in_volume, buffer.data, out_offset, out_volume, block_volume);
    }
    const AcIndex out_offset = {
        .x = 0,
        .y = 0,
        .z = 0,
        .w = 0,
    };
    acMapCross(device->streams[STREAM_DEFAULT], device->vba, in_offset, in_volume, buffer.data,
               out_offset, out_volume, block_volume);

    ///-------- TESTING START
    // const AcShape volume = {
    //     .x = dims.nn.x,
    //     .y = dims.nn.y,
    //     .z = dims.m1.z,
    //     .w = 3,
    // };
    const AcShape volume = out_volume;
    cudaDeviceSynchronize();
    printf("---Reindexed basic---\n");
    AcBuffer host = acBufferCreate(count, false);
    acBufferMigrate(buffer, &host);
    for (size_t i = 0; i < acShapeSize(volume); ++i) {
        if (!(i % volume.x)) {
            printf("\n");
            if (!((i / volume.x) % volume.y)) {
                printf("\n---\n");
                if (!(((i + 1) / (volume.x * volume.y)) % volume.z))
                    printf("\n--next buffer %zu--\n", (i + 1) / (volume.x * volume.y * volume.z));
            }
        }

        printf("%-4g ", i, host.data[i]);

        // if (!((i + 1) % volume.x)) {
        //     printf("\n");
        //     if (!(((i + 1) / volume.x) % volume.y)) {
        //         printf("\n---\n");
        //         if (!(((i + 1) / (volume.x * volume.y)) % volume.z))
        //             printf("\n--next buffer %zu--\n", (i + 1) / (volume.x * volume.y *
        //             volume.z));
        //     }
        // }
    }
    printf("\n");

    for (size_t i = 0; i < host.count; ++i)
        printf("%g ", host.data[i]);
    acBufferDestroy(&host);
    ///-------- TESTING END

    const size_t num_segments = num_blocks * out_volume.z;
    // acSegmentedReduce(device->streams[STREAM_DEFAULT], buffer.data, count, num_segments,
    //                   device->vba.profiles.in[0]);
    // cudaDeviceSynchronize();

    // Test
    const size_t segment_size = count / num_segments;
    AcBuffer d_profiles       = acBufferCreate(num_segments, true);
    acSegmentedReduce(device->streams[STREAM_DEFAULT], buffer.data, count, num_segments,
                      d_profiles.data);
    AcBuffer h_profiles = acBufferCreate(num_segments, false);
    acBufferMigrate(d_profiles, &h_profiles);
    for (size_t i = 0; i < num_segments; ++i)
        printf("Segment %zu: %g, average %g\n", i, h_profiles.data[i],
               h_profiles.data[i] / segment_size);
    // cudaDeviceSynchronize();
    // AcBuffer hostbuffer = acBufferCreate(num_segments, false);
    // cudaMemcpy(hostbuffer.data, device->vba.profiles.in[0],
    // sizeof(hostbuffer.data[0])*num_segments, cudaMemcpyDeviceToDevice);
    // // acBufferMigrate(buffer, &hostbuffer);
    // for (size_t w = 0; w < num_segments; ++w) {
    //     printf("start %zu: %g\n", w,
    //            hostbuffer.data[w * out_volume.x * out_volume.y * out_volume.z]);
    //     printf("end %zu: %g\n", w,
    //            hostbuffer.data[(w + 1) * out_volume.x * out_volume.y * out_volume.z - 1]);

    //     AcBuffer profiles = acBufferCreate(num_segments, false);
    //     cudaMemcpy(profiles.data, device->vba.profiles.in, sizeof(profiles.data[0]) *
    //     num_segments,
    //                cudaMemcpyDeviceToHost);
    //     printf("Profile %zu: %g\n", w, profiles.data[w]);
    //     acBufferDestroy(&profiles);
    // }
    // acBufferDestroy(&hostbuffer);

    acBufferDestroy(&buffer);
}
#endif

AcResult
acDeviceReduceXYAverages(const Device device, const Stream stream)
{
#ifdef AC_TFM_ENABLED
    AcMeshDims dims = acGetMeshDims(device->local_config);

    // Intermediate buffer
    const size_t num_compute_profiles = 5 * 3;
    const AcShape buffer_shape        = {
               .x = as_size_t(dims.nn.x),
               .y = as_size_t(dims.nn.y),
               .z = as_size_t(dims.m1.z),
               .w = num_compute_profiles,
    };
    const size_t buffer_size = acShapeSize(buffer_shape);
    AcBuffer buffer          = acBufferCreate(buffer_size, true);

    // Indices and shapes
    const AcIndex in_offset = {
        .x = as_size_t(dims.n0.x),
        .y = as_size_t(dims.n0.y),
        .z = 0,
        .w = 0,
    };
    const AcShape in_shape = {
        .x = as_size_t(dims.m1.x),
        .y = as_size_t(dims.m1.y),
        .z = as_size_t(dims.m1.z),
        .w = 1,
    };
    const AcShape block_shape = {
        .x = buffer_shape.x,
        .y = buffer_shape.y,
        .z = buffer_shape.z,
        .w = 1,
    };

    // Reindex
    VertexBufferHandle reindex_fields[] = {
        VTXBUF_UUX, VTXBUF_UUY,
        VTXBUF_UUZ, //
        TF_uxb11_x, TF_uxb11_y,
        TF_uxb11_z, //
        TF_uxb12_x, TF_uxb12_y,
        TF_uxb12_z, //
        TF_uxb21_x, TF_uxb21_y,
        TF_uxb21_z, //
        TF_uxb22_x, TF_uxb22_y,
        TF_uxb22_z, //
    };
    for (size_t w = 0; w < ARRAY_SIZE(reindex_fields); ++w) {
        const AcIndex buffer_offset = {
            .x = 0,
            .y = 0,
            .z = 0,
            .w = w,
        };
        acReindex(device->streams[STREAM_DEFAULT], //
                  device->vba.in[reindex_fields[w]], in_offset,
                  in_shape, //
                  buffer.data, buffer_offset, buffer_shape, block_shape);
    }
    // // Note no offset here: is applied in acMapCross instead due to how it works with SOA
    // vectors. const AcIndex buffer_offset = {
    //     .x = 0,
    //     .y = 0,
    //     .z = 0,
    //     .w = 0,
    // };
    // acReindexCross(device->streams[STREAM_DEFAULT],  //
    //                device->vba, in_offset, in_shape, //
    //                buffer.data, buffer_offset, buffer_shape, block_shape);

    // Reduce
    // Note the ordering of the fields. The ordering of the fields
    // in the input buffer must be the same as desired for the ordering of
    // profiles in the output array.
    const size_t num_segments = buffer_shape.z * buffer_shape.w;
    acSegmentedReduce(device->streams[STREAM_DEFAULT], //
                      buffer.data, buffer_size, num_segments, device->vba.profiles.in[0]);

    // NOTE: Revisit this
    const size_t gnx = as_size_t(device->local_config.int3_params[AC_global_grid_n].x);
    const size_t gny = as_size_t(device->local_config.int3_params[AC_global_grid_n].y);
    cudaSetDevice(device->id);
    acMultiplyInplace(1. / (gnx * gny), num_compute_profiles * device->vba.profiles.count,
                      device->vba.profiles.in[0]);

    acBufferDestroy(&buffer);
    return AC_SUCCESS;
#else
    ERROR("acDeviceReduceXYAverages called but AC_TFM_ENABLED was false");
    return AC_FAILURE;
#endif
}

/** Note: very inefficient. Should only be used for testing. */
AcResult
acDeviceWriteMeshToDisk(const Device device, const VertexBufferHandle vtxbuf, const char* filepath)
{
    AcMesh host_mesh;
    acHostMeshCreate(device->local_config, &host_mesh);

    acDeviceStoreMesh(device, STREAM_DEFAULT, &host_mesh);
    acDeviceSynchronizeStream(device, STREAM_DEFAULT);

    FILE* fp = fopen(filepath, "w");
    ERRCHK_ALWAYS(fp);

    const size_t count         = acVertexBufferSize(device->local_config);
    const size_t count_written = fwrite(host_mesh.vertex_buffer[vtxbuf], sizeof(AcReal), count, fp);
    ERRCHK_ALWAYS(count_written == count);

    fclose(fp);

    acHostMeshDestroy(&host_mesh);
    return AC_SUCCESS;
}

AcResult
acDeviceGetVBA(const Device device, VertexBufferArray* vba)
{
    *vba = device->vba;
    return AC_SUCCESS;
}
