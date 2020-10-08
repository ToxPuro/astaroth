#include "astaroth.h"

#include <string.h>
#include <vector>
//TODO: remove iostream once debugging messages are cleaned up
#include <iostream>
#include <utility>

#include "astaroth_utils.h"
#include "errchk.h"
#include "math_utils.h"
#include "timer_hires.h"

#include "kernels/kernels.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

#define MPI_GPUDIRECT_DISABLED (0) // Buffer through host memory, deprecated
#define MPI_DECOMPOSITION_AXES (3)
#define MPI_COMPUTE_ENABLED (1)
#define MPI_COMM_ENABLED (1)
#define MPI_INCL_CORNERS (0)
#define MPI_USE_PINNED (0)              // Do inter-node comm with pinned memory
#define MPI_USE_CUDA_DRIVER_PINNING (0) // Pin with cuPointerSetAttribute, otherwise cudaMallocHost
#define PACK_PIPELINE (0)

#if AC_DOUBLE_PRECISION == (1)
#define AC_MPI_TYPE MPI_DOUBLE
#else
#define AC_MPI_TYPE MPI_FLOAT
#endif

#define NUM_SEGMENTS 26

//Computational task dependencies
#define NUM_COMPUTE_FACE_DEPENDENCIES 1
#define NUM_COMPUTE_EDGE_DEPENDENCIES 3

#if MPI_INCL_CORNERS
#define NUM_COMPUTE_CORNER_DEPENDENCIES 7
#else
#define NUM_COMPUTE_CORNER_DEPENDENCIES 6
#endif

//Send task dependencies
#define NUM_SEND_FACE_DEPENDENCIES 9
#define NUM_SEND_EDGE_DEPENDENCIES 3

#define NUM_SEND_CORNER_DEPENDENCIES 1




const int3 nghost = (int3){NGHOST,NGHOST,NGHOST};


#include <cuda.h> // CUDA driver API (needed if MPI_USE_CUDA_DRIVER_PINNING is set)

AcResult
acDevicePrintInfo(const Device device)
{
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
    printf("    SP to DP flops performance ratio: %d:1\n", props.singleToDoublePrecisionPerfRatio);
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
    printf("    Local L1 cache supported: %d\n", props.localL1CacheSupported);
    printf("    Global L1 cache supported: %d\n", props.globalL1CacheSupported);
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
    printf("    Stream priorities supported: %d\n", props.streamPrioritiesSupported);
    printf("--------------------------------------------------\n");

    return AC_SUCCESS;
}

AcResult
acDeviceAutoOptimize(const Device device)
{
    cudaSetDevice(device->id);
    const int3 start = (int3){
        device->local_config.int_params[AC_nx_min],
        device->local_config.int_params[AC_ny_min],
        device->local_config.int_params[AC_nz_min],
    };
    const int3 end = (int3){
        device->local_config.int_params[AC_nx_max],
        device->local_config.int_params[AC_ny_max],
        device->local_config.int_params[AC_nz_max],
    };
    return acKernelAutoOptimizeIntegration(start, end, device->vba);
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
    cudaSetDevice(id);
    cudaDeviceReset(); // Would be good for safety, but messes stuff up if we want to emulate
    // multiple devices with a single GPU

    // Create Device
    struct device_s* device = (struct device_s*)malloc(sizeof(*device));
    ERRCHK_ALWAYS(device);

    device->id           = id;
    device->local_config = device_config;
#if AC_VERBOSE
    acDevicePrintInfo(device);

    // Check that the code was compiled for the proper GPU architecture
    printf("Trying to run a dummy kernel. If this fails, make sure that your\n"
           "device supports the CUDA architecture you are compiling for.\n"
           "Running dummy kernel... ");
#endif
    printf("Testing CUDA... ");
    fflush(stdout);
    acKernelDummy();
    printf("Success!\n");
    printf("\x1B[32m%s\x1B[0m\n", "OK!");
    fflush(stdout);

    // Concurrency
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithPriority(&device->streams[i], cudaStreamNonBlocking, i);
    }

    // Memory
    // VBA in/out
    const size_t vba_size_bytes = acVertexBufferSizeBytes(device_config);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->vba.in[i], vba_size_bytes));
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->vba.out[i], vba_size_bytes));
    }
    // VBA Profiles
    const size_t profile_size_bytes = sizeof(AcReal) * max(device_config.int_params[AC_mx],
                                                           max(device_config.int_params[AC_my],
                                                               device_config.int_params[AC_mz]));
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->vba.profiles[i], profile_size_bytes));
    }

    // Reductions
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->reduce_scratchpad,
                                  acVertexBufferCompdomainSizeBytes(device_config)));
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&device->reduce_result, sizeof(AcReal)));

    // Device constants
    acDeviceLoadDefaultUniforms(device);
    acDeviceLoadMeshInfo(device, device_config);

#if AC_VERBOSE
    printf("Created device %d (%p)\n", device->id, device);
#endif
    *device_handle = device;

    // Autoptimize
    acDeviceAutoOptimize(device);

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
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        cudaFree(device->vba.in[i]);
        cudaFree(device->vba.out[i]);
    }
    for (int i = 0; i < NUM_SCALARARRAY_HANDLES; ++i) {
        cudaFree(device->vba.profiles[i]);
    }

    cudaFree(device->reduce_scratchpad);
    cudaFree(device->reduce_result);

    // Concurrency
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(device->streams[i]);
    }

    // Destroy Device
    free(device);
    return AC_SUCCESS;
}

AcResult
acDeviceSwapBuffers(const Device device)
{
    cudaSetDevice(device->id);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        AcReal* tmp        = device->vba.in[i];
        device->vba.in[i]  = device->vba.out[i];
        device->vba.out[i] = tmp;
    }
    return AC_SUCCESS;
}

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
acDeviceIntegrateSubstep(const Device device, const Stream stream, const int step_number,
                         const int3 start, const int3 end, const AcReal dt)
{
    cudaSetDevice(device->id);
    acDeviceLoadScalarUniform(device, stream, AC_dt, dt);
    return acKernelIntegrateSubstep(device->streams[stream], step_number, start, end, device->vba);
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
acDeviceReduceScal(const Device device, const Stream stream, const ReductionType rtype,
                   const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = (int3){device->local_config.int_params[AC_nx_min],
                              device->local_config.int_params[AC_ny_min],
                              device->local_config.int_params[AC_nz_min]};

    const int3 end = (int3){device->local_config.int_params[AC_nx_max],
                            device->local_config.int_params[AC_ny_max],
                            device->local_config.int_params[AC_nz_max]};

    *result = acKernelReduceScal(device->streams[stream], rtype, start, end,
                                 device->vba.in[vtxbuf_handle], device->reduce_scratchpad,
                                 device->reduce_result);
    return AC_SUCCESS;
}

AcResult
acDeviceReduceVec(const Device device, const Stream stream, const ReductionType rtype,
                  const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                  const VertexBufferHandle vtxbuf2, AcReal* result)
{
    cudaSetDevice(device->id);

    const int3 start = (int3){device->local_config.int_params[AC_nx_min],
                              device->local_config.int_params[AC_ny_min],
                              device->local_config.int_params[AC_nz_min]};

    const int3 end = (int3){device->local_config.int_params[AC_nx_max],
                            device->local_config.int_params[AC_ny_max],
                            device->local_config.int_params[AC_nz_max]};

    *result = acKernelReduceVec(device->streams[stream], rtype, start, end, device->vba.in[vtxbuf0],
                                device->vba.in[vtxbuf1], device->vba.in[vtxbuf2],
                                device->reduce_scratchpad, device->reduce_result);
    return AC_SUCCESS;
}

#if AC_MPI_ENABLED
/**
Quick overview of the MPI implementation:

The halo is partitioned into segments. The first coordinate of a segment is b0.
The array containing multiple b0s is called... "b0s".

Each b0 maps to an index in the computational domain of some neighboring process a0.
We have a0 = mod(b0 - nghost, nn) + nghost.
Intuitively, we
  1) Transform b0 into a coordinate system where (0, 0, 0) is the first index in
     the comp domain.
  2) Wrap the transformed b0 around nn (comp domain)
  3) Transform b0 back to a coordinate system where (0, 0, 0) is the first index
     in the ghost zone

struct PackedData is used for packing and unpacking. Holds the actual data in
                  the halo partition
struct CommData holds multiple PackedDatas for sending and receiving halo
                partitions
struct Grid contains information about the local GPU device, decomposition, the
            total mesh dimensions and CommDatas


Basic steps:
  1) Distribute the mesh among ranks
  2) Integrate & communicate
    - start inner integration and at the same time, pack halo data and send it to neighbors
    - once all halo data has been received, unpack and do outer integration
    - sync and start again
  3) Gather the mesh to rank 0 for postprocessing
*/
#include <mpi.h>

#include <stdint.h>

typedef struct {
    uint64_t x, y, z;
} uint3_64;

static uint3_64
operator+(const uint3_64& a, const uint3_64& b)
{
    return (uint3_64){a.x + b.x, a.y + b.y, a.z + b.z};
}

static int3
make_int3(const uint3_64 a)
{
    return (int3){(int)a.x, (int)a.y, (int)a.z};
}

static uint64_t
mod(const int a, const int b)
{
    const int r = a % b;
    return r < 0 ? r + b : r;
}

static uint3_64
morton3D(const uint64_t pid)
{
    uint64_t i, j, k;
    i = j = k = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 3 * bit;
            k |= ((pid & (mask << 0)) >> 2 * bit) >> 0;
            j |= ((pid & (mask << 1)) >> 2 * bit) >> 1;
            i |= ((pid & (mask << 2)) >> 2 * bit) >> 2;
        }
    }
    // Just a quick copy/paste for other decomp dims
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 2 * bit;
            j |= ((pid & (mask << 0)) >> 1 * bit) >> 0;
            k |= ((pid & (mask << 1)) >> 1 * bit) >> 1;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << 1 * bit;
            k |= ((pid & (mask << 0)) >> 0 * bit) >> 0;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return (uint3_64){i, j, k};
}

static uint64_t
morton1D(const uint3_64 pid)
{
    uint64_t i = 0;

    if (MPI_DECOMPOSITION_AXES == 3) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 2 * bit;
            i |= ((pid.y & mask) << 1) << 2 * bit;
            i |= ((pid.x & mask) << 2) << 2 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 2) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.y & mask) << 0) << 1 * bit;
            i |= ((pid.z & mask) << 1) << 1 * bit;
        }
    }
    else if (MPI_DECOMPOSITION_AXES == 1) {
        for (int bit = 0; bit <= 21; ++bit) {
            const uint64_t mask = 0x1l << bit;
            i |= ((pid.z & mask) << 0) << 0 * bit;
        }
    }
    else {
        fprintf(stderr, "Invalid MPI_DECOMPOSITION_AXES\n");
        ERRCHK_ALWAYS(0);
    }

    return i;
}

static uint3_64
decompose(const uint64_t target)
{
    // This is just so beautifully elegant. Complex and efficient decomposition
    // in just one line of code.
    uint3_64 p = morton3D(target - 1) + (uint3_64){1, 1, 1};

    ERRCHK_ALWAYS(p.x * p.y * p.z == target);
    return p;
}

static uint3_64
wrap(const int3 i, const uint3_64 n)
{
    return (uint3_64){
        mod(i.x, n.x),
        mod(i.y, n.y),
        mod(i.z, n.z),
    };
}

static int
getPid(const int3 pid_raw, const uint3_64 decomp)
{
    const uint3_64 pid = wrap(pid_raw, decomp);
    return (int)morton1D(pid);
}

static int3
getPid3D(const uint64_t pid, const uint3_64 decomp)
{
    const uint3_64 pid3D = morton3D(pid);
    ERRCHK_ALWAYS(getPid(make_int3(pid3D), decomp) == (int)pid);
    return (int3){(int)pid3D.x, (int)pid3D.y, (int)pid3D.z};
}




/** Assumes that contiguous pids are on the same node and there is one process per GPU. */
static inline bool
onTheSameNode(const uint64_t pid_a, const uint64_t pid_b)
{
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    const uint64_t node_a = pid_a / devices_per_node;
    const uint64_t node_b = pid_b / devices_per_node;

    return node_a == node_b;
}

static inline int3
segment_id_to_halo_segment_position(const int3 seg_id, const int3 grid_dimensions)
{
    return (int3){
        seg_id.x < 0 ? 0 : seg_id.x > 0 ? NGHOST + grid_dimensions.x : NGHOST,
        seg_id.y < 0 ? 0 : seg_id.y > 0 ? NGHOST + grid_dimensions.y : NGHOST,
        seg_id.z < 0 ? 0 : seg_id.z > 0 ? NGHOST + grid_dimensions.z : NGHOST,
    };
}


static inline int3
segment_id_to_local_segment_position(const int3 seg_id, const int3 grid_dimensions)
{
    return (int3){
        seg_id.x > 0 ?  grid_dimensions.x : NGHOST,
        seg_id.y > 0 ?  grid_dimensions.y : NGHOST,
        seg_id.z > 0 ?  grid_dimensions.z : NGHOST,
    };
}

static inline int3
segment_id_to_dims(const int3 seg_id, const int3 grid_dimensions)
{
    return (int3){
        seg_id.x == 0? grid_dimensions.x : NGHOST,
        seg_id.y == 0? grid_dimensions.y : NGHOST,
        seg_id.z == 0? grid_dimensions.z : NGHOST,
    };
}

//Maps non-zero three-valued (-1,0,1) halo segment coords to a flat index from 0 to 25.
static inline int
segment_id_to_index(const int3 seg_id){
    return ((3+seg_id.x)%3)*9 + ((3+seg_id.y)%3)*3 + (3+seg_id.z)%3 - 1;
}

static inline int3
index_to_segment_id(int index)
{
    index++;
    int3 segment_id = (int3){index/9, (index%9)/3, index%3};
    segment_id.x = segment_id.x>1?-1:segment_id.x;
    segment_id.y = segment_id.y>1?-1:segment_id.y;
    segment_id.z = segment_id.z>1?-1:segment_id.z;
    return segment_id;
}

static inline int
segment_type(const int3 seg_id){
    int seg_type = (seg_id.x == 0?0:1)+(seg_id.y == 0?0:1)+(seg_id.z == 0?0:1);
    if (seg_type > 3 || seg_type < 0){
        throw std::runtime_error("Invalid segment id");
    }
    return seg_type; 
}


static PackedData
acCreatePackedData(const int3 dims)
{
    PackedData data = {};

    data.dims = dims;

    const size_t bytes = dims.x * dims.y * dims.z * sizeof(data.data[0]) * NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data.data, bytes));

#if MPI_USE_CUDA_DRIVER_PINNING
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data.data_pinned, bytes));

    unsigned int flag = 1;
    CUresult retval   = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                            (CUdeviceptr)data.data_pinned);
    ERRCHK_ALWAYS(retval == CUDA_SUCCESS);
#else
    ERRCHK_CUDA_ALWAYS(cudaMallocHost((void**)&data.data_pinned, bytes));
// ERRCHK_CUDA_ALWAYS(cudaMallocManaged((void**)&data.data_pinned, bytes)); // Significantly
// slower than pinned (38 ms vs. 125 ms)
#endif // USE_CUDA_DRIVER_PINNING

    return data;
}

static AcResult
acDestroyPackedData(PackedData* data)
{
    cudaFree(data->data_pinned);

    data->dims = (int3){-1, -1, -1};
    cudaFree(data->data);
    data->data = NULL;

    return AC_SUCCESS;
}

#if MPI_GPUDIRECT_DISABLED
static PackedData
acCreatePackedDataHost(const int3 dims)
{
    PackedData data = {};

    data.dims = dims;

    const size_t bytes = dims.x * dims.y * dims.z * sizeof(data.data[0]) * NUM_VTXBUF_HANDLES;
    data.data          = (AcReal*)malloc(bytes);
    ERRCHK_ALWAYS(data.data);

    return data;
}

static AcResult
acDestroyPackedDataHost(PackedData* data)
{
    data->dims = (int3){-1, -1, -1};
    free(data->data);
    data->data = NULL;

    return AC_SUCCESS;
}

static void
acTransferPackedDataToHost(const Device device, const cudaStream_t stream, const PackedData ddata,
                           PackedData* hdata)
{
    cudaSetDevice(device->id);

    const size_t bytes = ddata.dims.x * ddata.dims.y * ddata.dims.z * sizeof(ddata.data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(hdata->data, ddata.data, bytes, cudaMemcpyDeviceToHost, stream));
}

static void
acTransferPackedDataToDevice(const Device device, const cudaStream_t stream, const PackedData hdata,
                             PackedData* ddata)
{
    cudaSetDevice(device->id);

    const size_t bytes = hdata.dims.x * hdata.dims.y * hdata.dims.z * sizeof(hdata.data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(ddata->data, hdata.data, bytes, cudaMemcpyHostToDevice, stream));
}
#endif // MPI_GPUDIRECT_DISABLED

static void
acPinPackedData(const Device device, const cudaStream_t stream, PackedData* ddata)
{
    cudaSetDevice(device->id);
    // TODO sync stream
    ddata->pinned = true;

    const size_t bytes = ddata->dims.x * ddata->dims.y * ddata->dims.z * sizeof(ddata->data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(ddata->data_pinned, ddata->data, bytes, cudaMemcpyDefault, stream));
}

static void
acUnpinPackedData(const Device device, const cudaStream_t stream, PackedData* ddata)
{
    if (!ddata->pinned) // Unpin iff the data was pinned previously
        return;

    cudaSetDevice(device->id);
    // TODO sync stream
    ddata->pinned = false;

    const size_t bytes = ddata->dims.x * ddata->dims.y * ddata->dims.z * sizeof(ddata->data[0]) *
                         NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA(cudaMemcpyAsync(ddata->data, ddata->data_pinned, bytes, cudaMemcpyDefault, stream));
}

// TODO: do with packed data
static AcResult
acDeviceDistributeMeshMPI(const AcMesh src, const uint3_64 decomposition, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
#if AC_VERBOSE
    printf("Distributing mesh...\n");
    fflush(stdout);
#endif

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    ERRCHK_ALWAYS(dst);

    // Submesh nn
    const int3 nn = (int3){
        dst->info.int_params[AC_nx],
        dst->info.int_params[AC_ny],
        dst->info.int_params[AC_nz],
    };

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (int k = NGHOST; k < NGHOST + nn.z; ++k) {
                for (int j = NGHOST; j < NGHOST + nn.y; ++j) {
                    const int i       = NGHOST;
                    const int count   = nn.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, src.info);
                    const int dst_idx = acVertexBufferIdx(i, j, k, dst->info);
                    memcpy(&dst->vertex_buffer[vtxbuf][dst_idx], //
                           &src.vertex_buffer[vtxbuf][src_idx],  //
                           count * sizeof(src.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (int k = NGHOST; k < NGHOST + nn.z; ++k) {
            for (int j = NGHOST; j < NGHOST + nn.y; ++j) {
                const int i     = NGHOST;
                const int count = nn.x;

                if (pid != 0) {
                    const int dst_idx = acVertexBufferIdx(i, j, k, dst->info);
                    // Recv
                    MPI_Status status;
                    MPI_Recv(&dst->vertex_buffer[vtxbuf][dst_idx], count, datatype, 0, 0,
                             MPI_COMM_WORLD, &status);
                }
                else {
                    for (int tgt_pid = 1; tgt_pid < nprocs; ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid, decomposition);
                        const int src_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                              j + tgt_pid3d.y * nn.y, //
                                                              k + tgt_pid3d.z * nn.z, //
                                                              src.info);

                        // Send
                        MPI_Send(&src.vertex_buffer[vtxbuf][src_idx], count, datatype, tgt_pid, 0,
                                 MPI_COMM_WORLD);
                    }
                }
            }
        }
    }
    return AC_SUCCESS;
}

// TODO: do with packed data
static AcResult
acDeviceGatherMeshMPI(const AcMesh src, const uint3_64 decomposition, AcMesh* dst)
{
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Gathering mesh...\n");
    fflush(stdout);

    MPI_Datatype datatype = MPI_FLOAT;
    if (sizeof(AcReal) == 8)
        datatype = MPI_DOUBLE;

    int pid, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (pid == 0)
        ERRCHK_ALWAYS(dst);

    // Submesh nn
    const int3 nn = (int3){
        src.info.int_params[AC_nx],
        src.info.int_params[AC_ny],
        src.info.int_params[AC_nz],
    };

    // Submesh mm
    const int3 mm = (int3){
        src.info.int_params[AC_mx],
        src.info.int_params[AC_my],
        src.info.int_params[AC_mz],
    };

    // Send to self
    if (pid == 0) {
        for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
            // For pencils
            for (int k = 0; k < mm.z; ++k) {
                for (int j = 0; j < mm.y; ++j) {
                    const int i       = 0;
                    const int count   = mm.x;
                    const int src_idx = acVertexBufferIdx(i, j, k, src.info);
                    const int dst_idx = acVertexBufferIdx(i, j, k, dst->info);
                    memcpy(&dst->vertex_buffer[vtxbuf][dst_idx], //
                           &src.vertex_buffer[vtxbuf][src_idx],  //
                           count * sizeof(src.vertex_buffer[i][0]));
                }
            }
        }
    }

    for (int vtxbuf = 0; vtxbuf < NUM_VTXBUF_HANDLES; ++vtxbuf) {
        // For pencils
        for (int k = 0; k < mm.z; ++k) {
            for (int j = 0; j < mm.y; ++j) {
                const int i     = 0;
                const int count = mm.x;

                if (pid != 0) {
                    // Send
                    const int src_idx = acVertexBufferIdx(i, j, k, src.info);
                    MPI_Send(&src.vertex_buffer[vtxbuf][src_idx], count, datatype, 0, 0,
                             MPI_COMM_WORLD);
                }
                else {
                    for (int tgt_pid = 1; tgt_pid < nprocs; ++tgt_pid) {
                        const int3 tgt_pid3d = getPid3D(tgt_pid, decomposition);
                        const int dst_idx    = acVertexBufferIdx(i + tgt_pid3d.x * nn.x, //
                                                              j + tgt_pid3d.y * nn.y, //
                                                              k + tgt_pid3d.z * nn.z, //
                                                              dst->info);

                        // Recv
                        MPI_Status status;
                        MPI_Recv(&dst->vertex_buffer[vtxbuf][dst_idx], count, datatype, tgt_pid, 0,
                                 MPI_COMM_WORLD, &status);
                    }
                }
            }
        }
    }
    return AC_SUCCESS;
}

static int3
mod(const int3 a, const int3 n)
{
    return (int3){(int)mod(a.x, n.x), (int)mod(a.y, n.y), (int)mod(a.z, n.z)};
}

typedef struct ComputationTask {
    int3 start;
    int3 dims;
    int3 segment_id;
    int id;

    int total_dependencies;
    int unfulfilled_dependencies;
    
    //std::vector<HaloExchangeTask> dependents;

    Device device;
    Stream stream;

    ComputationTask(int3 _segment_id, int3 nn, Device _device, Stream _stream)
    :segment_id(_segment_id), id(segment_id_to_index(_segment_id)), device(_device), stream(_stream)
    {
        start = (int3){
            segment_id.x == -1? NGHOST : segment_id.x == 1? nn.x : NGHOST*2,
            segment_id.y == -1? NGHOST : segment_id.y == 1? nn.y : NGHOST*2,
            segment_id.z == -1? NGHOST : segment_id.z == 1? nn.z : NGHOST*2
        };

        dims = (int3){
            segment_id.x == 0 ? nn.x - NGHOST*2 : NGHOST,
            segment_id.y == 0 ? nn.y - NGHOST*2 : NGHOST,
            segment_id.z == 0 ? nn.z - NGHOST*2 : NGHOST
        };

        int seg_type = segment_type(segment_id);
        total_dependencies = (int[]){0,NUM_COMPUTE_FACE_DEPENDENCIES,NUM_COMPUTE_EDGE_DEPENDENCIES, NUM_COMPUTE_CORNER_DEPENDENCIES}[seg_type];
        unfulfilled_dependencies = total_dependencies;
    }

    void
    compute(int isubstep, AcReal dt)
    {
        //std::cout << "Computing " << id <<std::endl;
        cudaSetDevice(device->id);
        acDeviceIntegrateSubstep(device, stream, isubstep, start, start+dims, dt);
    }

    void
    FulfillDependency(int isubstep, AcReal dt)
    {
        unfulfilled_dependencies--;
        //std::cout << "Dependency " << id <<std::endl;
        if (unfulfilled_dependencies == 0){
            compute(isubstep, dt);
            //Is this a stupid side-effect? Should I do this explicitly instead?
            unfulfilled_dependencies = total_dependencies;
        }
    }
} ComputationTask;


#define BUFFER_CHAIN_LENGTH 2

//HaloMessage contains all information needed to send or receive a single message
//It's really just a wrapped PackedData, and I feel like an MPI_Request should be added to that struct instead

typedef struct HaloMessage {
    PackedData buffer;
    MPI_Request *request;
    int length;

    HaloMessage(){
        
    }

    HaloMessage(int3 dims, MPI_Request* _req)
        :request(_req)
    {
        *request = MPI_REQUEST_NULL;
        buffer = acCreatePackedData(dims);
        length = dims.x * dims.y * dims.z * NUM_VTXBUF_HANDLES;
    }

/*
    ~HaloMessage(){
        std::cout<< "Destroying halo msg\n";
        acDestroyPackedData(&buffer);
        //For now, the grid takes care of freeing MPI requests
    }
*/
} HaloMessage;

typedef struct MessageBufferSwapChain {
    int buf_idx;
    std::vector<HaloMessage> buffers; //NAMING: buffer used twice already on different levels
    MessageBufferSwapChain()
        :buf_idx(0)
    {
        buffers.reserve(BUFFER_CHAIN_LENGTH);
    }
/*
    ~MessageBufferSwapChain(){
        std::cout<< "Destroying swap chain\n";
        wait_all();
        //buffers.clear();
    }
*/
    void add_buffer(HaloMessage buffer){
        buffers.push_back(buffer);
    }
    
    void emplace_buffer(int3 dims, MPI_Request* req){
        buffers.emplace_back(dims,req);
    }
    //May block to wait for a free buffer

    HaloMessage get_current_buffer(){
        return buffers[buf_idx];
    }

    HaloMessage get_fresh_buffer(){
        
        buf_idx = (buf_idx + 1) % BUFFER_CHAIN_LENGTH;
        //std::cout << "Buffer idx " << buf_idx <<std::endl;
        MPI_Request* req = buffers[buf_idx].request;
        //std::cout << "request " << *req << " at address " << req << std::endl;
        if (*req != MPI_REQUEST_NULL){
            MPI_Wait(req, MPI_STATUS_IGNORE);
        }
        return buffers[buf_idx];
    }

    void wait_all(){
        //TODO:IMPLEMENT
        std::cout << "WARNING, waiting for buffer chain not implemented! \n"; 
    }
} MessageBufferSwapChain;

typedef struct HaloExchangeTask{
    
    int3 segment_id; //direction from center of comp domain
    //TODO: rename b0 and a0 to halo_coord and grid_coord respectively
    int3 halo_segment_position;
    int3 local_segment_position;
    bool active;

    //MessageBufferSwapChain recv_buffers;
    HaloMessage* recv_buffer;
    MessageBufferSwapChain send_buffers;

    std::vector<ComputationTask*> dependents;

    int send_tag;
    int recv_tag;
    int msglen;
    int rank;
    int send_rank;
    int recv_rank;

    Device device;
    cudaStream_t stream;

    HaloExchangeTask(const Device _device, const int _rank, const int3 _segment_id, const uint3_64 decomposition, const int3 grid_dimensions, MPI_Request* recv_requests, MPI_Request* send_requests)
        :rank(_rank), segment_id(_segment_id), device(_device)
    {

        int seg_type = segment_type(segment_id);
        active = (( MPI_INCL_CORNERS ) || seg_type != 3)? true:false;

        //total_dependencies = (int[]){0,NUM_SEND_FACE_DEPENDENCIES,NUM_SEND_EDGE_DEPENDENCIES, NUM_SEND_CORNER_DEPENDENCIES}[seg_type];
        //unfulfilled_dependencies = total_dependencies;
        
        //foreign = segment_id => pipeline
        //foreign = -segment_id => mirror-symmetrical
        const int3 foreign_segment_id = -segment_id;

//Coordinates
        halo_segment_position = segment_id_to_halo_segment_position(segment_id, grid_dimensions);
        local_segment_position = segment_id_to_local_segment_position(-foreign_segment_id, grid_dimensions);

//MPI
        const int3 domain_coordinates = getPid3D(rank, decomposition);
        const int3 segment_dimensions = segment_id_to_dims(segment_id, grid_dimensions);
        

        send_rank = getPid(domain_coordinates + segment_id, decomposition);
        recv_rank = getPid(domain_coordinates - foreign_segment_id, decomposition);
        
        send_tag = segment_id_to_index(segment_id);
        recv_tag = segment_id_to_index(foreign_segment_id);


        recv_buffer = (HaloMessage*) malloc(sizeof(HaloMessage));
        *recv_buffer = HaloMessage(segment_dimensions,&(recv_requests[send_tag]));        
    
        send_buffers = MessageBufferSwapChain();
        for (int i = 0; i < BUFFER_CHAIN_LENGTH; i++){
            send_buffers.emplace_buffer(segment_dimensions,&(send_requests[i*NUM_SEGMENTS+send_tag]));
        }
//CUDA
        cudaSetDevice(device->id);

        int low_prio, high_prio;
        cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
        cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, high_prio);
    }

    //Copy constructor, looks like it still gets called somewhere
    //HaloExchangeTask(const HaloExchangeTask& other) = delete;

    ~HaloExchangeTask()
    {
        cudaSetDevice(device->id);
        dependents.clear();
        cudaStreamDestroy(stream);
    }
    
/*
    friend std::ostream& operator<<(std::ostream& os, HaloExchangeTask const & hxt){
            return os
                    << "{\"segment_id\": [" << hxt.segment_id.x <<","<< hxt.segment_id.y << ","<<hxt.segment_id.z << "]," 
                    << "\n\"rank\":" << hxt.rank  <<","
                    << "\n\"neighbor_rank\":" << hxt.neighbor_rank  <<","
                    << "\n\"send_tag\": " << hxt.send_tag << ", \"recv_tag\": " << hxt.recv_tag <<","
                    << "\n\"local position\": ["<< hxt.local_segment_position.x <<"," <<hxt.local_segment_position.y << ","<<hxt.local_segment_position.z << "],"
                    << "\n\"halo position\": ["<< hxt.halo_segment_position.x <<"," <<hxt.halo_segment_position.y << ","<<hxt.halo_segment_position.z << "]}";
    }
  */  
    void pack()
    {
        auto msg = send_buffers.get_fresh_buffer();
        //auto msg = *send_buffer;
        acKernelPackData(stream, device->vba, local_segment_position, msg.buffer);
    }

    void unpack()
    {   
        //auto msg = recv_buffers.get_current_buffer();
        auto msg = *recv_buffer;
        msg.buffer.pinned = false;
        acUnpinPackedData(device, stream, &(msg.buffer));
        acKernelUnpackData(stream, msg.buffer, halo_segment_position, device->vba);
    }
    void sync()
    {
        cudaStreamSynchronize(stream);
    }
    
    void receiveDevice()
    {
        //auto msg = recv_buffers.get_fresh_buffer();
        auto msg = *recv_buffer;
        MPI_Irecv(msg.buffer.data, msg.length, AC_MPI_TYPE, recv_rank, recv_tag, MPI_COMM_WORLD, msg.request);
        msg.buffer.pinned = false;
    }

    void receiveHost()
    {
        //auto msg = recv_buffers.get_fresh_buffer();
        auto msg = *recv_buffer;
        MPI_Irecv(msg.buffer.data_pinned, msg.length, AC_MPI_TYPE, recv_rank, recv_tag, MPI_COMM_WORLD, msg.request);
        msg.buffer.pinned = true;
    }


    void sendDevice()
    {
        sync();
        auto msg = send_buffers.get_current_buffer();
        //auto msg = *send_buffer;
        MPI_Isend(msg.buffer.data, msg.length, AC_MPI_TYPE, send_rank, send_tag, MPI_COMM_WORLD, msg.request);
    }
    
    void sendHost()
    {
        auto msg = send_buffers.get_current_buffer();
        //auto msg = *send_buffer;
        acPinPackedData(device, stream, &msg.buffer);
        sync();
        MPI_Isend(msg.buffer.data_pinned, msg.length, AC_MPI_TYPE, send_rank, send_tag, MPI_COMM_WORLD, msg.request);
    }

    void
    exchangeDevice()
    {
        //cudaSetDevice(device->id);
        receiveDevice();
        sendDevice();
    }

    void
    exchangeHost()
    {
        //cudaSetDevice(device->id);
        //TODO: is it sensible to always use CUDA memory for node-local exchanges?
        //What if the MPI lib doesn't support CUDA? 
        
        if (onTheSameNode(rank,recv_rank))
            receiveDevice();
        else
            receiveHost();

        if (onTheSameNode(rank,send_rank))
            sendDevice();
        else 
            sendHost();
    }
    
    void
    exchange()
    {
#if MPI_USE_PINNED == (1)
        exchangeHost();
#else 
        exchangeDevice();
#endif
    }

} HaloExchangeTask;

typedef struct Grid{
    Device device;
    AcMesh submesh;
    uint3_64 decomposition;
    bool initialized;

    std::vector<HaloExchangeTask> halo_exchange_tasks;
    std::vector<ComputationTask> outer_segments;
    ComputationTask* inner_domain;

    MPI_Request* recv_reqs;
    MPI_Request* send_reqs;

    /*
    void swapSendBuffers(){
        //noop if back_send_reqs consists of MPI_REQUEST_NULLs
        MPI_Waitall(NUM_SEGMENTS, back_send_reqs, MPI_STATUSES_IGNORE);
        std::swap(send_reqs,back_send_reqs);
        for (auto &halo_task: halo_exchange_tasks){
            halo_task.swapSendBuffers();
        }
    }
    */

} Grid;

static Grid grid = {};

AcResult
acGridSynchronizeStream(const Stream stream)
{
    ERRCHK(grid.initialized);

    acDeviceSynchronizeStream(grid.device, stream);
    MPI_Barrier(MPI_COMM_WORLD);
    return AC_SUCCESS;
}

AcResult
acGridInit(const AcMeshInfo info)
{
    ERRCHK(!grid.initialized);

    // Check that MPI is initialized
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Processor %s. Process %d of %d.\n", processor_name, pid, nprocs);

    // Decompose
    AcMeshInfo submesh_info      = info;
    const uint3_64 decomposition = decompose(nprocs);
    const int3 pid3d             = getPid3D(pid, decomposition);

    printf("Decomposition: %lu, %lu, %lu\n", decomposition.x, decomposition.y, decomposition.z);
    printf("Process %d: (%d, %d, %d)\n", pid, pid3d.x, pid3d.y, pid3d.z);
    ERRCHK_ALWAYS(info.int_params[AC_nx] % decomposition.x == 0);
    ERRCHK_ALWAYS(info.int_params[AC_ny] % decomposition.y == 0);
    ERRCHK_ALWAYS(info.int_params[AC_nz] % decomposition.z == 0);

    const int submesh_nx                       = info.int_params[AC_nx] / decomposition.x;
    const int submesh_ny                       = info.int_params[AC_ny] / decomposition.y;
    const int submesh_nz                       = info.int_params[AC_nz] / decomposition.z;
    submesh_info.int_params[AC_nx]             = submesh_nx;
    submesh_info.int_params[AC_ny]             = submesh_ny;
    submesh_info.int_params[AC_nz]             = submesh_nz;
    submesh_info.int3_params[AC_global_grid_n] = (int3){
        info.int_params[AC_nx],
        info.int_params[AC_ny],
        info.int_params[AC_nz],
    };
    submesh_info.int3_params[AC_multigpu_offset] = pid3d *
                                                   (int3){submesh_nx, submesh_ny, submesh_nz};
    acUpdateBuiltinParams(&submesh_info);

    // GPU alloc
    int devices_per_node = -1;
    cudaGetDeviceCount(&devices_per_node);

    Device device;
    acDeviceCreate(pid % devices_per_node, submesh_info, &device);
    
    // CPU alloc
    AcMesh submesh;
    acMeshCreate(submesh_info, &submesh);

    // Configure
    const int3 grid_dimensions = (int3){
        device->local_config.int_params[AC_nx],
        device->local_config.int_params[AC_ny],
        device->local_config.int_params[AC_nz],
    };

    // Setup the global grid structure
    grid.device = device;
    grid.submesh = submesh;
    grid.decomposition = decomposition;

    grid.outer_segments.clear();
    grid.outer_segments.reserve(NUM_SEGMENTS);

    for (int idx = 0; idx < NUM_SEGMENTS; idx++){
        const int3 segment_id = index_to_segment_id(idx);
        Stream stream = (Stream)(idx + STREAM_DEFAULT);
        grid.outer_segments.emplace_back(segment_id, grid_dimensions, device, stream);
    }
    
    grid.inner_domain = (ComputationTask*)malloc(sizeof(ComputationTask));
    *grid.inner_domain = ComputationTask((int3){0,0,0}, grid_dimensions, device, STREAM_26);

    grid.halo_exchange_tasks.clear();
    grid.halo_exchange_tasks.reserve(NUM_SEGMENTS);
    
    grid.recv_reqs = (MPI_Request*) malloc(sizeof(MPI_Request)*NUM_SEGMENTS);
    //grid.recv_reqs = (MPI_Request*) malloc(sizeof(MPI_Request)*NUM_SEGMENTS*BUFFER_CHAIN_LENGTH);
    grid.send_reqs = (MPI_Request*) malloc(sizeof(MPI_Request)*NUM_SEGMENTS*BUFFER_CHAIN_LENGTH);


    for (int i = 0; i < NUM_SEGMENTS; i++){
        grid.halo_exchange_tasks.emplace_back(device, pid, index_to_segment_id(i),
                                              decomposition, grid_dimensions,
                                              grid.recv_reqs, grid.send_reqs);

        const int3 seg_id = index_to_segment_id(i);
        for (int x = (seg_id.x == 0? -1: seg_id.x); x< 2 && (seg_id.x ==0 || seg_id.x == x); x++){
            for (int y = (seg_id.y == 0? -1: seg_id.y); y< 2 && (seg_id.y ==0 || seg_id.y == y); y++){
                for (int z = (seg_id.z == 0?  -1: seg_id.z); z< 2 && (seg_id.z ==0 || seg_id.z == z); z++){
                    const int dependent_idx = segment_id_to_index((int3){x,y,z});
                    grid.halo_exchange_tasks[i].dependents.push_back(&grid.outer_segments[dependent_idx]);
                }
            }
        }
    }
    /*
    for (int halo_exchange_idx = 0; halo_exchange_idx  < NUM_SEGMENTS; halo_exchange_idx++){

        const int3 local_seg_id = index_to_segment_id(halo_exchange_idx);
        const int3 halo_seg_id = -local_seg_id;

        for (int compute_idx = 0; compute_idx < NUM_SEGMENTS; compute_idx++){

            const int3 compute_seg_id = index_to_segment_id(compute_idx);

            if (  ((halo_seg_id.x == 0) || (halo_seg_id.x == compute_seg_id.x))
                &&((halo_seg_id.y == 0) || (halo_seg_id.y == compute_seg_id.y))
                &&((halo_seg_id.z == 0) || (halo_seg_id.z == compute_seg_id.z))){

                grid.halo_exchange_tasks[halo_exchange_idx].dependents.push_back(&grid.outer_segments[compute_idx]);
            }
        }
    }
    */
    grid.initialized = true;

    acGridSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acGridQuit(void)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(STREAM_ALL);

    grid.halo_exchange_tasks.clear();

    //MPI_Waitall(NUM_SEGMENTS, grid.recv_reqs, MPI_STATUSES_IGNORE);
    //MPI_Waitall(NUM_SEGMENTS, grid.send_reqs, MPI_STATUSES_IGNORE);
        
    for (int i = 0; i < NUM_SEGMENTS; i++){
        MPI_Request* req = &(grid.recv_reqs[i]);
        if (*req != MPI_REQUEST_NULL)
            MPI_Request_free(req);    
    }

    for (int i = 0; i < NUM_SEGMENTS*BUFFER_CHAIN_LENGTH; i++){
        MPI_Request* req = &(grid.send_reqs[i]);
        if (*req != MPI_REQUEST_NULL)
            MPI_Request_free(req);    
    }

    free(grid.recv_reqs);
    free(grid.send_reqs);

    grid.outer_segments.clear();
    free(grid.inner_domain);

    grid.initialized   = false;
    grid.decomposition = (uint3_64){0, 0, 0};
    acMeshDestroy(&grid.submesh);
    acDeviceDestroy(grid.device);

    acGridSynchronizeStream(STREAM_ALL);
    return AC_SUCCESS;
}

AcResult
acGridLoadMesh(const AcMesh host_mesh, const Stream stream)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    acDeviceDistributeMeshMPI(host_mesh, grid.decomposition, &grid.submesh);
    acDeviceLoadMesh(grid.device, stream, grid.submesh);

    return AC_SUCCESS;
}

AcResult
acGridStoreMesh(const Stream stream, AcMesh* host_mesh)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    acDeviceStoreMesh(grid.device, stream, &grid.submesh);
    acGridSynchronizeStream(stream);

    acDeviceGatherMeshMPI(grid.submesh, grid.decomposition, host_mesh);

    return AC_SUCCESS;
}

AcResult
acGridIntegrate(const Stream stream, const AcReal dt)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);
    const Device device = grid.device;

    acDeviceSynchronizeStream(device, stream);
    cudaSetDevice(device->id);
    MPI_Barrier(MPI_COMM_WORLD);

    for (int isubstep = 0; isubstep < 3; ++isubstep) {
        
#if MPI_COMM_ENABLED
        for (auto &halo_task : grid.halo_exchange_tasks){
            if (halo_task.active)
                halo_task.pack();
        }
        
        for (auto &halo_task : grid.halo_exchange_tasks){
            if (halo_task.active)
                halo_task.exchange();
        }
#endif //MPI_COMM_ENABLED

#if MPI_COMPUTE_ENABLED
        grid.inner_domain->compute(isubstep,dt);
#endif 
        
#if MPI_COMM_ENABLED
        //Handle messages as they arrive in a fused loop pipeline
        int idx, prev_idx;
        for (int n = 0; n < NUM_SEGMENTS+1; n++){
            prev_idx = idx;
            if (n < NUM_SEGMENTS){
                MPI_Waitany(NUM_SEGMENTS, grid.recv_reqs, &idx, MPI_STATUS_IGNORE);
                if (grid.halo_exchange_tasks[idx].active && idx >= 0 && idx < NUM_SEGMENTS){
                    grid.halo_exchange_tasks[idx].unpack();
                }
            }
            if(n > 0){
                if ( grid.halo_exchange_tasks[prev_idx].active && prev_idx >= 0 && prev_idx < NUM_SEGMENTS){
                    grid.halo_exchange_tasks[prev_idx].sync();
                    for (auto &dependent : grid.halo_exchange_tasks[prev_idx].dependents){
                        dependent->FulfillDependency(isubstep, dt);
                    }
                }
            }
        }
#endif 
        //for (auto &task : grid.outer_segments){
        //    task.compute(isubstep,dt);
        //}
        //MPI_Waitall(NUM_SEGMENTS, grid.send_reqs, MPI_STATUSES_IGNORE);
        //grid.swapSendBuffers();
        acDeviceSwapBuffers(device);
        acDeviceSynchronizeStream(device, STREAM_ALL); // Wait until inner and outer done
    }

    MPI_Waitall(NUM_SEGMENTS*BUFFER_CHAIN_LENGTH, grid.send_reqs, MPI_STATUSES_IGNORE);
    //<-... send
    //
    return AC_SUCCESS;
}

AcResult
acGridPeriodicBoundconds(const Stream stream)
{
    ERRCHK(grid.initialized);
    acGridSynchronizeStream(stream);

    MPI_Barrier(MPI_COMM_WORLD);

    for (auto &halo_task : grid.halo_exchange_tasks){ halo_task.pack(); }
    
    MPI_Barrier(MPI_COMM_WORLD);

    for (auto &halo_task : grid.halo_exchange_tasks){ halo_task.exchange(); }
    
    MPI_Waitall(NUM_SEGMENTS, grid.recv_reqs, MPI_STATUSES_IGNORE);    

    for (auto &halo_task : grid.halo_exchange_tasks){ halo_task.unpack(); }

    for (auto &halo_task : grid.halo_exchange_tasks){ halo_task.sync(); }

    MPI_Waitall(NUM_SEGMENTS, grid.send_reqs, MPI_STATUSES_IGNORE);    

    return AC_SUCCESS;
}
#endif // AC_MPI_ENABLED
