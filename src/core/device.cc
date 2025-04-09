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
#include "../../acc-runtime/api/math_utils.h"
#include "kernels/kernels.h"
#include "util_funcs.h"

#if AC_MPI_ENABLED
static int ac_pid()
{
	if(!acGridInitialized()) return 0;
	return ac_MPI_Comm_rank();
}
#else
static int ac_pid()
{
	return 0;
}
#endif

struct device_s {
    int id;
    AcMeshInfo local_config;
    AcInputs input;

    // Concurrency
    cudaStream_t streams[NUM_STREAMS];

    // Memory
    VertexBufferArray vba;
#if PACKED_DATA_TRANSFERS
    // Declare memory for buffers in device memory needed for packed data transfers.
    AcReal *plate_buffers[NUM_PLATE_BUFFERS];
#endif
    AcDeviceKernelOutput output;
};

#include <math.h>

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

template <typename T, typename P>
static AcResult
acDeviceLoadUniform(const Device device, const Stream stream, const P param, const T value)
{
	cudaSetDevice(device->id);
	return acLoadUniform(device->streams[stream], param, value);
}

AcMeshInfo
acDeviceGetLocalConfig(const Device device)
{
    return device->local_config;
}

AcResult
acDeviceGetId(const Device device, int* id)
{
    *id = device->id;
    return AC_SUCCESS;
}

AcResult
acDeviceLoadScalarUniform(const Device device, const Stream stream, const AcRealParam param,
                          const AcReal value)
{
	return acDeviceLoadUniform(device,stream,param,value);
}

AcResult
acDeviceLoadVectorUniform(const Device device, const Stream stream, const AcReal3Param param,
                          const AcReal3 value)
{
	return acDeviceLoadUniform(device,stream,param,value);
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

// Recursive function to generate indices
void generateIndicesHelper(const std::vector<size_t>& dimensions, std::vector<int>& currentIndex,
                           std::vector<std::vector<int>>& result, size_t depth) {
    if (depth == dimensions.size()) {
        result.push_back(currentIndex);
        return;
    }

    for (size_t i = 0; i < dimensions[depth]; ++i) {
        currentIndex[depth] = i;
        generateIndicesHelper(dimensions, currentIndex, result, depth + 1);
    }
}

// Main function to generate index range
std::vector<std::vector<int>> generateIndexRange(const std::vector<size_t>& dimensions) {
    std::vector<std::vector<int>> result;
    std::vector<int> currentIndex(dimensions.size(), 0);
    generateIndicesHelper(dimensions, currentIndex, result, 0);
    return result;
}

template <typename P, typename V>
static AcResult
acDeviceStoreUniform(const Device device, const Stream stream, const P param, V* value)
{
	cudaSetDevice(device->id);
	if constexpr (IsArrayParam(param))
	{
		auto column_to_row_order = [](const P array, const AcMeshInfo host_info, auto* src, auto* dst)
		{
			const int n_dims       = get_array_n_dims(array);
			const auto sizes_array = get_array_dim_sizes(array,host_info);
			std::vector<size_t> sizes{sizes_array.data(), sizes_array.data() + n_dims};
			auto column_major_index = [&](const auto& indexes)
			{
				size_t res = 0;
				size_t coeff = 1;
				for(int i = 0; i < n_dims; ++i)
				{
					res += coeff*indexes[i];
					coeff *= sizes[i];
				}
				return res;
			};
			auto row_major_index = [&](const auto& indexes)
			{
				size_t res = 0;
				size_t coeff = 1;
				for(int i = n_dims-1; i >= 0; --i)
				{
					res += coeff*indexes[i];
					coeff *= sizes[i];
				}
				return res;
			};
			auto index_range = generateIndexRange(sizes);
			for(const auto& index : index_range)
				dst[row_major_index(index)] = src[column_major_index(index)];
		};
		const size_t len = get_array_length(param,device->local_config);
		V* dst = device->local_config[AC_host_has_row_memory_order] ? (V*)malloc(sizeof(V)*len) : value;
		ERRCHK_ALWAYS(acStoreUniform(param, dst, len) == AC_SUCCESS);
		if(device->local_config[AC_host_has_row_memory_order])
		{
			column_to_row_order(param,device->local_config,dst,value);
			free(dst);
		}
		return AC_SUCCESS;
	}
	else
		return acStoreUniform(device->streams[stream], param, value);
}
#define GEN_DEVICE_STORE_UNIFORM(PARAM_TYPE,VAL_TYPE,VAL_TYPE_UPPER_CASE) \
	AcResult \
	acDeviceStore##VAL_TYPE_UPPER_CASE##Uniform(const Device device, const Stream stream, const PARAM_TYPE param, VAL_TYPE* value) \
	{ \
    		return acDeviceStoreUniform(device, stream, param, value);\
	}
#define GEN_DEVICE_STORE_ARRAY(PARAM_TYPE,VAL_TYPE,VAL_TYPE_UPPER_CASE) \
	AcResult \
	acDeviceStore##VAL_TYPE_UPPER_CASE##Array(const Device device, const Stream stream, const PARAM_TYPE param, VAL_TYPE* value) \
	{ \
    		return acDeviceStoreUniform(device, stream, param, value);\
	}
#include "device_store_uniform.h"

AcResult
acDeviceUpdate(Device device, const AcMeshInfo config)
{
    acUpdateArrays(config);
    acDeviceLoadMeshInfo(device,config);
    return AC_SUCCESS;
}




template <typename P>
AcResult
acDeviceLoadArray(const Device device, const Stream stream, const AcMeshInfo host_info, const P array)
{
	auto row_to_column_order = [&]()
	{
		auto* src = host_info[array];
		const size_t len = get_array_length(array,host_info);
		auto* dst = (decltype(src)) malloc(sizeof(decltype(src))*len);
		const int n_dims       = get_array_n_dims(array);
		const auto sizes_array = get_array_dim_sizes(array,host_info);
		std::vector<size_t> sizes{sizes_array.data(), sizes_array.data() + n_dims};
		auto column_major_index = [&](const auto& indexes)
		{
			size_t res = 0;
			size_t coeff = 1;
			for(int i = 0; i < n_dims; ++i)
			{
				res += coeff*indexes[i];
				coeff *= sizes[i];
			}
			return res;
		};
		auto row_major_index = [&](const auto& indexes)
		{
			size_t res = 0;
			size_t coeff = 1;
			for(int i = n_dims-1; i >= 0; --i)
			{
				res += coeff*indexes[i];
				coeff *= sizes[i];
			}
			return res;
		};
		auto index_range = generateIndexRange(sizes);
		for(const auto& index : index_range)
			dst[column_major_index(index)] = src[row_major_index(index)];
		return dst;
	};

	cudaSetDevice(device->id);
	if(device->local_config[AC_host_has_row_memory_order])
	{
		auto* values = row_to_column_order();
		auto res= acLoadUniform(device->streams[stream],array,values,get_array_length(array,host_info));
		free(values);
		return res;
	}
	return acLoadUniform(device->streams[stream],array,host_info[array], get_array_length(array,host_info));
}




template <typename P>
struct load_all_scalars_uniform
{
	AcResult operator()(const Device device, const AcMeshInfo config)
	{
		AcResult res = AC_SUCCESS;
		for(P i : get_params<P>())
			res = acDeviceLoadUniform(device, STREAM_DEFAULT, i, config[i]) ? res : AC_FAILURE;
		return res;
	}
};

template <typename P>
struct load_all_arrays_uniform
{
	AcResult operator()(const Device device, const AcMeshInfo device_config)
	{
		AcResult res = AC_SUCCESS;
		for(P array : get_params<P>())
		{
			auto config_array = device_config[array];
      			if (config_array != nullptr)
				res = acDeviceLoadArray(device,STREAM_DEFAULT,device_config,array) ? res : AC_FAILURE;
			acDeviceSynchronizeStream(device,STREAM_ALL);
		}
		return res;
	}
};

AcResult
acDeviceLoadMeshInfo(const Device device, const AcMeshInfo config)
{
    cudaSetDevice(device->id);

    AcMeshInfo device_config = config;
    acHostUpdateParams(&device_config);

    ERRCHK_ALWAYS(device_config[AC_nlocal] == device->local_config[AC_nlocal]);
    ERRCHK_ALWAYS(device_config[AC_multigpu_offset] == device->local_config[AC_multigpu_offset]);

    AcScalarTypes::run<load_all_scalars_uniform>(device,device_config);
    AcArrayTypes::run<load_all_arrays_uniform>(device, device_config);

    // OL: added this assignment to make sure that whenever we load a new config,
    // it's updated on both the host Device structure, and the GPU
    device->local_config = device_config;

    acDeviceLoadStencilsFromConfig(device, STREAM_DEFAULT);
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

AcResult
acDeviceLoadStencilsFromConfig(const Device device, const Stream stream)
{
	[[maybe_unused]] auto DCONST = [&](const auto& param)
	{
		return device->local_config[param];
	};
	#include "coeffs.h"
	for(int stencil=0;stencil<NUM_STENCILS;stencil++)
	{
	        for(int x = 0; x<STENCIL_WIDTH; ++x)
	        {
	                for(int y=0;y<STENCIL_HEIGHT;++y)
	                {
	                        for(int z=0;z<STENCIL_DEPTH;++z)
	                        {
	                                if(isnan(stencils[stencil][x][y][z]))
	                                {
	                                        printf("loading a nan to stencil: %s, at %d,%d,%d!!\n", stencil_names[stencil],x,y,z);
	                                }
	                        }
	                }
	        }
	}
	return acDeviceLoadStencils(device, stream, stencils);
}
AcBoundary
acDeviceStencilAccessesBoundaries(const Device device, const Stencil stencil)
{
	[[maybe_unused]] auto DCONST = [&](const auto& param)
	{
		return device->local_config[param];
	};
        #include "coeffs.h"
	auto stencil_accesses_z_ghost_zone = [&]()
	{
	    bool res = false;
	    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
	      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
	        for (int width = 0; width < STENCIL_WIDTH; ++width) {
		  const bool dmid = (depth == (STENCIL_DEPTH-1)/2);
	          res |= !dmid && (stencils[stencil][depth][height][width] != (AcReal)0.0);
	        }
	      }
	    }
	    return res;
	};
	auto stencil_accesses_y_ghost_zone = [&]()
	{
	  // Check which stencils are invalid for profiles
	  // (computed in a new array to avoid side effects).
	    bool res = false;
	    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
	      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
	        for (int width = 0; width < STENCIL_WIDTH; ++width) {
		  const bool hmid = (height == (STENCIL_HEIGHT-1)/2);
	          res |= !hmid && (stencils[stencil][depth][height][width] != (AcReal)0.0);
	        }
	      }
	    }
	    return res;
	};
	
	auto stencil_accesses_x_ghost_zone = [&]()
	{
	  // Check which stencils are invalid for profiles
	  // (computed in a new array to avoid side effects).
	    bool res = false;
	    for (int depth = 0; depth < STENCIL_DEPTH; ++depth) {
	      for (int height = 0; height < STENCIL_HEIGHT; ++height) {
	        for (int width = 0; width < STENCIL_WIDTH; ++width) {
		  const bool wmid = (width == (STENCIL_WIDTH-1)/2);
	          res |= !wmid && (stencils[stencil][depth][height][width] != (AcReal)0.0);
	        }
	      }
	    }
	    return res;
	};
	int res = BOUNDARY_NONE;
	if(stencil_accesses_x_ghost_zone())
		res |= BOUNDARY_X;
	if(stencil_accesses_y_ghost_zone())
		res |= BOUNDARY_Y;
	if(stencil_accesses_z_ghost_zone())
		res |= BOUNDARY_Z;
	return (AcBoundary)res;
}


void
acCopyFromInfo(const AcMeshInfo src, AcMeshInfo dst, AcInt3Param param)
{
	dst[param] = src[param];
}
void
acCopyFromInfo(const AcMeshInfo, AcMeshInfo, const int3){}

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
    memset(&device->input,0,sizeof(device->input));
    memset(&device->output,0,sizeof(device->output));

    // Check that AC_global_grid_n and AC_multigpu_offset are valid
    // Replace if not and give a warning otherwise
    if (
        device->local_config[AC_multigpu_offset].x < 0 ||
        device->local_config[AC_multigpu_offset].y < 0 ||
        device->local_config[AC_multigpu_offset].z < 0) {
	WARNING("Invalid AC_multigpu_offset passed in device_config to acDeviceCreate. Replacting with AC_multigpu_offset = (int3){0,0,0}.");
        device->local_config[AC_multigpu_offset] = (int3){0, 0, 0};
    }
    if(
        device->local_config[AC_nlocal].x <= 0 ||
        device->local_config[AC_nlocal].y <= 0 ||
        device->local_config[AC_nlocal].z <= 0
      )
    {
        WARNING("Invalid AC_nlocal passed in device_config to "
                "acDeviceCreate. Replacing with AC_nlocal = AC_ngrid"
                );
	device->local_config[AC_nlocal] = device->local_config[AC_ngrid];
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

    acVerboseLogFromRootProc(ac_pid(), "memusage before create streams= %f MBytes\n", memusage()/1024.0);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreateWithPriority(&device->streams[i], cudaStreamNonBlocking, i);
    }
    acVerboseLogFromRootProc(ac_pid(),  "memusage after create streams= %f MBytes\n", memusage()/1024.0);

    // Memory
    // VBA in/out
    device->vba = acVBACreate(device_config);
    acDeviceSynchronizeStream(device,STREAM_ALL);

// Allocate any data buffer required for packed transfers here.
#if PACKED_DATA_TRANSFERS
// Buffer for packed transfer of halo plates.
    ERRCHK_CUDA_ALWAYS(
        cudaMalloc((void**)&(device->plate_buffers[AC_XZ|AC_BOT]), device->local_config[AC_xz_plate_bufsize]*sizeof(AcReal)));
    ERRCHK_CUDA_ALWAYS(
        cudaMalloc((void**)&(device->plate_buffers[AC_YZ|AC_BOT]), device->local_config[AC_yz_plate_bufsize]*sizeof(AcReal)));
    ERRCHK_CUDA_ALWAYS(
        cudaMalloc((void**)&(device->plate_buffers[AC_XZ|AC_TOP]), device->local_config[AC_xz_plate_bufsize]*sizeof(AcReal)));
    ERRCHK_CUDA_ALWAYS(
        cudaMalloc((void**)&(device->plate_buffers[AC_YZ|AC_TOP]), device->local_config[AC_yz_plate_bufsize]*sizeof(AcReal)));
//printf("device plate buffer pointers= %p %p %p %p \n", device->plate_buffers[AC_XZ|AC_BOT], device->plate_buffers[AC_YZ|AC_BOT], 
//                                                       device->plate_buffers[AC_XZ|AC_TOP], device->plate_buffers[AC_YZ|AC_TOP]);
#endif
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

AcResult acDeviceGetVertexBufferPtrs(Device device, const VertexBufferHandle vtxbuf, AcReal** in, AcReal** out) {
    *in  = device->vba.on_device.in[vtxbuf];
    *out = device->vba.on_device.out[vtxbuf];
    return AC_SUCCESS;
}

AcResult
acDeviceDestroy(Device* device_ptr)
{
    Device device = *device_ptr;
    if(device == NULL) return AC_SUCCESS;
    cudaSetDevice(device->id);
#if AC_VERBOSE
    printf("Destroying device %d (%p)\n", device->id, device);
#endif
    acDeviceSynchronizeStream(device, STREAM_ALL);

    // Memory
    acVBADestroy(&device->vba,device->local_config);
    
#if PACKED_DATA_TRANSFERS
// Free data required for packed tranfers here (cudaFree)
    for (int i=0; i<NUM_PLATE_BUFFERS; i++)
        cudaFree(device->plate_buffers[i]);
#endif

    // Concurrency
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamDestroy(device->streams[i]);
    }

    // Destroy Device
    free(device);
    *device_ptr = NULL;
    return AC_SUCCESS;
}

AcResult
acDeviceSwapBuffer(const Device device, const VertexBufferHandle handle)
{
    cudaSetDevice(device->id);

    AcReal* tmp             = device->vba.on_device.in[handle];
    device->vba.on_device.in[handle]  = device->vba.on_device.out[handle];
    device->vba.on_device.out[handle] = tmp;

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
AcResult
acDeviceLoadVertexBufferWithOffset(const Device device, const Stream stream, const AcMesh host_mesh,
                                   const VertexBufferHandle vtxbuf_handle, const int3 src,
                                   const int3 dst, const int num_vertices)
{
    //TP: to still allow loading the whole mesh even though some VertexBuffers are dead, loading dead VertexBuffers is a no-op
    if (!vtxbuf_is_alive[vtxbuf_handle]) return AC_NOT_ALLOCATED;
    cudaSetDevice(device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, host_mesh.info);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, device->local_config);

    const AcReal* src_ptr = &host_mesh.vertex_buffer[vtxbuf_handle][src_idx];
    AcReal* dst_ptr       = &device->vba.on_device.in[vtxbuf_handle][dst_idx];
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
    const size_t device_num_vertices = acVertexBufferSize(device->local_config);
    const size_t host_num_vertices   = acVertexBufferSize(host_mesh.info);
    ERRCHK_ALWAYS(device_num_vertices == host_num_vertices);
    return acDeviceLoadVertexBufferWithOffset(device, stream, host_mesh, vtxbuf_handle, src, dst,
                                       host_num_vertices);
}

#define GEN_DEVICE_LOAD_ARRAY(PARAM_NAME, VAL_NAME, NAME_UPPER_CASE) \
	AcResult \
	acDeviceLoad##NAME_UPPER_CASE##Array(const Device device, const Stream stream, const AcMeshInfo host_info, const PARAM_NAME array) \
	{ \
		return acDeviceLoadArray(device,stream,host_info,array); \
	}

#define GEN_DEVICE_LOAD_UNIFORM(PARAM_TYPE,VAL_TYPE,VAL_TYPE_UPPER_CASE) \
	AcResult \
	acDeviceLoad##VAL_TYPE_UPPER_CASE##Uniform(const Device device, const Stream stream, const PARAM_TYPE param, const VAL_TYPE value) \
	{ \
		return acDeviceLoadUniform(device,stream,param,value); \
	}


#include "device_load_uniform.h"

AcResult
acDeviceLoadMesh(const Device device, const Stream stream, const AcMesh host_mesh)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
	if (!vtxbuf_is_alive[i]) continue;
        acDeviceLoadVertexBuffer(device, stream, host_mesh, (VertexBufferHandle)i);
    }

    return AC_SUCCESS;
}

AcResult
acDeviceSetVertexBuffer(const Device device, const Stream stream, const VertexBufferHandle handle,
                        const AcReal value)
{
    if(!vtxbuf_is_alive[handle]) return AC_NOT_ALLOCATED;
    cudaSetDevice(device->id);

    const size_t count = acVertexBufferSize(device->local_config);
    AcReal* data       = (AcReal*)calloc(count, sizeof(AcReal));
    ERRCHK_ALWAYS(data);

    for (size_t i = 0; i < count; ++i)
        data[i] = value;

    // Set both in and out for safety (not strictly needed)
    ERRCHK_CUDA_ALWAYS(cudaMemcpyAsync(device->vba.on_device.in[handle], data, sizeof(data[0]) * count,
                                       cudaMemcpyHostToDevice, device->streams[stream]));
    ERRCHK_CUDA_ALWAYS(cudaMemcpyAsync(device->vba.on_device.out[handle], data, sizeof(data[0]) * count,
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
    {
    	if(!vtxbuf_is_alive[i]) continue;
        retval |= acKernelFlush(device->streams[stream], device->vba.on_device.out[i], count, (AcReal)0.0);
    }

    return (AcResult)retval;
}

AcResult
acDeviceStoreVertexBufferWithOffset(const Device device, const Stream stream,
                                    const VertexBufferHandle vtxbuf_handle, const int3 src,
                                    const int3 dst, const int num_vertices, AcMesh* host_mesh)
{
    //TP: to still allow storing the whole mesh back from the Device storing dead VertexBuffers is a no-op
    if(!vtxbuf_is_alive[vtxbuf_handle]) return AC_NOT_ALLOCATED;
    cudaSetDevice(device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, device->local_config);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, host_mesh->info);


    const AcReal* src_ptr = &device->vba.on_device.in[vtxbuf_handle][src_idx];
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
    const size_t device_num_vertices = acVertexBufferSize(device->local_config);
    const size_t host_num_vertices = acVertexBufferSize(host_mesh->info);
    if(device_num_vertices != host_num_vertices)
    {
	fprintf(stderr,"Host dims: %d,%d,%d\n",host_mesh->info[AC_mlocal].x,host_mesh->info[AC_mlocal].y,host_mesh->info[AC_mlocal].z);
	fprintf(stderr,"Device dims: %d,%d,%d\n",device->local_config[AC_mlocal].x,device->local_config[AC_mlocal].y,device->local_config[AC_mlocal].z);
	fflush(stderr);
    	ERRCHK_ALWAYS(device_num_vertices == host_num_vertices);
    }

    return acDeviceStoreVertexBufferWithOffset(device, stream, vtxbuf_handle, src, dst, host_num_vertices,
                                        host_mesh);
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
    //TP: to still allow transfering the whole mesh between devices transfering dead VertexBuffers is a no-op
    if(!vtxbuf_is_alive[vtxbuf_handle]) return AC_NOT_ALLOCATED;
    cudaSetDevice(src_device->id);
    const size_t src_idx = acVertexBufferIdx(src.x, src.y, src.z, src_device->local_config);
    const size_t dst_idx = acVertexBufferIdx(dst.x, dst.y, dst.z, dst_device->local_config);

    const AcReal* src_ptr = &src_device->vba.on_device.in[vtxbuf_handle][src_idx];
    AcReal* dst_ptr       = &dst_device->vba.on_device.in[vtxbuf_handle][dst_idx];
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

    return acDeviceTransferVertexBufferWithOffset(src_device, stream, vtxbuf_handle, src, dst,
                                           num_vertices, dst_device);
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
acDeviceLaunchKernel(const Device device, const Stream stream, const AcKernel kernel,
                     const Volume start, const Volume end)
{
    cudaSetDevice(device->id);
    return acLaunchKernel(kernel, device->streams[stream], start, end, device->vba);
}



AcResult
acDeviceBenchmarkKernel(const Device device, const AcKernel kernel, const int3 start, const int3 end)
{
    cudaSetDevice(device->id);
    return acBenchmarkKernel(kernel, start, end, device->vba);
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
                         const Volume start, const Volume end, const AcReal dt)
{
#ifdef AC_INTEGRATION_ENABLED
    const AcReal current_time = device->local_config[AC_current_time];
    cudaSetDevice(device->id);

#ifdef AC_SINGLEPASS_INTEGRATION
    device->vba.on_device.kernel_input_params.singlepass_solve.step_num = step_number;
    device->vba.on_device.kernel_input_params.singlepass_solve.time_params = {dt,current_time};
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

    device->vba.on_device.kernel_input_params.twopass_solve_intermediate.step_num = AC_SUBSTEP_NUMBER(step_number);
    device->vba.on_device.kernel_input_params.twopass_solve_intermediate.dt = dt;
    const AcResult res = acLaunchKernel(twopass_solve_intermediate, device->streams[stream], start,
                                        end, device->vba);
    if (res != AC_SUCCESS)
        return res;

    acDeviceSwapBuffers(device);
    device->vba.on_device.kernel_input_params.twopass_solve_final.current_time = current_time;

    device->vba.on_device.kernel_input_params.twopass_solve_final.step_num = step_number;
    device->vba.on_device.kernel_input_params.twopass_solve_final.current_time= current_time;
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
                              const VertexBufferHandle vtxbuf_handle, const Volume start,
                              const Volume end)
{
    cudaSetDevice(device->id);
    if(!vtxbuf_is_alive[vtxbuf_handle]) return AC_NOT_ALLOCATED;
    acLoadKernelParams(device->vba.on_device.kernel_input_params,BOUNDCOND_PERIODIC_DEVICE,vtxbuf_handle); 
    return acDeviceLaunchKernel(device, stream, BOUNDCOND_PERIODIC_DEVICE,start,end);
}

AcResult
acDevicePeriodicBoundconds(const Device device, const Stream stream, const Volume start,
                           const Volume end)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDevicePeriodicBoundcondStep(device, stream, (VertexBufferHandle)i, start, end);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceGeneralBoundcondStep(const Device device, const Stream ,
                             const VertexBufferHandle vtxbuf_handle, const Volume ,
                             const Volume , const AcMeshInfo , const int3 )
{
    if(!vtxbuf_is_alive[vtxbuf_handle]) return AC_NOT_ALLOCATED;
    cudaSetDevice(device->id);
    fprintf(stderr,"acDeviceGenerelBoundCondStep NOT ANYMORE SUPPORTED\n");
    exit(EXIT_FAILURE);
}

AcResult
acDeviceGeneralBoundconds(const Device device, const Stream stream, const Volume start,
                          const Volume end, const AcMeshInfo config, const int3 bindex)
{
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i) {
        acDeviceGeneralBoundcondStep(device, stream, (VertexBufferHandle)i, start, end, config,
                                     bindex);
    }
    return AC_SUCCESS;
}

//static int3
//constructInt3Param(const Device device, const AcIntParam a, const AcIntParam b, const AcIntParam c)
//{
//    return (int3){
//        device->local_config.int_params[a],
//        device->local_config.int_params[b],
//        device->local_config.int_params[c],
//    };
//}

AcResult
acDeviceReduceScalNoPostProcessing(const Device device, const Stream stream, const AcReduction reduction,
                              const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    if(!vtxbuf_is_alive[vtxbuf_handle]) return AC_NOT_ALLOCATED;
    cudaSetDevice(device->id);

    const Volume start = acGetMinNN(device->local_config);
    const Volume end   = acGetMaxNN(device->local_config);

    *result = acKernelReduceScal(device->streams[stream], reduction, vtxbuf_handle,
                                 start, end, AC_default_real_output, device->vba);
    return AC_SUCCESS;
}

AcResult
acDeviceReduceScal(const Device device, const Stream stream, const AcReduction reduction,
                   const VertexBufferHandle vtxbuf_handle, AcReal* result)
{
    if(!vtxbuf_is_alive[vtxbuf_handle]) return AC_NOT_ALLOCATED;
    acDeviceReduceScalNoPostProcessing(device, stream, reduction, vtxbuf_handle, result);

    switch (reduction.post_processing_op) {
    	case AC_RMS: {
    	    const Volume nn = acGetLocalNN(device->local_config);
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
acDeviceReduceVecNoPostProcessing(const Device device, const Stream stream, const AcReduction reduction,
                             const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                             const VertexBufferHandle vtxbuf2, AcReal* result)
{
    if(!vtxbuf_is_alive[vtxbuf0]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf1]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf2]) return AC_NOT_ALLOCATED;
    cudaSetDevice(device->id);

    const Volume start = acGetMinNN(device->local_config);
    const Volume end   = acGetMaxNN(device->local_config);

    *result = acKernelReduceVec(device->streams[stream], reduction, start, end, {vtxbuf0,vtxbuf1,vtxbuf2},device->vba,
                                AC_default_real_output);
    return AC_SUCCESS;
}

AcResult
acDeviceReduceVec(const Device device, const Stream stream, const AcReduction reduction,
                  const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                  const VertexBufferHandle vtxbuf2, AcReal* result)
{
    if(!vtxbuf_is_alive[vtxbuf0]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf1]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf2]) return AC_NOT_ALLOCATED;
    acDeviceReduceVecNoPostProcessing(device, stream, reduction, vtxbuf0, vtxbuf1, vtxbuf2, result);
    switch (reduction.post_processing_op) {
    	case AC_RMS: {
    	    const Volume nn = acGetLocalNN(device->local_config);
    	    const AcReal inv_n = AcReal(1.) / (nn.x * nn.y * nn.z);
    	    *result            = sqrt(inv_n * *result);
    	    break;
    	}
    	default: /* Do nothing */
        	break;
    };

    return AC_SUCCESS;
}

#include "device_finalize_reduce.h"
AcResult
acDeviceReduceVecScalNoPostProcessing(const Device device, const Stream stream,
                                 const AcReduction reduction, const VertexBufferHandle vtxbuf0,
                                 const VertexBufferHandle vtxbuf1, const VertexBufferHandle vtxbuf2,
                                 const VertexBufferHandle vtxbuf3, AcReal* result)
{
    if(!vtxbuf_is_alive[vtxbuf0]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf1]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf2]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf3]) return AC_NOT_ALLOCATED;
    cudaSetDevice(device->id);

    const Volume start = acGetMinNN(device->local_config);
    const Volume end   = acGetMaxNN(device->local_config);

    *result = acKernelReduceVecScal(device->streams[stream], reduction, start, end,
		    		    {vtxbuf0,vtxbuf1,vtxbuf2,vtxbuf3},
				    device->vba,
                                    AC_default_real_output);
    return AC_SUCCESS;
}

AcResult
acDeviceReduceVecScal(const Device device, const Stream stream, const AcReduction reduction,
                      const VertexBufferHandle vtxbuf0, const VertexBufferHandle vtxbuf1,
                      const VertexBufferHandle vtxbuf2, const VertexBufferHandle vtxbuf3,
                      AcReal* result)
{
    if(!vtxbuf_is_alive[vtxbuf0]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf1]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf2]) return AC_NOT_ALLOCATED;
    if(!vtxbuf_is_alive[vtxbuf3]) return AC_NOT_ALLOCATED;
    acDeviceReduceVecScalNoPostProcessing(device, stream, reduction, vtxbuf0, vtxbuf1, vtxbuf2, vtxbuf3,
                                     result);
    switch (reduction.post_processing_op) {
    	case AC_RMS: {
    	    const Volume nn = acGetLocalNN(device->local_config);
    	    const AcReal inv_n = AcReal(1.) / (nn.x * nn.y * nn.z);
    	    *result            = sqrt(inv_n * *result);
    	    break;
    	}
        case AC_RADIAL_WINDOW_RMS: {
	   ERROR("AC_RMS_RADIAL_WINDOW not implemented for acDeviceReduceVecScal\n");
	   break;
	}
    	default: /* Do nothing */
        	break;
    };

    return AC_SUCCESS;
}

/** XY averages */
AcResult
acDeviceReduceXY(const Device device, const Stream stream, const Field field,
                        const Profile profile, const AcReduction reduction)
{
    if (profile >= 0 && profile < NUM_PROFILES) {
        cudaSetDevice(device->id);
        acDeviceSynchronizeStream(device, stream);

        const AcMeshDims dims = acGetMeshDims(device->local_config);
        for (size_t k = 0; k < dims.m1.z; ++k) {
            const Volume start    = (Volume){dims.n0.x, dims.n0.y, k};
            const Volume end      = (Volume){dims.n1.x, dims.n1.y, k + 1};
            const size_t nxy    = (end.x - start.x) * (end.y - start.y);
            const AcReal result = AcReal(1. / nxy) * acKernelReduceScal(device->streams[stream],
                                                                  reduction, field,
                                                                  start, end,
								  AC_default_real_output,
                                                                  device->vba);

            // printf("%zu Profile: %g\n", k, result);
            // Could be optimized by performing the reduction completely in
            // device memory without the redundant device-host-device transfer
            cudaMemcpy(&device->vba.on_device.profiles.in[profile][k], &result, sizeof(result),
                       cudaMemcpyHostToDevice);
        }
        return AC_SUCCESS;
    }
    else {
        return AC_FAILURE;
    }
}

AcResult
acDeviceSwapProfileBuffer(const Device device, const Profile handle)
{
    cudaSetDevice(device->id);

    AcReal* tmp                      = device->vba.on_device.profiles.in[handle];
    device->vba.on_device.profiles.in[handle]  = device->vba.on_device.profiles.out[handle];
    device->vba.on_device.profiles.out[handle] = tmp;

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
    for (int i = 0; i < NUM_PROFILES; ++i)
        retval |= acDeviceSwapProfileBuffer(device, (Profile)i);

    return (AcResult)retval;
}

AcResult
acDeviceLoadProfile(const Device device, const AcReal* hostprofile, const size_t hostprofile_count,
                    const Profile profile)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    cudaSetDevice(device->id);
    ERRCHK_ALWAYS(hostprofile_count == device->vba.profile_count);
    ERRCHK_CUDA(cudaMemcpy(device->vba.on_device.profiles.in[profile], hostprofile,
                           sizeof(device->vba.on_device.profiles.in[profile][0]) * device->vba.profile_count,
                           cudaMemcpyHostToDevice));
    return AC_SUCCESS;
}



AcResult
acDeviceStoreProfile(const Device device, const Profile profile, AcMesh* host_mesh)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    cudaSetDevice(device->id);
    ERRCHK_CUDA(cudaMemcpy(host_mesh->profile[profile], device->vba.on_device.profiles.in[profile],
                           prof_size(profile,device->vba.dims.m1),
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
    for (int i = 0; i < NUM_PROFILES; ++i) {
        const size_t count = device->vba.profile_count;
        AcReal* host_profile = (AcReal*)malloc(sizeof(AcReal)*count);
        cudaMemcpy(host_profile, device->vba.on_device.profiles.in[i], sizeof(AcReal) * count,
                   cudaMemcpyDeviceToHost);
        printf("Profile %s (%d)-----------------\n", profile_names[i], i);
        for (size_t j = 0; j < count; ++j) {
            printf("%g (%zu), ", (double)host_profile[j], j);
        }
        printf("\n");
	free(host_profile);
    }
    return AC_SUCCESS;
}

AcResult
acDeviceVolumeCopy(const Device device, const Stream stream,                     //
                   const AcReal* in, const Volume in_offset, const Volume in_volume, //
                   AcReal* out, const Volume out_offset, const Volume out_volume)
{
    cudaSetDevice(device->id);
    return acKernelVolumeCopy(device->streams[stream], in, in_offset, in_volume, out, out_offset,
                              out_volume);
}

#if PACKED_DATA_TRANSFERS 
// Functions for calling packed data transfers
AcResult
acDeviceLoadPlateBuffer(const Device device, int3 start, int3 end, const Stream stream, AcReal* buffer, int plate)
{
    const int size_x=end.x-start.x, size_y=end.y-start.y, size_z=end.z-start.z;
    const int block_size = size_x*size_y*size_z;
    const int bufsiz = block_size*NUM_VTXBUF_HANDLES*sizeof(AcReal);
/*
printf("acDeviceLoadPlateBuffer:start,end= %d %d %d %d %d %d \n", start.x, start.y, start.z, end.x, end.y, end.z);
printf("acDeviceLoadPlateBuffer:bufsiz,block_size= %u %u\n",bufsiz,block_size);
printf("acDeviceLoadPlateBuffer:buffer= %p \n", buffer);
*/
printf("acDeviceLoadPlateBuffer:plate, device->plate_buffer= %d %p\n", plate, device->plate_buffers[plate]);
    cudaSetDevice(device->id);

    ERRCHK_CUDA(
        cudaMemcpyAsync(device->plate_buffers[plate], buffer, bufsiz,
                        cudaMemcpyHostToDevice, device->streams[stream])
    );
//  unpacking in global memory; done by GPU kernel "packUnpackPlate".
    acUnpackPlate(device, start, end, block_size, stream, plate);

    return AC_SUCCESS;
}

AcResult
acDeviceStorePlateBuffer(const Device device, int3 start, int3 end, const Stream stream, AcReal* buffer, int plate)
{
    const int size_x=end.x-start.x, size_y=end.y-start.y, size_z=end.z-start.z;
    const int block_size = size_x*size_y*size_z;
    const int bufsiz = block_size*NUM_VTXBUF_HANDLES*sizeof(AcReal);
/*
printf("acDeviceStorePlateBuffer:start,end,type= %d %d %d %d %d %d %d\n", start.x, start.y, start.z, end.x, end.y, end.z, plate);
printf("acDeviceStorePlateBuffer:bufsiz,block_size= %u %u\n",bufsiz,block_size);
printf("acDeviceStorePlateBuffer:buffer= %p \n", buffer);
*/
printf("acDeviceStorePlateBuffer:plate, device->plate_buffer= %d %p \n", plate, device->plate_buffers[plate]);
    cudaSetDevice(device->id);

//  packing from global memory; done by GPU kernel "packUnpackPlate".
    acPackPlate(device, start, end, block_size, stream, plate);
    ERRCHK_CUDA(cudaMemcpyAsync(buffer,device->plate_buffers[plate], bufsiz,
                                cudaMemcpyDeviceToHost, device->streams[stream])
    );

    return AC_SUCCESS;
}

AcResult
acDeviceStoreIXYPlate(const Device device, int3 start, int3 end, int src_offset, const Stream stream, AcMesh *host_mesh)
{
    cudaSetDevice(device->id);     // use first device

    int px=host_mesh->info[AC_mx]*sizeof(AcReal), sx=host_mesh[AC_nx]*sizeof(AcReal);

    size_t start_idx;
    void *dest, *src;

    for (int iv = 0; iv < NUM_VTXBUF_HANDLES; ++iv) {
      for (int k=start.z; k<end.z; k++){

        start_idx = acVertexBufferIdx(start.x,start.y,k,host_mesh->info);
        dest=&(host_mesh->vertex_buffer[iv][start_idx]);
        src=&device->vba.on_device.out[iv][start_idx+src_offset];
        cudaMemcpy2DAsync(dest, px, src, px, sx, host_mesh->info[AC_ny],
                          cudaMemcpyDeviceToHost, device->streams[stream]);
      }
    }
    return AC_SUCCESS;
}
#endif

AcResult
acDeviceResetMesh(const Device device, const Stream stream)
{
    cudaSetDevice(device->id);
    acDeviceSynchronizeStream(device, stream);
    return acVBAReset(device->streams[stream], &device->vba);
}
acKernelInputParams*
acDeviceGetKernelInputParamsObject(const Device device)
{
	return &device->vba.on_device.kernel_input_params;
}

AcMeshInfo
acDeviceGetConfig(const Device device)
{
	return device->local_config;
}
AcDeviceKernelOutput
acDeviceGetKernelOutput(const Device device)
{
	return device->output;
}

#include "device_set_input.h"
#include "device_get_input.h"
#include "device_get_output.h"


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

        acReindex(device->streams[STREAM_DEFAULT], device->vba.on_device.in[basic_fields[w]], in_offset,
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
    //                   device->vba.on_device.profiles.in[0]);
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
    // cudaMemcpy(hostbuffer.data, device->vba.on_device.profiles.in[0],
    // sizeof(hostbuffer.data[0])*num_segments, cudaMemcpyDeviceToDevice);
    // // acBufferMigrate(buffer, &hostbuffer);
    // for (size_t w = 0; w < num_segments; ++w) {
    //     printf("start %zu: %g\n", w,
    //            hostbuffer.data[w * out_volume.x * out_volume.y * out_volume.z]);
    //     printf("end %zu: %g\n", w,
    //            hostbuffer.data[(w + 1) * out_volume.x * out_volume.y * out_volume.z - 1]);

    //     AcBuffer profiles = acBufferCreate(num_segments, false);
    //     cudaMemcpy(profiles.data, device->vba.on_device.profiles.in, sizeof(profiles.data[0]) *
    //     num_segments,
    //                cudaMemcpyDeviceToHost);
    //     printf("Profile %zu: %g\n", w, profiles.data[w]);
    //     acBufferDestroy(&profiles);
    // }
    // acBufferDestroy(&hostbuffer);

    acBufferDestroy(&buffer);
}
#endif

#ifdef AC_TFM_ENABLED
AcResult
acDeviceReduceXYAverages(const Device device, const Stream stream)
{
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
                      buffer.data, buffer_size, num_segments, device->vba.on_device.profiles.in[0]);

    // NOTE: Revisit this
    const size_t gnx = as_size_t(device->local_config.int3_params[AC_global_grid_n].x);
    const size_t gny = as_size_t(device->local_config.int3_params[AC_global_grid_n].y);
    cudaSetDevice(device->id);
    acMultiplyInplace(1. / (gnx * gny), num_compute_profiles * device->vba.profile_count,
                      device->vba.on_device.profiles.in[0]);

    acBufferDestroy(&buffer);
    return AC_SUCCESS;
}

#else
AcResult
acDeviceReduceXYAverages(const Device , const Stream)
{
        ERROR("acDeviceReduceXYAverages called but AC_TFM_ENABLED was false");
	return AC_FAILURE;
}

#endif
AcBuffer
acDeviceTransposeVertexBuffer(const Device device, const Stream stream, const AcMeshOrder order, const VertexBufferHandle vtxbuf)
{
	return acDeviceTransposeBase(device,stream,order,device->vba.on_device.in[vtxbuf]);
}
AcBuffer
acDeviceTransposeBase(const Device device, const Stream stream, const AcMeshOrder order, const AcReal* src)
{
    const AcMeshDims dims = acGetMeshDims(device->local_config);
    AcBuffer res = acBufferCreate(acGetTransposeBufferShape(order,dims.m1),true);
    acTranspose(order,src,res.data, dims.m1, device->streams[stream]);
    return res;
}
AcResult
acDeviceReduceAverages(const Device device, const Stream stream, const Profile prof)
{
    if constexpr (NUM_PROFILES == 0) return AC_FAILURE;
    return acReduceProfile(prof,
			   device->vba.profile_reduce_buffers[prof],
			   device->vba.on_device.profiles.in[prof],
			   device->streams[stream]
		    );
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
acDevicePreprocessScratchPad(Device device, const int variable, const AcType type,const AcReduceOp op)
{
	return acPreprocessScratchPad(device->vba,variable,type,op);
}

AcResult
acDeviceMemGetInfo(const Device device, size_t* free_mem, size_t* total_mem)
{
	ERRCHK_ALWAYS(device != NULL);
	cudaSetDevice(device->id);
	return cudaMemGetInfo(free_mem,total_mem) == cudaSuccess ? AC_SUCCESS : AC_FAILURE;
}

//TP: these are internal not user-facing device-layer functions
//These exists since other modules should not modify the device structure directly but do it through API functions
//Because they are internal it is okay for them not to return an error code: any errors are fatal!!
VertexBufferArray
acDeviceGetVBA(const Device device)
{
	return device->vba;
}

acKernelInputParams*
acDeviceGetKernelInputParams(const Device device)
{
	return &device->vba.on_device.kernel_input_params;
}

int 
acDeviceGetId(const Device device)
{
	return device->id;
}

AcReduceBuffer
acDeviceGetProfileReduceBuffer(const Device device, const Profile prof)
{
	if constexpr (NUM_PROFILES == 0)
		ERRCHK_ALWAYS(NUM_PROFILES > 0);
	return device->vba.profile_reduce_buffers[prof];
}

AcReal*
acDeviceGetProfileBuffer(const Device device, const Profile prof)
{
	if constexpr (NUM_PROFILES == 0)
		ERRCHK_ALWAYS(NUM_PROFILES > 0);
	return device->vba.on_device.profiles.in[prof];
}
AcReal**
acDeviceGetStartOfProfiles(const Device device)
{
	return device->vba.on_device.profiles.in;
}


#include "device_set_output.h"



