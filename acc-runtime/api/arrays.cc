#include "acc_runtime.h"
#include "astaroth_cuda_wrappers.h"
#include "memcpy_to_gmem_arrays_header.h"
#include "memcpy_from_gmem_arrays_header.h"

template <typename P>
struct allocate_arrays
{
	void operator()(const AcMeshInfo& config) 
	{
		for(P array : get_params<P>())
		{
			if(config[array] == nullptr && is_accessed(array))
			{
				fprintf(stderr,"Passed %s as NULL but it is accessed kernels!!\n",get_name(array));
				fflush(stderr);
				ERRCHK_ALWAYS(config[array] != nullptr);
			}
			if (config[array] != nullptr && !is_dconst(array) && is_alive(array))
			{

#if AC_VERBOSE
				fprintf(stderr,"Allocating %s|%zu\n",get_name(array),get_array_length(array,config));
				fflush(stderr);
#endif
				auto d_mem_ptr = get_empty_pointer(array);
			        acDeviceMalloc(((void**)&d_mem_ptr), sizeof(config[array][0])*get_array_length(array,config));
				acMemcpyToGmemArray(array,d_mem_ptr);
			}
		}
	}
};

AcResult
acAllocateArrays(const AcMeshInfo config)
{
  AcArrayTypes::run<allocate_arrays>(config);
  return AC_SUCCESS;
}

template <typename P>
struct update_arrays
{
	void operator()(const AcMeshInfo& config)
	{
		for(P array : get_params<P>())
		{
			if (is_dconst(array) || !is_alive(array)) continue;
			auto config_array = config[array];
			auto gmem_array   = get_empty_pointer(array);
			acMemcpyFromGmemArray(array,gmem_array);
			size_t bytes = sizeof(config_array[0])*get_array_length(array,config);
			if (config_array == nullptr && gmem_array != nullptr) 
				acDeviceFree((void**)&gmem_array,bytes);
			else if (config_array != nullptr && gmem_array  == nullptr) 
				acDeviceMalloc((void**)&gmem_array,bytes);
			acMemcpyToGmemArray(array,gmem_array);
		}
	}
};

AcResult
acUpdateArrays(const AcMeshInfo config)
{
  AcArrayTypes::run<update_arrays>(config);
  return AC_SUCCESS;
}

template <typename P>
struct free_arrays
{
	void operator()(const AcMeshInfo& config)
	{
		for(P array: get_params<P>())
		{
			auto config_array = config[array];
			if (config_array == nullptr || is_dconst(array) || !is_alive(array)) continue;
			auto gmem_array = get_empty_pointer(array);
			acMemcpyFromGmemArray(array,gmem_array);
			acDeviceFree((void**)&gmem_array, get_array_length(array,config));
			acMemcpyToGmemArray(array,gmem_array);
		}
	}
};

AcResult
acFreeArrays(const AcMeshInfo config)
{
  AcArrayTypes::run<free_arrays>(config);
  return AC_SUCCESS;
}

