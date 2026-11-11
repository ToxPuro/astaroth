#pragma once
#include <map>
#include <memory>
#include <vector>
#include "astaroth_cuda_wrappers.h"
#include "ac_helpers.h"
#include "distributed_fft.h"

//TP: Common machinery for the distributed FFT backends which are built on top of a single-GPU FFT library (cufft.cc, rocfft.cc).
//The subdomain is first copied to the host. Before transforming along an axis the data is redistributed on the host
//to pencils which hold that whole axis, the pencil is copied to the device, transformed with batched 1D transforms
//and copied back to the host. At the end the result is copied back to the subdomain on the device.
//
//The backend provides PencilPlans, which holds the 1D plans of a layout and has to provide:
//	PencilPlans(const DistributedLayout& layout)
//	void transform_axis(std::vector<AcComplex>& data, const int axis, const bool inverse)
//	which does the 1D transforms along axis for data laid out as the pencil of this process for axis.

//Layout of the batched 1D transforms along axis for data laid out as box:
//loops times a batch of 1D transforms of length n, where element i of transform b in loop l is at
//l*loop_offset + b*dist + i*stride.
//A single batch dimension suffices for x and z. For y the batches span both x and z, so loop over the z-slices
typedef struct
{
	size_t n;
	size_t stride;
	size_t dist;
	size_t batch;
	size_t loops;
	size_t loop_offset;
} PencilBatchLayout;

static inline PencilBatchLayout
get_pencil_batch_layout(const Box& box, const int axis)
{
	const size_t nx = box.dims[0];
	const size_t ny = box.dims[1];
	const size_t nz = box.dims[2];
	switch(axis)
	{
		case 0:
			return (PencilBatchLayout){nx,1,nx,ny*nz,1,0};
		case 1:
			return (PencilBatchLayout){ny,nx,1,nx,nz,nx*ny};
		default:
			return (PencilBatchLayout){nz,nx*ny,1,nx*ny,1,0};
	}
}

//Copies the subdomain at starting_point of the device buffer src (of size domain_size) to the contiguous host buffer dst
static inline void
copy_subdomain_to_host(const AcComplex* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst)
{
	ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
	for(size_t k = 0; k < subdomain_size.z; ++k)
	{
		const size_t src_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*(starting_point.z+k));
		ERRCHK_CUDA_ALWAYS(cudaMemcpy2D(dst + k*subdomain_size.x*subdomain_size.y, sizeof(AcComplex)*subdomain_size.x,
					    src + src_offset, sizeof(AcComplex)*domain_size.x,
					    sizeof(AcComplex)*subdomain_size.x, subdomain_size.y,
					    cudaMemcpyDeviceToHost));
	}
}

//Copies the contiguous host buffer src to the subdomain at starting_point of the device buffer dst (of size domain_size)
static inline void
copy_subdomain_to_device(const AcComplex* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst)
{
	for(size_t k = 0; k < subdomain_size.z; ++k)
	{
		const size_t dst_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*(starting_point.z+k));
		ERRCHK_CUDA_ALWAYS(cudaMemcpy2D(dst + dst_offset, sizeof(AcComplex)*domain_size.x,
					    src + k*subdomain_size.x*subdomain_size.y, sizeof(AcComplex)*subdomain_size.x,
					    sizeof(AcComplex)*subdomain_size.x, subdomain_size.y,
					    cudaMemcpyHostToDevice));
	}
	ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
}

template <typename PencilPlans>
class DeviceDistributedFFT : public DistributedFFT<AcComplex>
{
public:
	DeviceDistributedFFT(const char* name_) : DistributedFFT<AcComplex>(name_) {}

	void
	init(const AcCommunicator* astaroth_comm, const int* global_offset_)
	{
		pencil_plans.clear();
		DistributedFFT<AcComplex>::init(astaroth_comm,global_offset_);
	}

	void
	quit()
	{
		pencil_plans.clear();
		DistributedFFT<AcComplex>::quit();
	}

	//Whether the transform of subdomain_size spans multiple processes and has to go through transform_c2c
	bool
	is_distributed(const Volume subdomain_size)
	{
		return get_layout(subdomain_size).nprocs > 1;
	}

	//Transforms the subdomain at starting_point of the device buffer src and writes the result to the same subdomain of dst
	AcResult
	transform_c2c(const AcComplex* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst,
		      const bool inverse)
	{
		const DistributedLayout& layout = get_layout(subdomain_size);
		PencilPlans& plans = get_pencil_plans(subdomain_size,layout);
		std::vector<AcComplex> data(subdomain_size.x*subdomain_size.y*subdomain_size.z);
		copy_subdomain_to_host(src,domain_size,subdomain_size,starting_point,data.data());
		transform(layout,data,inverse,
			[&](std::vector<AcComplex>& pencil, const Box&, const int axis){plans.transform_axis(pencil,axis,inverse);},
			[](std::vector<AcComplex>& elems, const AcReal scale)
			{
				for(auto& elem : elems)
				{
					elem.x *= scale;
					elem.y *= scale;
				}
			});
		copy_subdomain_to_device(data.data(),domain_size,subdomain_size,starting_point,dst);
		return AC_SUCCESS;
	}

private:
	PencilPlans&
	get_pencil_plans(const Volume subdomain_size, const DistributedLayout& layout)
	{
		const LayoutKey key = get_layout_key(subdomain_size);
		auto it = pencil_plans.find(key);
		if(it == pencil_plans.end())
			it = pencil_plans.emplace(key,std::make_unique<PencilPlans>(layout)).first;
		return *it->second;
	}

	//The plans and device buffers of the 1D transforms, cached by the shape of the local subdomain like the layouts
	std::map<LayoutKey,std::unique_ptr<PencilPlans>> pencil_plans{};
};
