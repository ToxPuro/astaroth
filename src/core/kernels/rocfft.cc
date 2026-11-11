#include <rocfft/rocfft.h>
#include <stdio.h>

#include <array>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ac_fft.h"
#include "ac_helpers.h"
#include "ac_mpi.h"
#include "astaroth_cuda_wrappers.h"
#include "common_kernels.h"
#include "errchk.h"
#include "host_datatypes.h"
#include "distributed_fft_device.h"

#if AC_DOUBLE_PRECISION
#define AC_FFT_PRECISION rocfft_precision_double
#else
#define AC_FFT_PRECISION rocfft_precision_single
#endif

static  bool
operator==(const Volume& a, const Volume& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

struct VolumeHash {
    std::size_t operator()(const Volume& v) const {
        return std::hash<size_t>()(v.x) ^ std::hash<size_t>()(v.y) << 1 ^ std::hash<size_t>()(v.z) << 2;
    }
};

struct VolumePairHash {
    std::size_t operator()(const std::pair<Volume,Volume>& v) const {
        return VolumeHash()(v.first) ^ VolumeHash()(v.second) << 3;
    }
};

#define check_rocfft_status(expression) { const auto _status_ = expression; print_rocfft_error(_status_); ERRCHK_ALWAYS(_status_ == rocfft_status_success); }

void
print_rocfft_error(rocfft_status status)
{
    if(status != rocfft_status_success)
    {
            const char* msg = NULL;
            if(status == rocfft_status_failure)
            {
                    msg ="Failure!\n";
            }
            if(status == rocfft_status_invalid_arg_value)
            {
                    msg ="Invalid arg value!\n";
            }
            if(status == rocfft_status_invalid_dimensions)
            {
                    msg ="Invalid dimensions!\n";
            }
            if(status == rocfft_status_invalid_array_type)
            {
                    msg ="Invalid array type!\n";
            }
            if(status == rocfft_status_invalid_strides)
            {
                    msg ="Invalid strides!\n";
            }
            if(status == rocfft_status_invalid_distance)
            {
                    msg ="Invalid distance!\n";
            }
            if(status == rocfft_status_invalid_offset)
            {
                    msg ="Invalid offset!\n";
            }
            if(status == rocfft_status_invalid_work_buffer)
            {
                    msg ="Invalid work buffer!\n";
            }
            fprintf(stderr,"Rocfft Error: %s\n",msg);
            fflush(stderr);
    }
}
//TP: Multi-GPU support for the C2C transforms, see distributed_fft.h and distributed_fft_device.h.
//The rocFFT plans of the batched 1D transforms along each axis (forward and inverse) and the device buffers for the pencils of a layout.
class RocFFTPencilPlans
{
public:
	RocFFTPencilPlans(const DistributedLayout& layout)
	{
		size_t max_count = 0;
		size_t work_buf_size = 0;
		for(int axis = 0; axis < 3; ++axis)
		{
			const Box& box = layout.pencil_boxes[axis][layout.rank];
			max_count = std::max(max_count,box_count(box));
			if(layout.global_dims[axis] == 1 || box_count(box) == 0) continue;
			batch_layouts[axis] = get_pencil_batch_layout(box,axis);
			const PencilBatchLayout& bl = batch_layouts[axis];
			size_t stride[] = {bl.stride};
			rocfft_plan_description desc = nullptr;
			check_rocfft_status(rocfft_plan_description_create(&desc));
			check_rocfft_status(rocfft_plan_description_set_data_layout(
				desc,
				rocfft_array_type_complex_interleaved,
				rocfft_array_type_complex_interleaved,
				nullptr,
				nullptr,
				1,
				stride,
				bl.dist,
				1,
				stride,
				bl.dist
				));
			size_t lengths[] = {bl.n};
			for(int inverse = 0; inverse < 2; ++inverse)
			{
				const auto rocfft_type = inverse ? rocfft_transform_type_complex_inverse : rocfft_transform_type_complex_forward;
				check_rocfft_status(rocfft_plan_create(
					&plans[axis][inverse],
					rocfft_placement_inplace,
					rocfft_type,
					AC_FFT_PRECISION,
					1,
					lengths,
					bl.batch,
					desc));
				size_t size = 0;
				check_rocfft_status(rocfft_plan_get_work_buffer_size(plans[axis][inverse],&size));
				work_buf_size = std::max(work_buf_size,size);
			}
			check_rocfft_status(rocfft_plan_description_destroy(desc));
		}
		if(max_count) acDeviceMalloc((void**)&d_buf,sizeof(AcComplex)*max_count);
		check_rocfft_status(rocfft_execution_info_create(&exec_info));
		if(work_buf_size)
		{
			ERRCHK_CUDA_ALWAYS(acMalloc(&work_buf,work_buf_size));
			check_rocfft_status(rocfft_execution_info_set_work_buffer(exec_info,work_buf,work_buf_size));
		}
	}
	~RocFFTPencilPlans()
	{
		for(int axis = 0; axis < 3; ++axis)
			for(int inverse = 0; inverse < 2; ++inverse)
				if(plans[axis][inverse]) rocfft_plan_destroy(plans[axis][inverse]);
		if(exec_info) rocfft_execution_info_destroy(exec_info);
		if(work_buf) acFree(work_buf);
		if(d_buf) acDeviceFree(&d_buf,0);
	}
	RocFFTPencilPlans(const RocFFTPencilPlans&) = delete;
	RocFFTPencilPlans& operator=(const RocFFTPencilPlans&) = delete;

	void
	transform_axis(std::vector<AcComplex>& data, const int axis, const bool inverse)
	{
		const rocfft_plan plan = plans[axis][inverse];
		if(!plan) return;
		const size_t bytes = sizeof(AcComplex)*data.size();
		ERRCHK_CUDA_ALWAYS(acMemcpy(d_buf,data.data(),bytes,cudaMemcpyHostToDevice));
		for(size_t i = 0; i < batch_layouts[axis].loops; ++i)
		{
			void* buffer[] = {reinterpret_cast<void*>(d_buf + i*batch_layouts[axis].loop_offset)};
			check_rocfft_status(rocfft_execute(plan, buffer, nullptr, exec_info));
		}
		ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
		ERRCHK_CUDA_ALWAYS(acMemcpy(data.data(),d_buf,bytes,cudaMemcpyDeviceToHost));
	}
private:
	std::array<std::array<rocfft_plan,2>,3> plans{};
	std::array<PencilBatchLayout,3> batch_layouts{};
	rocfft_execution_info exec_info = nullptr;
	void* work_buf = nullptr;
	AcComplex* d_buf = NULL;
};

static DeviceDistributedFFT<RocFFTPencilPlans> distributed_fft("RocFFT");

rocfft_field
get_rocfft_field(const Volume domain_size, const Volume subdomain_size)
{
    rocfft_field field_layout = nullptr;
    check_rocfft_status(rocfft_field_create(&field_layout));

    //Only used for single-process transforms (multi-process ones go through distributed_fft),
    //so the single brick spans the whole transform
    size_t lower[3] = {0,0,0};
    size_t dims[3] = {
            subdomain_size.x,
            subdomain_size.y,
            subdomain_size.z,
    };

    size_t upper[3] = {
	    lower[0] + dims[0],
	    lower[1] + dims[1],
	    lower[2] + dims[2]
    };

    size_t stride[3] = {
            1,
            (domain_size.x),
            (domain_size.x)*(domain_size.y)
    };

    rocfft_brick brick_layout = nullptr;
    int device_id = -1;
    ERRCHK_CUDA(acGetDevice(&device_id));
    check_rocfft_status(rocfft_brick_create(
                      &brick_layout,
                      lower,
                      upper,
                      stride,
                      3,
                      device_id
                    ));

    check_rocfft_status(rocfft_field_add_brick(field_layout,brick_layout));
    return field_layout;
}


//The strides depend on domain_size, so cache by both domain_size and subdomain_size
std::unordered_map<std::pair<Volume,Volume>,rocfft_plan_description,VolumePairHash> data_layouts{};
static rocfft_plan_description 
get_data_layout(const Volume domain_size, const Volume subdomain_size)
{
    const auto key = std::make_pair(domain_size,subdomain_size);
    if(data_layouts.find(key) != data_layouts.end())
    {
	    return data_layouts[key];
    }
    // Create plan description
    rocfft_plan_description desc = nullptr;
    check_rocfft_status(rocfft_plan_description_create(&desc));
    //Most variables are not set when using Field-API
    check_rocfft_status(rocfft_plan_description_set_data_layout(
        desc,
        rocfft_array_type_complex_interleaved,  // in_array_type
        rocfft_array_type_complex_interleaved,  // out_array_type
        nullptr,
        nullptr,
        0,
        nullptr,
        0,
        0,
        nullptr,
        0
        ));

    check_rocfft_status(rocfft_plan_description_add_infield(desc,get_rocfft_field(domain_size,subdomain_size)));
    check_rocfft_status(rocfft_plan_description_add_outfield(desc,get_rocfft_field(domain_size,subdomain_size)));

#if AC_MPI_ENABLED
    //TP: has to be on comment for now since 1.) doesn't work if RocFFT has not been MPI-enabled
    //                                       2.) requires at least rocm/6.3.0
    //rocfft_plan_description_set_comm(desc,rocfft_comm_mpi,&communicator);
#endif

    data_layouts[key] = desc;
    return desc;
}

static rocfft_execution_info
get_execution_info(const Volume subdomain, const rocfft_plan plan)
{
	static std::unordered_map<size_t,void*> work_buffers{};
    	rocfft_execution_info info = nullptr;
    	check_rocfft_status(rocfft_execution_info_create(&info));
	size_t work_buf_size = 0;
	check_rocfft_status(rocfft_plan_get_work_buffer_size(plan,&work_buf_size));
	if(work_buf_size)
	{
		void* work_buf = nullptr;
		if(work_buffers.find(work_buf_size) != work_buffers.end())
		{
			work_buf = work_buffers[work_buf_size];
		}
		else
		{
			ERRCHK_CUDA_ALWAYS(acMalloc(&work_buf,work_buf_size));
			work_buffers[work_buf_size] = work_buf;
		}
		check_rocfft_status(rocfft_execution_info_set_work_buffer(info,work_buf,work_buf_size));
	}
	return info;
}

// Combine hash helper function
template <typename T>
void hash_combine(std::size_t &seed, const T& value) {
    seed ^= std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

using KeyType = std::tuple<Volume,Volume,bool>;
struct KeyHash {
    std::size_t operator()(const KeyType &key) const {
        const auto &[domain_size,subdomain_size,inverse] = key;
        std::size_t seed = 0;
        hash_combine(seed, domain_size.x);
        hash_combine(seed, domain_size.y);
        hash_combine(seed, domain_size.z);
        hash_combine(seed, subdomain_size.x);
        hash_combine(seed, subdomain_size.y);
        hash_combine(seed, subdomain_size.z);
        hash_combine(seed, inverse);
        return seed;
    }
};
struct KeyEqual {
    bool operator()(const KeyType &lhs,
                    const KeyType &rhs) const {
        return lhs == rhs; 
    }
};
static std::unordered_map<KeyType, rocfft_plan, KeyHash, KeyEqual> rocfft_plans{};

static rocfft_plan
get_plan(const Volume domain_size, const Volume subdomain_size, const bool inverse)
{
    const KeyType key = (KeyType){domain_size,subdomain_size,inverse};
    if(rocfft_plans.find(key) != rocfft_plans.end())
    {
	    return rocfft_plans[key];
    }
    // Create plan
    rocfft_plan plan = nullptr;
    const rocfft_plan_description desc = get_data_layout(domain_size,subdomain_size);
    size_t lengths[] = {subdomain_size.x,subdomain_size.y,subdomain_size.z};
    const auto rocfft_type = inverse ? rocfft_transform_type_complex_inverse : rocfft_transform_type_complex_forward;
    check_rocfft_status(rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,
        rocfft_type,
	AC_FFT_PRECISION,
        3,            // Dimensions
        lengths,      // lengths
        1,            // batch
        desc)        // description
	);
    rocfft_plans[key] = plan;
    return plan;
}

static AcResult
acFFTTransformC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst, const bool inverse) {
    if(distributed_fft.is_distributed(subdomain_size))
	    return distributed_fft.transform_c2c(src,domain_size,subdomain_size,starting_point,dst,inverse);
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    const auto plan = get_plan(domain_size,subdomain_size,inverse);
    const auto exec_info = get_execution_info(subdomain_size,plan);
    // Execute
    void* in_buffer[] = {const_cast<void*>(reinterpret_cast<const void*>(src+starting_offset))};
    void* out_buffer[] = {reinterpret_cast<void*>(dst+starting_offset)};
    check_rocfft_status(rocfft_execute(plan, in_buffer, out_buffer,exec_info));
    // Scaling (just like CUFFT doesn't scale by default)
    size_t complex_domain_size = domain_size.x * domain_size.y * domain_size.z;
    const AcReal scale = AcReal(1.0) / (subdomain_size.x * subdomain_size.y * subdomain_size.z);
    if(!inverse) acMultiplyInplaceComplex(scale, complex_domain_size, dst);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst) {
	return acFFTTransformC2C(src,domain_size,subdomain_size,starting_point,dst,false);
}

AcResult
acFFTBackwardTransformC2C(const AcComplex* src,
                                 const Volume domain_size,
                                 const Volume subdomain_size,
                                 const Volume starting_point,
                                 AcComplex* dst) {
    return acFFTTransformC2C(src,domain_size,subdomain_size,starting_point,dst,true);
}

AcResult
acFFTForwardTransformSymmetricR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	return AC_FAILURE;
}

AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*) {
	return AC_FAILURE;
}

//Cached by the element count and a slot so that the temporary buffers used together in a function
//(slots 0 and 1) are distinct: the transforms are out-of-place and require the input and output not to alias
std::map<std::pair<size_t,int>,AcComplex*> tmp_buffers{};
static AcComplex*
get_fresh_tmp_complex_buffer(const size_t count, const int slot = 0)
{
    const auto key = std::make_pair(count,slot);
    if(tmp_buffers.find(key) != tmp_buffers.end())
    {
	return tmp_buffers[key];
    }
    const size_t bytes = sizeof(AcComplex)*count;
    AcComplex* res = NULL;
    acDeviceMalloc((void**)&res,bytes);
    acMultiplyInplaceComplex(AcReal(0.0),count,res);
    tmp_buffers[key] = res;
    return res;
}

AcResult
acFFTBackwardTransformC2R(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer) {
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp = get_fresh_tmp_complex_buffer(count);
    acFFTBackwardTransformC2C(transformed_in,domain_size,subdomain_size,starting_point,tmp);
    acComplexToReal(tmp,count,buffer);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformR2C(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst) {
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp = get_fresh_tmp_complex_buffer(count);
    acRealToComplex(src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,dst);
    return AC_SUCCESS;
}


AcResult
acFFTForwardTransformR2Planar(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_tmp_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_tmp_complex_buffer(count,1);

    acRealToComplex(src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acKernelVolumeCopyComplexToPlanar(0,tmp2
		    			,starting_point,subdomain_size,domain_size
					,real_dst,imag_dst
		    			,starting_point,subdomain_size,domain_size
		    );
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformR2HermitianPlanarBatched(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst, const int batch_size, cudaStream_t stream)
{
	return AC_FAILURE;
}

AcResult
acFFTForwardTransformR2PlanarBatched(const void* src_, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, void* real_dst_, void* imag_dst_, const int batch_size, const AcPrecision input_precision, const AcPrecision output_precision)
{
    
    if(input_precision != AC_REAL_PRECISION || output_precision != AC_REAL_PRECISION) return AC_FAILURE;

    const AcReal* src      = (AcReal*)src_;
    AcReal* real_dst = (AcReal*)real_dst_;
    AcReal* imag_dst = (AcReal*)imag_dst_;
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    for(int offset = 0; offset < batch_size; ++offset)
    {
	if(acFFTForwardTransformR2Planar(
					src + offset*count,
					domain_size,
					subdomain_size,
					starting_point,
					real_dst + offset*count,
					imag_dst + offset*count
				) == AC_FAILURE)
		return AC_FAILURE;
    }
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_tmp_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_tmp_complex_buffer(count,1);

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp2,count,real_dst,imag_dst);

    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_tmp_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_tmp_complex_buffer(count,1);

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTBackwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp2,count,real_dst,imag_dst);

    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformPlanar2R(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_tmp_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_tmp_complex_buffer(count,1);

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTBackwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acKernelVolumeCopyComplexToReal(0,tmp2,starting_point,subdomain_size,domain_size,dst,starting_point,subdomain_size,domain_size);

    return AC_SUCCESS;
}


AcResult
acFFTInit(const AcCommunicator* astaroth_comm, const int* global_offset)
{
	check_rocfft_status(rocfft_setup());
	distributed_fft.init(astaroth_comm,global_offset);
	return AC_SUCCESS;
}

AcResult
acFFTQuit()
{
	distributed_fft.quit();
	return AC_SUCCESS;
}
