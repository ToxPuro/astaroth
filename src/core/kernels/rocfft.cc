#include "host_datatypes.h"
#include "ac_fft.h"
#include "astaroth_cuda_wrappers.h"
#include "errchk.h"
#include "ac_helpers.h"
#include "common_kernels.h"
#include <stdio.h>
#include <cstdlib>

#if AC_DOUBLE_PRECISION
#define AC_FFT_PRECISION rocfft_precision_double
#else
#define AC_FFT_PRECISION rocfft_precision_single
#endif
#include <rocfft.h>

static  bool
operator==(const Volume& a, const Volume& b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

#include <unordered_map>
struct VolumeHash {
    std::size_t operator()(const Volume& v) const {
        return std::hash<size_t>()(v.x) ^ std::hash<size_t>()(v.y) << 1 ^ std::hash<size_t>()(v.z) << 2;
    }
};

std::unordered_map<Volume,rocfft_plan_description,VolumeHash> data_layouts{};
static rocfft_plan_description 
get_data_layout(const Volume domain_size)
{
    if(data_layouts.find(domain_size) != data_layouts.end())
    {
	    return data_layouts[domain_size];
    }
    //TP: not sure are the offsets for rocfft in bytes or in number of elements so prefer to do the offseting via pointer arithmetic myself
    size_t offsets[]  = {0,0,0};
    size_t strides[]  = {domain_size.x*domain_size.y,domain_size.x,1};
    size_t distance = domain_size.x*domain_size.y*domain_size.z;
    // Create plan description
    rocfft_plan_description desc = nullptr;
    rocfft_status status = rocfft_plan_description_create(&desc);
    ERRCHK_ALWAYS((status == rocfft_status_success));
    status = rocfft_plan_description_set_data_layout(
        desc,
        rocfft_array_type_complex_interleaved,  // in_array_type
        rocfft_array_type_complex_interleaved,  // out_array_type
        offsets,
        offsets,
        3,
        strides,
        distance,
        3,
        strides,
        distance
        );

    ERRCHK_ALWAYS((status == rocfft_status_success));
    data_layouts[domain_size] = desc;
    return desc;
}

static rocfft_execution_info
get_execution_info()
{
	static rocfft_execution_info info{};
	static bool first_call = true;
	if(first_call)
	{
    	        // Create execution info
    	        info = nullptr;
    	        ERRCHK_ALWAYS(rocfft_execution_info_create(&info) == rocfft_status_success)
		first_call = false;
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
    const rocfft_plan_description desc = get_data_layout(domain_size);
    size_t lengths[] = {subdomain_size.z,subdomain_size.y,subdomain_size.x};
    const auto rocfft_type = inverse ? rocfft_transform_type_complex_inverse : rocfft_transform_type_complex_forward;
    ERRCHK_ALWAYS(rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,
        rocfft_type,
	AC_FFT_PRECISION,
        3,            // Dimensions
        lengths,      // lengths
        1,            // batch
        desc)        // description
	== rocfft_status_success);
    rocfft_plans[key] = plan;
    return plan;
}
static AcResult
acFFTTransformC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst, const bool inverse) {
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    const auto plan = get_plan(domain_size,subdomain_size,inverse);
    // Execute
    void* in_buffer[] = {const_cast<void*>(reinterpret_cast<const void*>(src+starting_offset))};
    void* out_buffer[] = {reinterpret_cast<void*>(dst+starting_offset)};
    ERRCHK_ALWAYS(rocfft_execute(plan, in_buffer, out_buffer, get_execution_info()) == rocfft_status_success);
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

std::unordered_map<size_t,AcComplex*> tmp_buffers{};
static AcComplex*
get_fresh_tmp_complex_buffer(const size_t count)
{
    if(tmp_buffers.find(count) != tmp_buffers.end())
    {
	return tmp_buffers[count];
    }
    const size_t bytes = sizeof(AcComplex)*count;
    AcComplex* res = NULL;
    acDeviceMalloc((void**)&res,bytes);
    acMultiplyInplaceComplex(AcReal(0.0),count,res);
    tmp_buffers[count] = res;
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
    AcComplex* tmp2 = get_fresh_tmp_complex_buffer(count);

    acRealToComplex(src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp2,count,real_dst,imag_dst);

    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_tmp_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_tmp_complex_buffer(count);

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp,count,real_dst,imag_dst);

    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_complex_buffer(count);

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTBackwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp,count,real_dst,imag_dst);

    acDeviceFree(&tmp,0);
    acDeviceFree(&tmp2,0);
    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformPlanar2R(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_complex_buffer(count);

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTBackwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToReal(tmp,count,dst);

    acDeviceFree(&tmp,0);
    acDeviceFree(&tmp2,0);
    return AC_SUCCESS;
