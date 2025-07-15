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

static rocfft_plan_description 
get_data_layout(const Volume domain_size)
{
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
    return desc;
}
static AcResult
acFFTTransformC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst, const bool inverse) {
    rocfft_plan_description desc = get_data_layout(domain_size);
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    // Create plan
    rocfft_plan plan = nullptr;
    size_t lengths[] = {subdomain_size.z,subdomain_size.y,subdomain_size.x};
    const auto rocfft_type = inverse ? rocfft_transform_type_complex_inverse : rocfft_transform_type_complex_forward;
    rocfft_status status = rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,
        rocfft_type,
	AC_FFT_PRECISION,
        3,            // Dimensions
        lengths,      // lengths
        1,            // batch
        desc);        // description
    if (status != rocfft_status_success) return AC_FAILURE;

    // Create execution info
    rocfft_execution_info info = nullptr;
    status = rocfft_execution_info_create(&info);
    if (status != rocfft_status_success) return AC_FAILURE;

    // Execute
    void* in_buffer[] = {const_cast<void*>(reinterpret_cast<const void*>(src+starting_offset))};
    void* out_buffer[] = {reinterpret_cast<void*>(dst+starting_offset)};
    status = rocfft_execute(plan, in_buffer, out_buffer, info);
    if (status != rocfft_status_success) return AC_FAILURE;

    // Cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    rocfft_plan_description_destroy(desc);

    // Scaling (just like CUFFT doesn't scale by default)
    size_t complex_domain_size = domain_size.x * domain_size.y * domain_size.z;
    const AcReal scale = 1.0 / (subdomain_size.x * subdomain_size.y * subdomain_size.z);
    if(!inverse) acMultiplyInplaceComplex(scale, complex_domain_size, dst);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst) {
	acFFTTransformC2C(src,domain_size,subdomain_size,staring_point,dst,false);
}

AcResult
acFFTBackwardTransformC2C(const AcComplex* src,
                                 const Volume domain_size,
                                 const Volume subdomain_size,
                                 const Volume starting_point,
                                 AcComplex* dst) {
    acFFTTransformC2C(src,domain_size,subdomain_size,staring_point,dst,true);
}

AcResult
acFFTForwardTransformSymmetricR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	return AC_FAILURE;
}

AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*) {
	return AC_FAILURE;
}

static AcComplex*
get_fresh_complex_buffer(const size_t count)
{
    const size_t bytes = sizeof(AcComplex)*count;
    AcComplex* res = NULL;
    acDeviceMalloc((void**)&res,bytes);
    acMultiplyInplaceComplex(AcReal(0.0),count,res);
    return res;
}

AcResult
acFFTBackwardTransformC2R(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer) {
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp = get_fresh_complex_buffer(count);
    acFFTBackwardTransformC2C(transformed_in,domain_size,subdomain_size,starting_point,tmp);
    acComplexToReal(tmp,count,buffer);
    acDeviceFree(&tmp,0);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformR2C(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst) {
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp = get_fresh_complex_buffer(count);
    acRealToComplex(src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,dst);
    acDeviceFree(&tmp,0);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformR2Planar(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_complex_buffer(count);

    acRealToComplex(src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp2,count,real_dst,imag_dst);

    acDeviceFree(&tmp,0);
    acDeviceFree(&tmp2,0);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    AcComplex* tmp  = get_fresh_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_complex_buffer(count);

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp,count,real_dst,imag_dst);

    acDeviceFree(&tmp,0);
    acDeviceFree(&tmp2,0);
    return AC_SUCCESS;
}
