#include "acc_runtime.h"
#include "kernels.h"
#if AC_FFT_ENABLED
#if AC_USE_HIP
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

AcResult
acFFTForwardTransformC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst) {
    rocfft_plan_description desc = get_data_layout(domain_size);
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    // Create plan
    rocfft_plan plan = nullptr;
    size_t lengths[] = {subdomain_size.z,subdomain_size.y,subdomain_size.x};
    rocfft_status status = rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,
        rocfft_transform_type_complex_forward,
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
    acMultiplyInplaceComplex(scale, complex_domain_size, dst);

    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformC2C(const AcComplex* src,
                                 const Volume domain_size,
                                 const Volume subdomain_size,
                                 const Volume starting_point,
                                 AcComplex* dst) {
    // Create plan description
    rocfft_plan_description desc = get_data_layout(domain_size);
    // Create inverse plan
    rocfft_plan plan = nullptr;
    size_t lengths[] = {subdomain_size.z,subdomain_size.y,subdomain_size.x};
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    rocfft_status status = rocfft_plan_create(
        &plan,
        rocfft_placement_notinplace,
        rocfft_transform_type_complex_inverse,
	AC_FFT_PRECISION,
        3,           // Dimensions
        lengths,     // FFT size
        1,           // Batch size
        desc);
    if (status != rocfft_status_success) return AC_FAILURE;

    // Create execution info
    rocfft_execution_info info = nullptr;
    status = rocfft_execution_info_create(&info);
    if (status != rocfft_status_success) return AC_FAILURE;

    void* in_buffer[] = {const_cast<void*>(reinterpret_cast<const void*>(src+starting_offset))};
    void* out_buffer[] = {reinterpret_cast<void*>(dst+starting_offset)};

    status = rocfft_execute(plan, in_buffer, out_buffer, info);
    if (status != rocfft_status_success) return AC_FAILURE;

    // Cleanup
    rocfft_execution_info_destroy(info);
    rocfft_plan_destroy(plan);
    rocfft_plan_description_destroy(desc);

    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformSymmetricR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	return AC_FAILURE;
}

AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*) {
	return AC_FAILURE;
}

#else
#include <cufftXt.h>
#include <cuComplex.h>

// cufft API error chekcing
#ifndef CUFFT_CALL
#define CUFFT_CALL( call )                                                                                             \
    {                                                                                                                  \
        auto status = static_cast<cufftResult>( call );                                                                \
        if ( status != CUFFT_SUCCESS )                                                                                 \
	    {                                                                                                          \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUFFT call \"%s\" in line %d of file %s failed "                                          \
                     "with "                                                                                           \
                     "code (%d).\n",                                                                                   \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     status );                                                                                         \
		abort();                                                                                               \
	    }                                                                                                          \
    }
#endif  // CUFFT_CALL
// TODO: if the buffer on GPU would be properly padded:
// https://docs.nvidia.com/cuda/cufft/index.html#data-layout
// we could use in-place transformation and save one buffer allocation
// Padding as mentioned in the link: padded to (n/2 + 1) in the least significant dimension.
AcResult
acFFTForwardTransformSymmetricR2C(const AcReal* buffer, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* transformed_in) {
    buffer = buffer + (starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z));
    // Number of elements in each dimension to use
    int dims[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)subdomain_size.x};
    // NOTE: inembed[0] and onembed[0] are not used directly, but as idist and odist
    // Sizes of input dimension of the buffer used
    int inembed[] = {(int)domain_size.z, (int)domain_size.y, (int)domain_size.x};
    // Sizes of the output dimension of the buffer used
    int onembed[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)(subdomain_size.x / 2) + 1};
    
    cufftHandle plan_r2c{};
    CUFFT_CALL(cufftCreate(&plan_r2c));
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlanMany(plan_r2c, 3, dims,
        inembed, 1, inembed[0], // in case inembed and onembed not needed could be: nullptr, 1, 0
        onembed, 1, onembed[0], //                                                  nullptr, 1, 0
        CUFFT_D2Z, 1, &workspace_size));
    
    size_t orig_domain_size = inembed[0] * inembed[1] * inembed[2];
    size_t complex_domain_size = onembed[0] * onembed[1] * onembed[2];    
    
    cuDoubleComplex* transformed = reinterpret_cast<cuDoubleComplex*>(transformed_in);
    // Execute the plan_r2c
    CUFFT_CALL(cufftXtExec(plan_r2c, (void*)buffer, transformed, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan_r2c));
    // Scale complex results that inverse FFT results in original values
    const AcReal scale{1.0 / orig_domain_size};
    acMultiplyInplaceComplex(scale, complex_domain_size, transformed_in);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformC2C(const AcComplex* buffer, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* transformed_in) {
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    buffer = buffer + starting_offset;
    // Number of elements in each dimension to use
    int dims[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)subdomain_size.x};
    // NOTE: inembed[0] and onembed[0] are not used directly, but as idist and odist
    // Sizes of input dimension of the buffer used
    int inembed[] = {(int)domain_size.z, (int)domain_size.y, (int)domain_size.x};
    // Sizes of the output dimension of the buffer used
    int onembed[] = {(int)domain_size.z, (int)domain_size.y, (int)(domain_size.x)};
    
    cufftHandle plan_r2c{};
    CUFFT_CALL(cufftCreate(&plan_r2c));
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlanMany(plan_r2c, 3, dims,
        inembed, 1, inembed[0], // in case inembed and onembed not needed could be: nullptr, 1, 0
        onembed, 1, onembed[0], //                                                  nullptr, 1, 0
        CUFFT_Z2Z, 1, &workspace_size));
    
    size_t complex_domain_size = onembed[0] * onembed[1] * onembed[2];    
    
    cuDoubleComplex* transformed = reinterpret_cast<cuDoubleComplex*>(transformed_in + starting_offset);
    // Execute the plan_r2c
    CUFFT_CALL(cufftXtExec(plan_r2c, (void*)buffer, transformed, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan_r2c));
    // Scale complex results that inverse FFT results in original values
    const AcReal scale{1.0 / ( dims[0] * dims[1] * dims[2])};
    acMultiplyInplaceComplex(scale, complex_domain_size, transformed_in);
    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer) {
    buffer = buffer + (starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z));
    // Number of elements in each dimension to use
    int dims[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)subdomain_size.x};
    // NOTE: inembed[0] and onembed[0] are not used directly, but as idist and odist
    // Sizes of input dimension of the buffer used
    int inembed[] = {(int)domain_size.z, (int)domain_size.y, (int)domain_size.x};
    // Sizes of the output dimension of the buffer used
    int onembed[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)(((int)subdomain_size.x) / 2) + 1};
    
    cufftHandle plan_c2r{};
    CUFFT_CALL(cufftCreate(&plan_c2r));
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlanMany(plan_c2r, 3, dims,
        onembed, 1, onembed[0],
        inembed, 1, inembed[0],
        CUFFT_Z2D, 1, &workspace_size));
    const cuDoubleComplex* transformed = reinterpret_cast<const cuDoubleComplex*>(transformed_in);
    CUFFT_CALL(cufftXtExec(plan_c2r, (void*)transformed, buffer, CUFFT_INVERSE));
    CUFFT_CALL(cufftDestroy(plan_c2r));
    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformC2C(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcComplex* buffer) {
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    buffer = buffer + starting_offset;
    // Number of elements in each dimension to use
    int dims[] = {(int)subdomain_size.z, (int)subdomain_size.y, (int)subdomain_size.x};
    // NOTE: inembed[0] and onembed[0] are not used directly, but as idist and odist
    // Sizes of input dimension of the buffer used
    int inembed[] = {(int)domain_size.z, (int)domain_size.y, (int)domain_size.x};
    // Sizes of the output dimension of the buffer used
    int onembed[] = {(int)domain_size.z, (int)domain_size.y, (int)(((int)domain_size.x))};
    
    cufftHandle plan_c2r{};
    CUFFT_CALL(cufftCreate(&plan_c2r));
    size_t workspace_size;
    CUFFT_CALL(cufftMakePlanMany(plan_c2r, 3, dims,
        onembed, 1, onembed[0],
        inembed, 1, inembed[0],
        CUFFT_Z2Z, 1, &workspace_size));
    const cuDoubleComplex* transformed = reinterpret_cast<const cuDoubleComplex*>(transformed_in + starting_offset);
    CUFFT_CALL(cufftXtExec(plan_c2r, (void*)transformed, buffer, CUFFT_INVERSE));
    CUFFT_CALL(cufftDestroy(plan_c2r));
    return AC_SUCCESS;
}
#endif //AC_USE_HIP
       //
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

#else
AcResult
acFFTForwardTransformSymmetricR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
AcResult
acFFTForwardTransformR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTBackwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
AcResult
acFFTBackwardTransformC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTBackwardTransform!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}

AcResult
acFFTForwardTransformR2Planar(const AcReal*, const Volume, const Volume, const Volume, AcReal*, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransformR2Planar!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}

AcResult
acFFTForwardTransformPlanar(const AcReal*, const AcReal*,const Volume, const Volume, const Volume, AcReal*, AcReal*)
{
	fprintf(stderr,"FATAL: need to have FFT_ENABLED=ON for acFFTForwardTransformPlanar!\n");
	fflush(stderr);
	exit(EXIT_FAILURE);
	return AC_FAILURE;
}
#endif

