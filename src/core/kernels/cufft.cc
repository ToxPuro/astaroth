#include "host_datatypes.h"
#include "ac_fft.h"
#include "astaroth_cuda_wrappers.h"
#include "errchk.h"
#include "ac_helpers.h"
#include "common_kernels.h"
#include <stdio.h>
#include <cstdlib>
#include <cufftXt.h>
#include <cuComplex.h>

#if AC_DOUBLE_PRECISION
using  cuFFTPrecision = cuDoubleComplex;
#define CUFFT_COMPLEX2COMPLEX CUFFT_Z2Z
#else
#define CUFFT_COMPLEX2COMPLEX CUFFT_C2C
using  cuFFTPrecision = cuFloatComplex;
#endif

#if AC_MPI_ENABLED
#include <mpi.h>
struct AcCommunicator
{
	MPI_Comm handle;
};
static MPI_Comm communicator{};
#endif

void
check_if_distributed()
{
#if AC_MPI_ENABLED
        int nprocs{};
        MPI_Comm_size(communicator,&nprocs);
        if(nprocs > 1) 
        {
                fprintf(stderr,"CuFFT integration not yet working for multiple processes!\n");
                exit(EXIT_FAILURE);
        }
#endif
}

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
    check_if_distributed();
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
    
    cuFFTPrecision* transformed = reinterpret_cast<cuFFTPrecision*>(transformed_in);
    // Execute the plan_r2c
    CUFFT_CALL(cufftXtExec(plan_r2c, (void*)buffer, transformed, CUFFT_FORWARD));
    CUFFT_CALL(cufftDestroy(plan_r2c));
    // Scale complex results that inverse FFT results in original values
    const AcReal scale{AcReal(1.0) / orig_domain_size};
    acMultiplyInplaceComplex(scale, complex_domain_size, transformed_in);
    return AC_SUCCESS;
}
AcResult
acFFTTransformC2C(const AcComplex* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst,
		  const bool inverse)
{
    check_if_distributed();
    ERRCHK_ALWAYS(src != NULL);
    ERRCHK_ALWAYS(dst != NULL);
    ERRCHK_ALWAYS(subdomain_size.x <= domain_size.x);
    ERRCHK_ALWAYS(subdomain_size.y <= domain_size.y);
    ERRCHK_ALWAYS(subdomain_size.z <= domain_size.z);
    ERRCHK_ALWAYS(starting_point.x <= domain_size.x);
    ERRCHK_ALWAYS(starting_point.y <= domain_size.y);
    ERRCHK_ALWAYS(starting_point.z <= domain_size.z);
    ERRCHK_ALWAYS(starting_point.x  + subdomain_size.x<= domain_size.x);
    ERRCHK_ALWAYS(starting_point.y  + subdomain_size.y<= domain_size.y);
    ERRCHK_ALWAYS(starting_point.z  + subdomain_size.z<= domain_size.z);
    const size_t starting_offset = starting_point.x + domain_size.x*(starting_point.y + domain_size.y*starting_point.z);
    src = src + starting_offset;
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
        CUFFT_COMPLEX2COMPLEX, 1, &workspace_size));
    
    size_t complex_domain_size = onembed[0] * onembed[1] * onembed[2];    
    
    cuFFTPrecision* transformed = reinterpret_cast<cuFFTPrecision*>(dst + starting_offset);
    const auto direction = inverse ? CUFFT_INVERSE: CUFFT_FORWARD;
    // Execute the plan_r2c
    CUFFT_CALL(cufftXtExec(plan_r2c, (void*)src, transformed, direction));
    CUFFT_CALL(cufftDestroy(plan_r2c));
    // Scale complex results that inverse FFT results in original values
    const AcReal scale{AcReal(1.0) / ( dims[0] * dims[1] * dims[2])};
    if(!inverse) acMultiplyInplaceComplex(scale, complex_domain_size, dst);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformC2C(const AcComplex* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst) {
	return acFFTTransformC2C(src,domain_size,subdomain_size,starting_point,dst,false);
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
    const cuFFTPrecision* transformed = reinterpret_cast<const cuFFTPrecision*>(transformed_in);
    CUFFT_CALL(cufftXtExec(plan_c2r, (void*)transformed, buffer, CUFFT_INVERSE));
    CUFFT_CALL(cufftDestroy(plan_c2r));
    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformC2C(const AcComplex* src,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcComplex* dst) {
    return acFFTTransformC2C(src,domain_size,subdomain_size,starting_point,dst,true);
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
acFFTForwardTransformR2HermitianPlanarBatched(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst, const int batch_size, cudaStream_t stream)
{
	return AC_FAILURE;
}

AcResult
acFFTForwardTransformR2PlanarBatched(const void* src_, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, void* real_dst_, void* imag_dst_, const int batch_size, const AcPrecision precision)
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
    AcComplex* tmp  = get_fresh_complex_buffer(count);
    AcComplex* tmp2 = get_fresh_complex_buffer(count);

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp2,count,real_dst,imag_dst);

    acDeviceFree(&tmp,0);
    acDeviceFree(&tmp2,0);
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
    acComplexToPlanar(tmp2,count,real_dst,imag_dst);

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
    acKernelVolumeCopyComplexToReal(0,tmp2,starting_point,subdomain_size,domain_size,dst,starting_point,subdomain_size,domain_size);

    acDeviceFree(&tmp,0);
    acDeviceFree(&tmp2,0);
    return AC_SUCCESS;
}

AcResult
acFFTInit(const AcCommunicator* astaroth_comm, const int*)
{
#if AC_MPI_ENABLED
	communicator = astaroth_comm->handle;
#endif
	return AC_SUCCESS;
}

AcResult
acFFTQuit()
{
	return AC_SUCCESS;
}
