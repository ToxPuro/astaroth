#include "host_datatypes.h"
#include "ac_fft.h"
#include <stdio.h>
#include <cstdlib>
#include "kissfft/kiss_fft.h"
#include "kissfft/kiss_fftnd.h"
#include "astaroth_cuda_wrappers.h"
#include "ac_helpers.h"
#include "common_kernels.h"

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
                fprintf(stderr,"CPUFFT integration not yet working for multiple processes!\n");
                exit(EXIT_FAILURE);
        }
#endif
}

static AcResult
acFFTC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst,  const bool inverse) {
	check_if_distributed();
	const size_t subdomain_count = subdomain_size.x*subdomain_size.y*subdomain_size.z;
 	kiss_fft_cpx* tmp = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * subdomain_count); kiss_fft_cpx* res = (kiss_fft_cpx*)malloc(sizeof(kiss_fft_cpx) * subdomain_count);
	for(size_t i = 0; i < subdomain_size.x; ++i)
	{
		for(size_t j = 0; j < subdomain_size.y; ++j)
		{
			for(size_t k = 0; k < subdomain_size.z; ++k)
			{
				const int subdomain_index = i + subdomain_size.x*(j + subdomain_size.y*k);
				const int domain_index =    (starting_point.x+i) + domain_size.x*((starting_point.y+j) + domain_size.y*(starting_point.z+k));
				tmp[subdomain_index].r = src[domain_index].x;
				tmp[subdomain_index].i = src[domain_index].y;
			}
		}
	}
	int dims[3] = {(int)subdomain_size.x, (int)subdomain_size.y, (int)subdomain_size.z};
	kiss_fftnd_cfg fwd_cfg = kiss_fftnd_alloc(dims, 3, inverse, nullptr, nullptr);
	kiss_fftnd(fwd_cfg, tmp, res);
	if(!inverse)
	{
		for (size_t i = 0; i < subdomain_count; ++i) {
        		res[i].r /= subdomain_count;
        		res[i].i /= subdomain_count;
    		}
	}
	for(size_t i = 0; i < subdomain_size.x; ++i)
	{
		for(size_t j = 0; j < subdomain_size.y; ++j)
		{
			for(size_t k = 0; k < subdomain_size.z; ++k)
			{
				const int subdomain_index = i + subdomain_size.x*(j + subdomain_size.y*k);
				const int domain_index =    (starting_point.x+i) + domain_size.x*((starting_point.y+j) + domain_size.y*(starting_point.z+k));
				dst[domain_index].x = res[subdomain_index].r;
				dst[domain_index].y = res[subdomain_index].i;
			}
		}
	}

	free(tmp);
	free(res);
	free(fwd_cfg);

	return AC_SUCCESS;
}

AcResult
acFFTForwardTransformC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst) {

	return acFFTC2C(src,domain_size,subdomain_size,starting_point,dst,false);
}

AcResult
acFFTBackwardTransformC2C(const AcComplex* src,
                                 const Volume domain_size,
                                 const Volume subdomain_size,
                                 const Volume starting_point,
                                 AcComplex* dst) {
	return acFFTC2C(src,domain_size,subdomain_size,starting_point,dst,true);
}

AcResult
acFFTForwardTransformSymmetricR2C(const AcReal*, const Volume, const Volume, const Volume, AcComplex*) {
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
acFFTForwardTransformR2PlanarBatched(const void* src_, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, void* real_dst_, void* imag_dst_, const int batch_size, AcPrecision input_precision, AcPrecision output_precision)
{
    if(input_precision != AC_REAL_PRECISION || output_precision != AC_REAL_PRECISION) return AC_FAILURE;

    const AcReal* src = (AcReal*)src_;
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
    acComplexToPlanar(tmp,count,real_dst,imag_dst);

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
