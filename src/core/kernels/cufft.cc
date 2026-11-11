#include <cuComplex.h>
#include <cufftXt.h>
#include <stdio.h>

#include <array>
#include <cstdlib>
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
using  cuFFTPrecision = cuDoubleComplex;
#define CUFFT_COMPLEX2COMPLEX CUFFT_Z2Z
#else
#define CUFFT_COMPLEX2COMPLEX CUFFT_C2C
using  cuFFTPrecision = cuFloatComplex;
#endif

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

//TP: Multi-GPU support for the C2C transforms, see distributed_fft.h and distributed_fft_device.h.
//The cuFFT plans of the batched 1D transforms along each axis and the device buffer for the pencils of a layout.
class CuFFTPencilPlans
{
public:
	CuFFTPencilPlans(const DistributedLayout& layout)
	{
		size_t max_count = 0;
		for(int axis = 0; axis < 3; ++axis)
		{
			const Box& box = layout.pencil_boxes[axis][layout.rank];
			max_count = std::max(max_count,box_count(box));
			if(layout.global_dims[axis] == 1 || box_count(box) == 0) continue;
			batch_layouts[axis] = get_pencil_batch_layout(box,axis);
			const PencilBatchLayout& bl = batch_layouts[axis];
			int n       = (int)bl.n;
			int embed[] = {n};
			CUFFT_CALL(cufftCreate(&plans[axis]));
			has_plan[axis] = true;
			size_t workspace_size;
			CUFFT_CALL(cufftMakePlanMany(plans[axis], 1, &n,
				embed, (int)bl.stride, (int)bl.dist,
				embed, (int)bl.stride, (int)bl.dist,
				CUFFT_COMPLEX2COMPLEX, (int)bl.batch, &workspace_size));
		}
		if(max_count) acDeviceMalloc((void**)&d_buf,sizeof(AcComplex)*max_count);
	}
	~CuFFTPencilPlans()
	{
		for(int axis = 0; axis < 3; ++axis)
			if(has_plan[axis]) cufftDestroy(plans[axis]);
		if(d_buf) acDeviceFree(&d_buf,0);
	}
	CuFFTPencilPlans(const CuFFTPencilPlans&) = delete;
	CuFFTPencilPlans& operator=(const CuFFTPencilPlans&) = delete;

	void
	transform_axis(std::vector<AcComplex>& data, const int axis, const bool inverse)
	{
		if(!has_plan[axis]) return;
		const size_t bytes = sizeof(AcComplex)*data.size();
		ERRCHK_CUDA_ALWAYS(acMemcpy(d_buf,data.data(),bytes,cudaMemcpyHostToDevice));
		const auto direction = inverse ? CUFFT_INVERSE : CUFFT_FORWARD;
		for(size_t i = 0; i < batch_layouts[axis].loops; ++i)
		{
			cuFFTPrecision* ptr = reinterpret_cast<cuFFTPrecision*>(d_buf + i*batch_layouts[axis].loop_offset);
			CUFFT_CALL(cufftXtExec(plans[axis], ptr, ptr, direction));
		}
		ERRCHK_CUDA_ALWAYS(acDeviceSynchronize());
		ERRCHK_CUDA_ALWAYS(acMemcpy(data.data(),d_buf,bytes,cudaMemcpyDeviceToHost));
	}
private:
	std::array<cufftHandle,3> plans{};
	std::array<bool,3> has_plan{false,false,false};
	std::array<PencilBatchLayout,3> batch_layouts{};
	AcComplex* d_buf = NULL;
};

static DeviceDistributedFFT<CuFFTPencilPlans> distributed_fft("CuFFT");

void
check_if_distributed()
{
        if(distributed_fft.nprocs() > 1)
        {
                fprintf(stderr,"CuFFT integration of symmetric R2C/C2R transforms not yet working for multiple processes!\n");
                exit(EXIT_FAILURE);
        }
}

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
    if(distributed_fft.is_distributed(subdomain_size))
	    return distributed_fft.transform_c2c(src,domain_size,subdomain_size,starting_point,dst,inverse);
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
    check_if_distributed();
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
    acKernelVolumeCopyComplexToPlanar(0,tmp2
		    			,starting_point,subdomain_size,domain_size
					,real_dst,imag_dst
		    			,starting_point,subdomain_size,domain_size
		    );

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
acFFTInit(const AcCommunicator* astaroth_comm, const int* global_offset)
{
	distributed_fft.init(astaroth_comm,global_offset);
	return AC_SUCCESS;
}

AcResult
acFFTQuit()
{
	distributed_fft.quit();
	return AC_SUCCESS;
}
