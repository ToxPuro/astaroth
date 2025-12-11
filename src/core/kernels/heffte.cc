#include "host_datatypes.h"
#include "ac_fft.h"
#include "astaroth_cuda_wrappers.h"
#include "errchk.h"
#include "ac_helpers.h"
#include "common_kernels.h"
#include <stdio.h>
#include <cstdlib>
#include "heffte.h"
#include <mpi.h>
struct AcCommunicator
{
	MPI_Comm handle;
};
static MPI_Comm communicator{};
[[maybe_unused]] static Volume global_offset = (Volume){0,0,0};

static AcComplex*
get_fresh_complex_buffer(const size_t count)
{
    const size_t bytes = sizeof(AcComplex)*count;
    AcComplex* res = NULL;
    acDeviceMalloc((void**)&res,bytes);
    acMultiplyInplaceComplex(AcReal(0.0),count,res);
    return res;
}

typedef struct
{
	AcComplex* in;
	AcComplex* out;
} AcComplexInAndOut;

#include <unordered_map>
static std::unordered_map<size_t,heffte::fft3d<heffte::backend::rocfft>> plans{};
static std::unordered_map<size_t,heffte::fft3d_r2c<heffte::backend::rocfft>> plans_r2c{};


AcResult
acFFTForwardTransformSymmetricR2C(const AcReal* buffer, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* transformed_in) {
	(void)buffer;
	(void)domain_size;
	(void)subdomain_size;
	(void)starting_point;
	(void)transformed_in;
	ERRCHK_ALWAYS(false); //Not implemented
}

AcResult
acFFTTransformR2CBase(cudaStream_t stream, const AcReal* src, const Volume domain_size, AcComplex* dst,
		  const bool inverse, const int batch_size)
{
    static std::unordered_map<size_t,AcComplex*> work_buffers{};
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    //HeFFTe is surprisingly inclusive in the domain
    const int3 lower = (int3)
    {
	    (int)global_offset.x,
	    (int)global_offset.y,
	    (int)global_offset.z
    };
    const int3 dims = (int3)
    {
	    (int)domain_size.x,
	    (int)domain_size.y,
	    (int)domain_size.z
    };
    const int3 upper = lower+dims-(int3){1,1,1};
    const int3 output_lower = (int3)
    {
	    (int)global_offset.x,
	    (int)global_offset.y,
	    (int)global_offset.z/2
    };
    const int3 output_dims = (int3)
    {
	    (int)domain_size.x,
	    (int)domain_size.y,
	    (int)domain_size.z/2
    };
    const int3 output_upper = output_lower+output_dims-(int3){1,1,1};
    if(plans_r2c.find(count) == plans_r2c.end())
    {
        heffte::box3d<> const input_box  = {{lower.x,lower.y,lower.z},{upper.x,upper.y,upper.z}};
        heffte::box3d<> const output_box = {{output_lower.x,output_lower.y,output_lower.z},{output_upper.x,output_upper.y,output_upper.z}};
	heffte::plan_options options = heffte::default_options<heffte::backend::rocfft>();
        options.algorithm = heffte::reshape_algorithm::p2p_plined;
        //options.algorithm = heffte::reshape_algorithm::alltoall;
        //options.algorithm = heffte::reshape_algorithm::p2p;
	options.use_pencils = true;
	options.use_reorder = true;
        //options.algorithm = heffte::reshape_algorithm::alltoallv;
	//options.use_gpu_aware = false;
        heffte::fft3d_r2c<heffte::backend::rocfft> fft(stream, input_box, output_box, 2,communicator, options);
	plans_r2c.emplace(count,std::move(fft));
	work_buffers[count] = get_fresh_complex_buffer(batch_size*fft.size_workspace());
    }
    AcComplex* workspace = work_buffers[count];
    if(inverse)
    {
    	;//plans_r2c.at(count).backward(batch_size,src, (std::complex<AcReal>*)dst, (std::complex<AcReal>*)workspace, heffte::scale::none);
    }
    else
    {
    	plans_r2c.at(count).forward(batch_size,src, (std::complex<AcReal>*)dst, (std::complex<AcReal>*)workspace, heffte::scale::full);
    }
    return AC_SUCCESS;
}

AcResult
acFFTTransformC2CBase(const AcComplex* src, const Volume domain_size, AcComplex* dst,
		  const bool inverse, const int batch_size)
{
    static std::unordered_map<size_t,AcComplex*> work_buffers{};
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    //HeFFTe is surprisingly inclusive in the domain
    const int3 lower = (int3)
    {
	    (int)global_offset.x,
	    (int)global_offset.y,
	    domain_size.z == 1 ? 0 : (int)global_offset.z
    };
    const int3 dims = (int3)
    {
	    (int)domain_size.x,
	    (int)domain_size.y,
	    (int)domain_size.z
    };
    const int3 upper = lower+dims-(int3){1,1,1};
    if(plans.find(count) == plans.end())
    {
        heffte::box3d<> const my_box = {{lower.x,lower.y,lower.z},{upper.x,upper.y,upper.z}};
        heffte::fft3d<heffte::backend::rocfft> fft(my_box, my_box, communicator);
	plans.emplace(count,std::move(fft));
	work_buffers[count] = get_fresh_complex_buffer(batch_size*fft.size_workspace());
    }
    AcComplex* workspace = work_buffers[count];
    if(inverse)
    {
    	plans.at(count).backward(batch_size,(std::complex<AcReal>*)src, (std::complex<AcReal>*)dst, (std::complex<AcReal>*)workspace, heffte::scale::none);
    }
    else
    {
    	plans.at(count).forward(batch_size,(std::complex<AcReal>*)src, (std::complex<AcReal>*)dst, (std::complex<AcReal>*)workspace, heffte::scale::full);
    }
    return AC_SUCCESS;
}

AcResult
acFFTTransformC2C(const AcComplex* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst,
		  const bool inverse)
{
    static std::unordered_map<size_t,AcComplexInAndOut> tmp_buffers{};
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
    const size_t count = subdomain_size.x*subdomain_size.y*subdomain_size.z;


    if (tmp_buffers.find(count) == tmp_buffers.end())
    {

    	AcComplex* tmp_in  = get_fresh_complex_buffer(count);
    	AcComplex* tmp_out = get_fresh_complex_buffer(count);
    	tmp_buffers[count] = (AcComplexInAndOut){tmp_in,tmp_out};
    }
    AcComplex* tmp_in  = tmp_buffers[count].in;
    AcComplex* tmp_out = tmp_buffers[count].out;

    acKernelVolumeCopyComplex(0,src,starting_point,domain_size,tmp_in,(Volume){0,0,0},subdomain_size);
    acFFTTransformC2CBase(tmp_in, subdomain_size, tmp_out, inverse,1);
    acKernelVolumeCopyComplex(0,tmp_out,(Volume){0,0,0},subdomain_size,dst,starting_point,domain_size);
    return AC_SUCCESS;
}


AcResult
acFFTForwardTransformC2C(const AcComplex* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst) {
	return acFFTTransformC2C(src,domain_size,subdomain_size,starting_point,dst,false);
}

AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer) {
	(void)transformed_in;
	(void)domain_size;
	(void)subdomain_size;
	(void)starting_point;
	(void)buffer;
	ERRCHK_ALWAYS(false); //Not implemented
}

AcResult
acFFTBackwardTransformC2C(const AcComplex* src,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcComplex* dst) {
    return acFFTTransformC2C(src,domain_size,subdomain_size,starting_point,dst,true);
}


AcResult
acFFTBackwardTransformC2R(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer) {
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    static std::unordered_map<size_t,AcComplex*> tmp_buffers{};
    if (tmp_buffers.find(count) == tmp_buffers.end())
    {
    	AcComplex* tmp = get_fresh_complex_buffer(count);
	tmp_buffers[count] = tmp;
    }
    AcComplex* tmp = tmp_buffers[count];
    acFFTBackwardTransformC2C(transformed_in,domain_size,subdomain_size,starting_point,tmp);
    acComplexToReal(tmp,count,buffer);
    return AC_SUCCESS;
}


AcResult
acFFTForwardTransformR2C(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst) {
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    static std::unordered_map<size_t,AcComplex*> tmp_buffers{};
    if (tmp_buffers.find(count) == tmp_buffers.end())
    {
    	AcComplex* tmp = get_fresh_complex_buffer(count);
	tmp_buffers[count] = tmp;
    }
    AcComplex* tmp = tmp_buffers[count];
    acRealToComplex(src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,dst);
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformR2Planar(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = subdomain_size.x*subdomain_size.y*subdomain_size.z;
    static std::unordered_map<size_t,AcComplexInAndOut> tmp_buffers{};
    if (tmp_buffers.find(count) == tmp_buffers.end())
    {
    	AcComplex* tmp  = get_fresh_complex_buffer(count);
    	AcComplex* tmp2 = get_fresh_complex_buffer(count);
	tmp_buffers[count].in  = tmp;
	tmp_buffers[count].out = tmp2;
    }

    AcComplex* tmp_in  = tmp_buffers[count].in;
    AcComplex* tmp_out = tmp_buffers[count].out;

    acKernelVolumeCopyRealToComplex(0,src,starting_point,subdomain_size,domain_size,tmp_in,(Volume){0,0,0},subdomain_size,subdomain_size);
    acFFTTransformC2CBase(tmp_in, subdomain_size, tmp_out, false,1);

    acKernelVolumeCopyComplexToPlanar(0,tmp_out,(Volume){0,0,0},subdomain_size,subdomain_size,real_dst,imag_dst,starting_point,subdomain_size,domain_size);
    return AC_SUCCESS;
}


//This is not a correct implementation simply for benchmarking purposes
AcResult
acFFTForwardTransformR2HermitianPlanarBatched(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst, const int batch_size, const cudaStream_t stream)
{
    const size_t count = subdomain_size.x*subdomain_size.y*subdomain_size.z*batch_size;
    static std::unordered_map<size_t,AcComplexInAndOut> tmp_buffers{};
    if (tmp_buffers.find(count) == tmp_buffers.end())
    {
    	AcComplex* tmp  = get_fresh_complex_buffer(count);
    	AcComplex* tmp2 = get_fresh_complex_buffer(count);
	tmp_buffers[count].in  = tmp;
	tmp_buffers[count].out = tmp2;
    }

    AcComplex* tmp_in  = tmp_buffers[count].in;
    AcComplex* tmp_out = tmp_buffers[count].out;
    acKernelVolumeCopyRealToComplexBatched(stream,src,starting_point,domain_size,tmp_in,(Volume){0,0,0},subdomain_size,batch_size);

    acFFTTransformR2CBase(stream,src,subdomain_size,tmp_out,false,batch_size);
    acKernelVolumeCopyComplexToPlanarBatched(stream,tmp_out,(Volume){0,0,0},subdomain_size,real_dst,imag_dst,starting_point,domain_size,batch_size);

    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformR2PlanarBatched(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst, const int batch_size)
{
    const size_t count = subdomain_size.x*subdomain_size.y*subdomain_size.z*batch_size;
    static std::unordered_map<size_t,AcComplexInAndOut> tmp_buffers{};
    if (tmp_buffers.find(count) == tmp_buffers.end())
    {
    	AcComplex* tmp  = get_fresh_complex_buffer(count);
    	AcComplex* tmp2 = get_fresh_complex_buffer(count);
	tmp_buffers[count].in  = tmp;
	tmp_buffers[count].out = tmp2;
    }

    AcComplex* tmp_in  = tmp_buffers[count].in;
    AcComplex* tmp_out = tmp_buffers[count].out;
    acKernelVolumeCopyRealToComplexBatched(0,src,starting_point,domain_size,tmp_in,(Volume){0,0,0},subdomain_size,batch_size);

    acFFTTransformC2CBase(tmp_in,subdomain_size,tmp_out,false,batch_size);
    acKernelVolumeCopyComplexToPlanarBatched(0,tmp_out,(Volume){0,0,0},subdomain_size,real_dst,imag_dst,starting_point,domain_size,batch_size);

    return AC_SUCCESS;
}


AcResult
acFFTForwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    static std::unordered_map<size_t,AcComplexInAndOut> tmp_buffers{};
    if (tmp_buffers.find(count) == tmp_buffers.end())
    {
    	AcComplex* tmp  = get_fresh_complex_buffer(count);
    	AcComplex* tmp2 = get_fresh_complex_buffer(count);
	tmp_buffers[count].in  = tmp;
	tmp_buffers[count].out = tmp2;
    }

    AcComplex* tmp  = tmp_buffers[count].in;
    AcComplex* tmp2 = tmp_buffers[count].out;

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTForwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp2,count,real_dst,imag_dst);

    return AC_SUCCESS;
}

AcResult
acFFTBackwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    static std::unordered_map<size_t,AcComplexInAndOut> tmp_buffers{};
    if (tmp_buffers.find(count) == tmp_buffers.end())
    {
    	AcComplex* tmp  = get_fresh_complex_buffer(count);
    	AcComplex* tmp2 = get_fresh_complex_buffer(count);
	tmp_buffers[count].in  = tmp;
	tmp_buffers[count].out = tmp2;
    }

    AcComplex* tmp  = tmp_buffers[count].in;
    AcComplex* tmp2 = tmp_buffers[count].out;

    acPlanarToComplex(real_src,imag_src,count,tmp);
    acFFTBackwardTransformC2C(tmp, domain_size,subdomain_size,starting_point,tmp2);
    acComplexToPlanar(tmp2,count,real_dst,imag_dst);

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
acFFTInit(const AcCommunicator* astaroth_comm, const int* global_offset_)
{
	communicator = astaroth_comm->handle;
	global_offset = (Volume){(size_t)global_offset_[0],(size_t)global_offset_[1],(size_t)global_offset_[2]};
	return AC_SUCCESS;

}

AcResult
acFFTQuit()
{
	plans.clear();
	plans_r2c.clear();
	return AC_SUCCESS;
}
