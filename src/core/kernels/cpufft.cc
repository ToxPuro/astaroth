#include "host_datatypes.h"
#include "ac_fft.h"
#include "ac_mpi.h"
#include <stdio.h>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>
#include "kissfft/kiss_fft.h"
#include "errchk.h"
#include "distributed_fft.h"

//TP: CPU FFT binding built on kissfft.
//See distributed_fft.h for the semantics of the distributed transform.
//
//TP: Would not read too much into the code: is supposed to be CPU mirror of heffte.cc

static_assert(sizeof(kiss_fft_scalar) == sizeof(AcReal), "kissfft has to use the same precision as AcReal");

static DistributedFFT<kiss_fft_cpx> distributed_fft("CPUFFT");
static std::map<std::pair<long,bool>,kiss_fft_cfg> kiss_cfgs{};

static kiss_fft_cfg
get_kiss_cfg(const long n, const bool inverse)
{
	const auto key = std::make_pair(n,inverse);
	if(kiss_cfgs.find(key) == kiss_cfgs.end())
		kiss_cfgs[key] = kiss_fft_alloc((int)n,inverse,nullptr,nullptr);
	return kiss_cfgs[key];
}

//1D transforms along axis for data laid out as box, which has to hold the whole axis
static void
transform_axis(std::vector<kiss_fft_cpx>& data, const Box& box, const int axis, const bool inverse)
{
	const long n = box.dims[axis];
	if(box_count(box) == 0 || n == 1) return;
	const size_t strides[3] = {1, (size_t)box.dims[0], (size_t)(box.dims[0]*box.dims[1])};
	const size_t stride = strides[axis];
	const int a = (axis+1) % 3;
	const int b = (axis+2) % 3;
	kiss_fft_cfg cfg = get_kiss_cfg(n,inverse);
	std::vector<kiss_fft_cpx> in(n), out(n);
	for(long j = 0; j < box.dims[b]; ++j)
	{
		for(long i = 0; i < box.dims[a]; ++i)
		{
			const size_t base = i*strides[a] + j*strides[b];
			for(long k = 0; k < n; ++k) in[k] = data[base + k*stride];
			kiss_fft(cfg,in.data(),out.data());
			for(long k = 0; k < n; ++k) data[base + k*stride] = out[k];
		}
	}
}

//Transforms data, which is laid out as the local subdomain of layout
static void
transform(const DistributedLayout& layout, std::vector<kiss_fft_cpx>& data, const bool inverse)
{
	distributed_fft.transform(layout,data,inverse,
		[&](std::vector<kiss_fft_cpx>& pencil, const Box& box, const int axis){transform_axis(pencil,box,axis,inverse);},
		[](std::vector<kiss_fft_cpx>& elems, const AcReal scale)
		{
			for(auto& elem : elems)
			{
				elem.r *= scale;
				elem.i *= scale;
			}
		});
}

static size_t
get_domain_index(const Volume domain_size, const Volume starting_point, const size_t i, const size_t j, const size_t k)
{
	return (starting_point.x+i) + domain_size.x*((starting_point.y+j) + domain_size.y*(starting_point.z+k));
}

static void
check_subdomain(const Volume domain_size, const Volume subdomain_size, const Volume starting_point)
{
    ERRCHK_ALWAYS(starting_point.x + subdomain_size.x <= domain_size.x);
    ERRCHK_ALWAYS(starting_point.y + subdomain_size.y <= domain_size.y);
    ERRCHK_ALWAYS(starting_point.z + subdomain_size.z <= domain_size.z);
}

//Transforms the subdomain at starting_point of the domain given by load and writes the subdomain back with store.
//load(domain_index) returns the input value and store(domain_index,value) writes the output value
template <typename Load, typename Store>
static AcResult
acFFTTransform(const Volume domain_size, const Volume subdomain_size, const Volume starting_point, const bool inverse,
	       const Load& load, const Store& store)
{
	check_subdomain(domain_size,subdomain_size,starting_point);
	const DistributedLayout& layout = distributed_fft.get_layout(subdomain_size);
	std::vector<kiss_fft_cpx> data(subdomain_size.x*subdomain_size.y*subdomain_size.z);
	for(size_t k = 0; k < subdomain_size.z; ++k)
		for(size_t j = 0; j < subdomain_size.y; ++j)
			for(size_t i = 0; i < subdomain_size.x; ++i)
				data[i + subdomain_size.x*(j + subdomain_size.y*k)] = load(get_domain_index(domain_size,starting_point,i,j,k));

	transform(layout,data,inverse);

	for(size_t k = 0; k < subdomain_size.z; ++k)
		for(size_t j = 0; j < subdomain_size.y; ++j)
			for(size_t i = 0; i < subdomain_size.x; ++i)
				store(get_domain_index(domain_size,starting_point,i,j,k),data[i + subdomain_size.x*(j + subdomain_size.y*k)]);
	return AC_SUCCESS;
}

static kiss_fft_cpx
to_cpx(const AcReal real, const AcReal imag)
{
	kiss_fft_cpx res;
	res.r = real;
	res.i = imag;
	return res;
}

static AcResult
acFFTC2C(const AcComplex* src, const Volume domain_size,
                                const Volume subdomain_size, const Volume starting_point,
                                AcComplex* dst,  const bool inverse) {
	return acFFTTransform(domain_size,subdomain_size,starting_point,inverse,
			[&](const size_t idx){return to_cpx(src[idx].x,src[idx].y);},
			[&](const size_t idx, const kiss_fft_cpx val){dst[idx].x = val.r; dst[idx].y = val.i;}
			);
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
	ERRCHK_ALWAYS(false); //Not implemented
	return AC_FAILURE;
}


AcResult
acFFTBackwardTransformSymmetricC2R(const AcComplex*,const Volume, const Volume,const Volume, AcReal*)
{
	ERRCHK_ALWAYS(false); //Not implemented
	return AC_FAILURE;
}

AcResult
acFFTBackwardTransformC2R(const AcComplex* transformed_in,const Volume domain_size, const Volume subdomain_size,const Volume starting_point, AcReal* buffer) {
	return acFFTTransform(domain_size,subdomain_size,starting_point,true,
			[&](const size_t idx){return to_cpx(transformed_in[idx].x,transformed_in[idx].y);},
			[&](const size_t idx, const kiss_fft_cpx val){buffer[idx] = val.r;}
			);
}

AcResult
acFFTForwardTransformR2C(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcComplex* dst) {
	return acFFTTransform(domain_size,subdomain_size,starting_point,false,
			[&](const size_t idx){return to_cpx(src[idx],0.0);},
			[&](const size_t idx, const kiss_fft_cpx val){dst[idx].x = val.r; dst[idx].y = val.i;}
			);
}

AcResult
acFFTForwardTransformR2Planar(const AcReal* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
	return acFFTTransform(domain_size,subdomain_size,starting_point,false,
			[&](const size_t idx){return to_cpx(src[idx],0.0);},
			[&](const size_t idx, const kiss_fft_cpx val){real_dst[idx] = val.r; imag_dst[idx] = val.i;}
			);
}

AcResult
acFFTForwardTransformR2HermitianPlanarBatched(const AcReal*, const Volume, const Volume, const Volume, AcReal*, AcReal*, const int, cudaStream_t)
{
	return AC_FAILURE;
}

template <typename InputType, typename OutputType>
static AcResult
acFFTForwardTransformR2PlanarBatchedBase(const InputType* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, OutputType* real_dst, OutputType* imag_dst, const int batch_size)
{
    const size_t count = domain_size.x*domain_size.y*domain_size.z;
    for(int batch = 0; batch < batch_size; ++batch)
    {
	const InputType* batch_src = src + batch*count;
	OutputType* batch_real_dst = real_dst + batch*count;
	OutputType* batch_imag_dst = imag_dst + batch*count;
	const AcResult res = acFFTTransform(domain_size,subdomain_size,starting_point,false,
			[&](const size_t idx){return to_cpx((AcReal)batch_src[idx],0.0);},
			[&](const size_t idx, const kiss_fft_cpx val){batch_real_dst[idx] = (OutputType)val.r; batch_imag_dst[idx] = (OutputType)val.i;}
			);
	if(res != AC_SUCCESS) return res;
    }
    return AC_SUCCESS;
}

AcResult
acFFTForwardTransformR2PlanarBatched(const void* src, const Volume domain_size, const Volume subdomain_size, const Volume starting_point, void* real_dst, void* imag_dst, const int batch_size, AcPrecision input_precision, AcPrecision output_precision)
{
    if(input_precision == AC_SINGLE_PRECISION && output_precision == AC_REAL_PRECISION) return AC_FAILURE;
    if(input_precision == AC_HALF_PRECISION || output_precision == AC_HALF_PRECISION) return AC_FAILURE;
    if(output_precision == AC_SINGLE_PRECISION)
    {
	if(input_precision == AC_SINGLE_PRECISION)
		return acFFTForwardTransformR2PlanarBatchedBase((const float*)src,domain_size,subdomain_size,starting_point,(float*)real_dst,(float*)imag_dst,batch_size);
	return acFFTForwardTransformR2PlanarBatchedBase((const AcReal*)src,domain_size,subdomain_size,starting_point,(float*)real_dst,(float*)imag_dst,batch_size);
    }
    return acFFTForwardTransformR2PlanarBatchedBase((const AcReal*)src,domain_size,subdomain_size,starting_point,(AcReal*)real_dst,(AcReal*)imag_dst,batch_size);
}

AcResult
acFFTForwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
	return acFFTTransform(domain_size,subdomain_size,starting_point,false,
			[&](const size_t idx){return to_cpx(real_src[idx],imag_src[idx]);},
			[&](const size_t idx, const kiss_fft_cpx val){real_dst[idx] = val.r; imag_dst[idx] = val.i;}
			);
}

AcResult
acFFTBackwardTransformPlanar(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* real_dst, AcReal* imag_dst)
{
	return acFFTTransform(domain_size,subdomain_size,starting_point,true,
			[&](const size_t idx){return to_cpx(real_src[idx],imag_src[idx]);},
			[&](const size_t idx, const kiss_fft_cpx val){real_dst[idx] = val.r; imag_dst[idx] = val.i;}
			);
}

AcResult
acFFTBackwardTransformPlanar2R(const AcReal* real_src, const AcReal* imag_src ,const Volume domain_size, const Volume subdomain_size, const Volume starting_point, AcReal* dst)
{
	return acFFTTransform(domain_size,subdomain_size,starting_point,true,
			[&](const size_t idx){return to_cpx(real_src[idx],imag_src[idx]);},
			[&](const size_t idx, const kiss_fft_cpx val){dst[idx] = val.r;}
			);
}

static void
clear_kiss_cfgs()
{
	for(auto& [key, cfg] : kiss_cfgs) kiss_fft_free(cfg);
	kiss_cfgs.clear();
}

AcResult
acFFTInit(const AcCommunicator* astaroth_comm, const int* global_offset)
{
	clear_kiss_cfgs();
	distributed_fft.init(astaroth_comm,global_offset);
	return AC_SUCCESS;
}

AcResult
acFFTQuit()
{
	clear_kiss_cfgs();
	distributed_fft.quit();
	return AC_SUCCESS;
}
