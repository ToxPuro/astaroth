#include "host_datatypes.h"
#include "ac_fft.h"
#include "ac_mpi.h"
#include <stdio.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <array>
#include <map>
#include <utility>
#include <vector>
#include "kissfft/kiss_fft.h"
#include "errchk.h"

//TP: CPU FFT binding built on kissfft.
//Has the same semantics as the HeFFTe binding:
//	Each process passes its local subdomain, which is placed in the global grid at the offset given to acFFTInit.
//	The transform is taken over the global grid formed by the union of the subdomains of all processes.
//	If the subdomain is a single xy-plane (z == 1) the transform is taken over the global xy-plane,
//	i.e. only together with the processes sharing the same z-offset.
//	Forward transforms are normalized by the global element count, backward transforms are unnormalized.
//
//The distributed transform is done with 1D transforms along each axis.
//Before transforming along an axis the data is redistributed to pencils which hold that whole axis.
//
//TP: Would not read too much into the code: is supposed to be CPU mirror of heffte.cc

static_assert(sizeof(kiss_fft_scalar) == sizeof(AcReal), "kissfft has to use the same precision as AcReal");

#if AC_MPI_ENABLED
static MPI_Comm communicator          = MPI_COMM_NULL;
static MPI_Comm xy_plane_communicator = MPI_COMM_NULL;
static MPI_Datatype complex_type      = MPI_DATATYPE_NULL;
#endif
static Volume global_offset = (Volume){0,0,0};

typedef std::array<long,3> Long3;

typedef struct
{
	Long3 lower;
	Long3 dims;
} Box;

static size_t
box_count(const Box& box)
{
	return box.dims[0]*box.dims[1]*box.dims[2];
}

static Box
intersect(const Box& a, const Box& b)
{
	Box res{};
	for(int d = 0; d < 3; ++d)
	{
		const long lower = std::max(a.lower[d],b.lower[d]);
		const long upper = std::min(a.lower[d]+a.dims[d],b.lower[d]+b.dims[d]);
		res.lower[d] = lower;
		res.dims[d]  = std::max(upper-lower,0L);
	}
	return res;
}

typedef struct
{
#if AC_MPI_ENABLED
	MPI_Comm comm;
#endif
	int nprocs;
	int rank;
	Long3 global_dims;
	//Boxes of all processes: the user layout and the pencil layout for each axis
	std::vector<Box> user_boxes;
	std::array<std::vector<Box>,3> pencil_boxes;
} Plan;

//Plans are cached by the shape of the local subdomain
typedef std::array<long,3> PlanKey;
static std::map<PlanKey,Plan> plans{};
static std::map<std::pair<long,bool>,kiss_fft_cfg> kiss_cfgs{};

static void
split(const long n, const int parts, const int i, long* lower, long* dims)
{
	const long start = (n*i)/parts;
	const long end   = (n*(i+1))/parts;
	*lower = start;
	*dims  = end-start;
}

//Distributes the plane spanned by the axes other than axis among nprocs processes
static std::vector<Box>
get_pencils(const Long3 global_dims, const int axis, const int nprocs)
{
	const int a = (axis+1) % 3;
	const int b = (axis+2) % 3;
	//Choose the factorization pa*pb == nprocs which keeps the pencils as square as possible
	//while preferring factorizations where no process is left empty
	int best_pa = 1;
	double best_score = -1;
	for(int pa = 1; pa <= nprocs; ++pa)
	{
		if(nprocs % pa) continue;
		const int pb = nprocs/pa;
		const bool fits = pa <= global_dims[a] && pb <= global_dims[b];
		const double ratio = (double(global_dims[a])/pa) / (double(global_dims[b])/pb);
		const double score = (fits ? 2.0 : 0.0) + std::min(ratio,1.0/ratio);
		if(score > best_score)
		{
			best_score = score;
			best_pa = pa;
		}
	}
	const int pa = best_pa;
	const int pb = nprocs/pa;
	std::vector<Box> res(nprocs);
	for(int rank = 0; rank < nprocs; ++rank)
	{
		Box& box = res[rank];
		box.lower[axis] = 0;
		box.dims[axis]  = global_dims[axis];
		split(global_dims[a],pa,rank % pa,&box.lower[a],&box.dims[a]);
		split(global_dims[b],pb,rank / pa,&box.lower[b],&box.dims[b]);
	}
	return res;
}

static Plan&
get_plan(const Volume subdomain_size)
{
	const PlanKey key = {(long)subdomain_size.x,(long)subdomain_size.y,(long)subdomain_size.z};
	if(plans.find(key) != plans.end()) return plans.at(key);

	Plan plan{};
	const Box my_box =
	{
		{(long)global_offset.x,(long)global_offset.y, subdomain_size.z == 1 ? 0L : (long)global_offset.z},
		{(long)subdomain_size.x,(long)subdomain_size.y,(long)subdomain_size.z},
	};
#if AC_MPI_ENABLED
	plan.comm = subdomain_size.z == 1 ? xy_plane_communicator : communicator;
	ERRCHK_ALWAYS(plan.comm != MPI_COMM_NULL);
	MPI_Comm_size(plan.comm,&plan.nprocs);
	MPI_Comm_rank(plan.comm,&plan.rank);
	plan.user_boxes.resize(plan.nprocs);
	MPI_Allgather(&my_box,sizeof(Box),MPI_BYTE,plan.user_boxes.data(),sizeof(Box),MPI_BYTE,plan.comm);
#else
	plan.nprocs = 1;
	plan.rank   = 0;
	plan.user_boxes = {my_box};
#endif
	//The global grid is the bounding box of the subdomains, which have to tile it
	Long3 lower = my_box.lower;
	Long3 upper = my_box.lower;
	for(int d = 0; d < 3; ++d) upper[d] += my_box.dims[d];
	size_t total_count = 0;
	for(const Box& box : plan.user_boxes)
	{
		for(int d = 0; d < 3; ++d)
		{
			lower[d] = std::min(lower[d],box.lower[d]);
			upper[d] = std::max(upper[d],box.lower[d]+box.dims[d]);
		}
		total_count += box_count(box);
	}
	for(int d = 0; d < 3; ++d) plan.global_dims[d] = upper[d]-lower[d];
	if(total_count != (size_t)(plan.global_dims[0]*plan.global_dims[1]*plan.global_dims[2]))
	{
		fprintf(stderr,"CPUFFT: the subdomains of the processes do not tile the global domain!\n");
		fflush(stderr);
		exit(EXIT_FAILURE);
	}
	//Work relative to the origin of the global grid
	for(Box& box : plan.user_boxes)
		for(int d = 0; d < 3; ++d) box.lower[d] -= lower[d];

	for(int axis = 0; axis < 3; ++axis)
		plan.pencil_boxes[axis] = get_pencils(plan.global_dims,axis,plan.nprocs);

	plans.emplace(key,std::move(plan));
	return plans.at(key);
}

//Copies the region of src (laid out as src_box) that overlaps dst_box into dst (laid out as dst_box)
static void
copy_box(const kiss_fft_cpx* src, const Box& src_box, kiss_fft_cpx* dst, const Box& dst_box, const Box& region)
{
	for(long k = 0; k < region.dims[2]; ++k)
	{
		for(long j = 0; j < region.dims[1]; ++j)
		{
			const long x = region.lower[0];
			const long y = region.lower[1]+j;
			const long z = region.lower[2]+k;
			const size_t src_index = (x-src_box.lower[0]) + src_box.dims[0]*((y-src_box.lower[1]) + src_box.dims[1]*(z-src_box.lower[2]));
			const size_t dst_index = (x-dst_box.lower[0]) + dst_box.dims[0]*((y-dst_box.lower[1]) + dst_box.dims[1]*(z-dst_box.lower[2]));
			memcpy(&dst[dst_index],&src[src_index],sizeof(kiss_fft_cpx)*region.dims[0]);
		}
	}
}

//Redistributes the data from the layout src_boxes to the layout dst_boxes
static std::vector<kiss_fft_cpx>
reshape(const Plan& plan, const std::vector<kiss_fft_cpx>& src, const std::vector<Box>& src_boxes, const std::vector<Box>& dst_boxes)
{
	const Box& my_src = src_boxes[plan.rank];
	const Box& my_dst = dst_boxes[plan.rank];
	std::vector<kiss_fft_cpx> dst(box_count(my_dst));
#if AC_MPI_ENABLED
	if(plan.nprocs > 1)
	{
		std::vector<int> send_counts(plan.nprocs), send_displs(plan.nprocs);
		std::vector<int> recv_counts(plan.nprocs), recv_displs(plan.nprocs);
		size_t send_total = 0;
		size_t recv_total = 0;
		for(int i = 0; i < plan.nprocs; ++i)
		{
			const size_t send_count = box_count(intersect(my_src,dst_boxes[i]));
			const size_t recv_count = box_count(intersect(src_boxes[i],my_dst));
			ERRCHK_ALWAYS(send_total + send_count <= (size_t)INT32_MAX);
			ERRCHK_ALWAYS(recv_total + recv_count <= (size_t)INT32_MAX);
			send_counts[i] = send_count;
			recv_counts[i] = recv_count;
			send_displs[i] = send_total;
			recv_displs[i] = recv_total;
			send_total += send_count;
			recv_total += recv_count;
		}
		std::vector<kiss_fft_cpx> send_buf(send_total);
		std::vector<kiss_fft_cpx> recv_buf(recv_total);
		for(int i = 0; i < plan.nprocs; ++i)
		{
			const Box region = intersect(my_src,dst_boxes[i]);
			if(box_count(region) == 0) continue;
			copy_box(src.data(),my_src,&send_buf[send_displs[i]],region,region);
		}
		MPI_Alltoallv(send_buf.data(),send_counts.data(),send_displs.data(),complex_type,
			      recv_buf.data(),recv_counts.data(),recv_displs.data(),complex_type,
			      plan.comm);
		for(int i = 0; i < plan.nprocs; ++i)
		{
			const Box region = intersect(src_boxes[i],my_dst);
			if(box_count(region) == 0) continue;
			copy_box(&recv_buf[recv_displs[i]],region,dst.data(),my_dst,region);
		}
		return dst;
	}
#endif
	copy_box(src.data(),my_src,dst.data(),my_dst,intersect(my_src,my_dst));
	return dst;
}

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

//Transforms data, which is laid out as the local subdomain of the plan
static void
transform(const Plan& plan, std::vector<kiss_fft_cpx>& data, const bool inverse)
{
	const std::vector<Box>* layout = &plan.user_boxes;
	for(int axis = 0; axis < 3; ++axis)
	{
		if(plan.global_dims[axis] == 1) continue;
		data = reshape(plan,data,*layout,plan.pencil_boxes[axis]);
		layout = &plan.pencil_boxes[axis];
		transform_axis(data,(*layout)[plan.rank],axis,inverse);
	}
	if(layout != &plan.user_boxes)
		data = reshape(plan,data,*layout,plan.user_boxes);
	if(!inverse)
	{
		const AcReal scale = AcReal(1.0)/(plan.global_dims[0]*plan.global_dims[1]*plan.global_dims[2]);
		for(auto& elem : data)
		{
			elem.r *= scale;
			elem.i *= scale;
		}
	}
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
	const Plan& plan = get_plan(subdomain_size);
	std::vector<kiss_fft_cpx> data(subdomain_size.x*subdomain_size.y*subdomain_size.z);
	for(size_t k = 0; k < subdomain_size.z; ++k)
		for(size_t j = 0; j < subdomain_size.y; ++j)
			for(size_t i = 0; i < subdomain_size.x; ++i)
				data[i + subdomain_size.x*(j + subdomain_size.y*k)] = load(get_domain_index(domain_size,starting_point,i,j,k));

	transform(plan,data,inverse);

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
clear_plans()
{
	plans.clear();
	for(auto& [key, cfg] : kiss_cfgs) kiss_fft_free(cfg);
	kiss_cfgs.clear();
}

AcResult
acFFTInit(const AcCommunicator* astaroth_comm, const int* global_offset_)
{
	clear_plans();
	global_offset = (Volume){(size_t)global_offset_[0],(size_t)global_offset_[1],(size_t)global_offset_[2]};
#if AC_MPI_ENABLED
	communicator = astaroth_comm->handle;
	int rank{};
	MPI_Comm_rank(communicator,&rank);
	if(xy_plane_communicator != MPI_COMM_NULL) MPI_Comm_free(&xy_plane_communicator);
	MPI_Comm_split(communicator,(int)global_offset.z,rank,&xy_plane_communicator);
	if(complex_type == MPI_DATATYPE_NULL)
	{
		MPI_Type_contiguous(sizeof(kiss_fft_cpx),MPI_BYTE,&complex_type);
		MPI_Type_commit(&complex_type);
	}
#else
	(void)astaroth_comm;
#endif
	return AC_SUCCESS;
}

AcResult
acFFTQuit()
{
	clear_plans();
#if AC_MPI_ENABLED
	if(xy_plane_communicator != MPI_COMM_NULL) MPI_Comm_free(&xy_plane_communicator);
	if(complex_type != MPI_DATATYPE_NULL) MPI_Type_free(&complex_type);
#endif
	return AC_SUCCESS;
}
