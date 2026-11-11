#pragma once
#include <stdio.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <utility>
#include <vector>
#include "host_datatypes.h"
#include "ac_mpi.h"
#include "errchk.h"

//TP: Common machinery for the distributed FFT backends which are built on top of a single-process FFT library (cpufft.cc, cufft.cc).
//Has the same semantics as the HeFFTe binding:
//	Each process passes its local subdomain, which is placed in the global grid at the offset given to acFFTInit.
//	The transform is taken over the global grid formed by the union of the subdomains of all processes.
//	If the subdomain is a single xy-plane (z == 1) the transform is taken over the global xy-plane,
//	i.e. only together with the processes sharing the same z-offset.
//	Forward transforms are normalized by the global element count, backward transforms are unnormalized.
//
//The distributed transform is done with 1D transforms along each axis.
//Before transforming along an axis the data is redistributed to pencils which hold that whole axis.
//This header provides the layouts of the pencils and the redistribution (reshape) between them on the host,
//the backends provide the 1D transforms.

typedef std::array<long,3> Long3;

typedef struct
{
	Long3 lower;
	Long3 dims;
} Box;

static inline size_t
box_count(const Box& box)
{
	return box.dims[0]*box.dims[1]*box.dims[2];
}

static inline Box
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
} DistributedLayout;

//Layouts are cached by the shape of the local subdomain
typedef std::array<long,3> LayoutKey;

static inline LayoutKey
get_layout_key(const Volume subdomain_size)
{
	return {(long)subdomain_size.x,(long)subdomain_size.y,(long)subdomain_size.z};
}

static inline void
split(const long n, const int parts, const int i, long* lower, long* dims)
{
	const long start = (n*i)/parts;
	const long end   = (n*(i+1))/parts;
	*lower = start;
	*dims  = end-start;
}

//Distributes the plane spanned by the axes other than axis among nprocs processes
static inline std::vector<Box>
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

//Copies the region of src (laid out as src_box) that overlaps dst_box into dst (laid out as dst_box)
template <typename T>
static void
copy_box(const T* src, const Box& src_box, T* dst, const Box& dst_box, const Box& region)
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
			std::copy(&src[src_index],&src[src_index]+region.dims[0],&dst[dst_index]);
		}
	}
}

//The communicators and the cached layouts of a distributed FFT backend.
//T is the complex type the backend uses on the host
template <typename T>
class DistributedFFT
{
public:
	DistributedFFT(const char* name_) : name(name_) {}
	DistributedFFT(const DistributedFFT&) = delete;
	DistributedFFT& operator=(const DistributedFFT&) = delete;

	void
	init(const AcCommunicator* astaroth_comm, const int* global_offset_)
	{
		layouts.clear();
		global_offset = global_offset_ ?
			(Volume){(size_t)global_offset_[0],(size_t)global_offset_[1],(size_t)global_offset_[2]} :
			(Volume){0,0,0};
#if AC_MPI_ENABLED
		communicator = astaroth_comm->handle;
		int rank{};
		MPI_Comm_rank(communicator,&rank);
		if(xy_plane_communicator != MPI_COMM_NULL) MPI_Comm_free(&xy_plane_communicator);
		MPI_Comm_split(communicator,(int)global_offset.z,rank,&xy_plane_communicator);
		if(complex_type == MPI_DATATYPE_NULL)
		{
			MPI_Type_contiguous(sizeof(T),MPI_BYTE,&complex_type);
			MPI_Type_commit(&complex_type);
		}
#else
		(void)astaroth_comm;
#endif
	}

	void
	quit()
	{
		layouts.clear();
#if AC_MPI_ENABLED
		if(xy_plane_communicator != MPI_COMM_NULL) MPI_Comm_free(&xy_plane_communicator);
		if(complex_type != MPI_DATATYPE_NULL) MPI_Type_free(&complex_type);
		communicator = MPI_COMM_NULL;
#endif
	}

	//Number of processes taking part in the whole transform (not in a single xy-plane)
	int
	nprocs() const
	{
#if AC_MPI_ENABLED
		if(communicator == MPI_COMM_NULL) return 1;
		int res{};
		MPI_Comm_size(communicator,&res);
		return res;
#else
		return 1;
#endif
	}

	const DistributedLayout&
	get_layout(const Volume subdomain_size)
	{
		const LayoutKey key = get_layout_key(subdomain_size);
		if(layouts.find(key) != layouts.end()) return layouts.at(key);

		DistributedLayout layout{};
		const Box my_box =
		{
			{(long)global_offset.x,(long)global_offset.y, subdomain_size.z == 1 ? 0L : (long)global_offset.z},
			{(long)subdomain_size.x,(long)subdomain_size.y,(long)subdomain_size.z},
		};
		layout.nprocs = 1;
		layout.rank   = 0;
		layout.user_boxes = {my_box};
#if AC_MPI_ENABLED
		layout.comm = subdomain_size.z == 1 ? xy_plane_communicator : communicator;
		//If acFFTInit has not been called treat as a single process
		if(layout.comm != MPI_COMM_NULL)
		{
			MPI_Comm_size(layout.comm,&layout.nprocs);
			MPI_Comm_rank(layout.comm,&layout.rank);
			layout.user_boxes.resize(layout.nprocs);
			MPI_Allgather(&my_box,sizeof(Box),MPI_BYTE,layout.user_boxes.data(),sizeof(Box),MPI_BYTE,layout.comm);
		}
#endif
		//The global grid is the bounding box of the subdomains, which have to tile it
		Long3 lower = my_box.lower;
		Long3 upper = my_box.lower;
		for(int d = 0; d < 3; ++d) upper[d] += my_box.dims[d];
		size_t total_count = 0;
		for(const Box& box : layout.user_boxes)
		{
			for(int d = 0; d < 3; ++d)
			{
				lower[d] = std::min(lower[d],box.lower[d]);
				upper[d] = std::max(upper[d],box.lower[d]+box.dims[d]);
			}
			total_count += box_count(box);
		}
		for(int d = 0; d < 3; ++d) layout.global_dims[d] = upper[d]-lower[d];
		if(total_count != (size_t)(layout.global_dims[0]*layout.global_dims[1]*layout.global_dims[2]))
		{
			fprintf(stderr,"%s: the subdomains of the processes do not tile the global domain!\n",name);
			fflush(stderr);
			exit(EXIT_FAILURE);
		}
		//Work relative to the origin of the global grid
		for(Box& box : layout.user_boxes)
			for(int d = 0; d < 3; ++d) box.lower[d] -= lower[d];

		for(int axis = 0; axis < 3; ++axis)
			layout.pencil_boxes[axis] = get_pencils(layout.global_dims,axis,layout.nprocs);

		layouts.emplace(key,std::move(layout));
		return layouts.at(key);
	}

	//Redistributes the data from the layout src_boxes to the layout dst_boxes
	std::vector<T>
	reshape(const DistributedLayout& layout, const std::vector<T>& src, const std::vector<Box>& src_boxes, const std::vector<Box>& dst_boxes) const
	{
		const Box& my_src = src_boxes[layout.rank];
		const Box& my_dst = dst_boxes[layout.rank];
		std::vector<T> dst(box_count(my_dst));
#if AC_MPI_ENABLED
		if(layout.nprocs > 1)
		{
			std::vector<int> send_counts(layout.nprocs), send_displs(layout.nprocs);
			std::vector<int> recv_counts(layout.nprocs), recv_displs(layout.nprocs);
			size_t send_total = 0;
			size_t recv_total = 0;
			for(int i = 0; i < layout.nprocs; ++i)
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
			std::vector<T> send_buf(send_total);
			std::vector<T> recv_buf(recv_total);
			for(int i = 0; i < layout.nprocs; ++i)
			{
				const Box region = intersect(my_src,dst_boxes[i]);
				if(box_count(region) == 0) continue;
				copy_box(src.data(),my_src,&send_buf[send_displs[i]],region,region);
			}
			MPI_Alltoallv(send_buf.data(),send_counts.data(),send_displs.data(),complex_type,
				      recv_buf.data(),recv_counts.data(),recv_displs.data(),complex_type,
				      layout.comm);
			for(int i = 0; i < layout.nprocs; ++i)
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

	//Transforms data, which is laid out as the local subdomain of layout.
	//transform_axis(data,box,axis) does the 1D transforms along axis for data laid out as box, which holds the whole axis.
	//scale(data,factor) multiplies the data by factor
	template <typename TransformAxis, typename Scale>
	void
	transform(const DistributedLayout& layout, std::vector<T>& data, const bool inverse,
		  const TransformAxis& transform_axis, const Scale& scale) const
	{
		const std::vector<Box>* boxes = &layout.user_boxes;
		for(int axis = 0; axis < 3; ++axis)
		{
			if(layout.global_dims[axis] == 1) continue;
			data  = reshape(layout,data,*boxes,layout.pencil_boxes[axis]);
			boxes = &layout.pencil_boxes[axis];
			const Box& box = (*boxes)[layout.rank];
			if(box_count(box) == 0) continue;
			transform_axis(data,box,axis);
		}
		if(boxes != &layout.user_boxes)
			data = reshape(layout,data,*boxes,layout.user_boxes);
		if(!inverse)
			scale(data,AcReal(1.0)/(layout.global_dims[0]*layout.global_dims[1]*layout.global_dims[2]));
	}

private:
	const char* name;
#if AC_MPI_ENABLED
	MPI_Comm communicator          = MPI_COMM_NULL;
	MPI_Comm xy_plane_communicator = MPI_COMM_NULL;
	MPI_Datatype complex_type      = MPI_DATATYPE_NULL;
#endif
	Volume global_offset = (Volume){0,0,0};
	std::map<LayoutKey,DistributedLayout> layouts{};
};
