#pragma once

static __global__ void
kernel_pack_data(const VertexBufferArray vba, const int3 vba_start, PackedData packed)
{
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= packed.dims.x || //
        j_packed >= packed.dims.y || //
        k_packed >= packed.dims.z) {
        return;
    }

    const int i_unpacked = i_packed + vba_start.x;
    const int j_unpacked = j_packed + vba_start.y;
    const int k_unpacked = k_packed + vba_start.z;

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +               //
                           j_packed * packed.dims.x + //
                           k_packed * packed.dims.x * packed.dims.y;

    const size_t vtxbuf_offset = packed.dims.x * packed.dims.y * packed.dims.z;

    //#pragma unroll
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        packed.data[packed_idx + i * vtxbuf_offset] = vba.in[i][unpacked_idx];
}

static __global__ void
kernel_unpack_data(const PackedData packed, const int3 vba_start, VertexBufferArray vba)
{
    const int i_packed = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_packed = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_packed = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_packed >= packed.dims.x || //
        j_packed >= packed.dims.y || //
        k_packed >= packed.dims.z) {
        return;
    }

    const int i_unpacked = i_packed + vba_start.x;
    const int j_unpacked = j_packed + vba_start.y;
    const int k_unpacked = k_packed + vba_start.z;

    const int unpacked_idx = DEVICE_VTXBUF_IDX(i_unpacked, j_unpacked, k_unpacked);
    const int packed_idx   = i_packed +               //
                           j_packed * packed.dims.x + //
                           k_packed * packed.dims.x * packed.dims.y;

    const size_t vtxbuf_offset = packed.dims.x * packed.dims.y * packed.dims.z;

    //#pragma unroll
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i)
        vba.in[i][unpacked_idx] = packed.data[packed_idx + i * vtxbuf_offset];
}

static __global__ void
kernel_extract_packed_data(const PackedData old_packed, const int3 offset, const PackedData repacked)
{
    const int i_repacked = threadIdx.x + blockIdx.x * blockDim.x;
    const int j_repacked = threadIdx.y + blockIdx.y * blockDim.y;
    const int k_repacked = threadIdx.z + blockIdx.z * blockDim.z;

    // If within the start-end range (this allows threadblock dims that are not
    // divisible by end - start)
    if (i_repacked >= repacked.dims.x || //
        j_repacked >= repacked.dims.y || //
        k_repacked >= repacked.dims.z) {
        return;
    }

    const int repacked_idx   = i_repacked +               //
                           j_repacked * repacked.dims.x + //
                           k_repacked * repacked.dims.x * repacked.dims.y;

    const size_t repacked_vtxbuf_offset = repacked.dims.x * repacked.dims.y * repacked.dims.z;

    //old indices
    const int i_old_packed = i_repacked + offset.x;
    const int j_old_packed = j_repacked + offset.y;
    const int k_old_packed = k_repacked + offset.z;

    const int old_packed_idx = i_old_packed +
                           j_old_packed * old_packed.dims.x + //
                           k_old_packed * old_packed.dims.x * old_packed.dims.y;

    const size_t old_vtxbuf_offset = old_packed.dims.x * old_packed.dims.y * old_packed.dims.z;

    //kernel_extract_packed_data gets an offset and a dim into a face 
    //Repack the data from the offset into a new PackedData structure
    //

    //#pragma unroll
    for (int i = 0; i < NUM_VTXBUF_HANDLES; ++i){
        repacked.data[repacked_idx + i * repacked_vtxbuf_offset] = old_packed.data[old_packed_idx + i * old_vtxbuf_offset];
    }

}

PackedData
acCreatePackedData(const int3 dims)
{
    PackedData data = {};

    data.dims = dims;

    const size_t bytes = dims.x * dims.y * dims.z * sizeof(data.data[0]) * NUM_VTXBUF_HANDLES;
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data.data, bytes));

#if MPI_USE_CUDA_DRIVER_PINNING
    ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&data.data_pinned, bytes));

    unsigned int flag = 1;
    CUresult retval   = cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                            (CUdeviceptr)data.data_pinned);
    ERRCHK_ALWAYS(retval == CUDA_SUCCESS);
#else
    ERRCHK_CUDA_ALWAYS(cudaMallocHost((void**)&data.data_pinned, bytes));
// ERRCHK_CUDA_ALWAYS(cudaMallocManaged((void**)&data.data_pinned, bytes)); // Significantly
// slower than pinned (38 ms vs. 125 ms)
#endif // USE_CUDA_DRIVER_PINNING

    return data;
}

AcResult
acKernelPackData(const cudaStream_t stream, const VertexBufferArray vba, const int3 vba_start,
                 PackedData packed)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(packed.dims.x / (float)tpb.x),
                   (unsigned int)ceil(packed.dims.y / (float)tpb.y),
                   (unsigned int)ceil(packed.dims.z / (float)tpb.z));

    kernel_pack_data<<<bpg, tpb, 0, stream>>>(vba, vba_start, packed);
    ERRCHK_CUDA_KERNEL();

    return AC_SUCCESS;
}

AcResult
acKernelUnpackData(const cudaStream_t stream, const PackedData packed, const int3 vba_start,
                   VertexBufferArray vba)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(packed.dims.x / (float)tpb.x),
                   (unsigned int)ceil(packed.dims.y / (float)tpb.y),
                   (unsigned int)ceil(packed.dims.z / (float)tpb.z));

    kernel_unpack_data<<<bpg, tpb, 0, stream>>>(packed, vba_start, vba);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

AcResult
acKernelExtractPackedData(const cudaStream_t stream, const PackedData packed, const int3 offset,
                   PackedData extracted)
{
    const dim3 tpb(32, 8, 1);
    const dim3 bpg((unsigned int)ceil(extracted.dims.x / (float)tpb.x),
                   (unsigned int)ceil(extracted.dims.y / (float)tpb.y),
                   (unsigned int)ceil(extracted.dims.z / (float)tpb.z));

    /*
    printf("Packed: (%d,%d,%d), extracted: (%d,%d,%d) -> Offset: (%d,%d,%d)\n",
		    packed.dims.x,packed.dims.y,packed.dims.z,
		    extracted.dims.x,extracted.dims.y,extracted.dims.z,
		    offset.x,offset.y,offset.z);
*/
    kernel_extract_packed_data<<<bpg, tpb, 0, stream>>>(packed, offset, extracted);
    ERRCHK_CUDA_KERNEL();
    return AC_SUCCESS;
}

#include <sstream>
#include <iostream>

void
acPrintPackedData(const PackedData packed, const int print_src, int from, const int to,const int through, const int3 halo_coords, const int real)
{
    std::stringstream buf;
    std::stringstream databuf;
    buf << std::endl << "{ \"printed_by\":"<< print_src <<",";
    buf << "\"type\":" << (real != 0? "\"Original\"": "\"Extracted\"");
    buf << ",\"from\":" << from << ",\"to\":" << to << ",\"through\":"<< through << ",\"coords\": [" << halo_coords.x << "," << halo_coords.y << "," << halo_coords.z << "],";
    buf << "\"dims\": [" << packed.dims.x << "," << packed.dims.y << "," << packed.dims.z << "]";
    buf << ",\"data\":[";
    for (int i = 0; i < packed.dims.x;i++)
        for(int j = 0; j < packed.dims.y;j++)
            for(int k = 0; k < packed.dims.z;k++)
            	for(int l = 0; l < NUM_VTXBUF_HANDLES;l++){
			if (i != 0|| j != 0 || k!= 0 || l!= 0){
				buf << ",";
			}
			buf << "\"" << packed.data_pinned[i + j*packed.dims.x + k*packed.dims.x*packed.dims.y + l*packed.dims.x*packed.dims.y*packed.dims.z] << "\"";
		}
    buf << "]";
    buf << "}" << std::endl;
    std::string out = buf.str();
    std::cout << out;
}


