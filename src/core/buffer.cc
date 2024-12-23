#include "acc_runtime.h"
static size_t
get_size_from_dim(const int dim, const Volume dims)
{
    	const auto size   = dim == X_ORDER_INT ? dims.x :
        		    dim == Y_ORDER_INT ? dims.y :
        		    dims.z;
        return size;
}
AcShape
acGetTransposeBufferShape(const AcMeshOrder order, const Volume dims)
{
	const int first_dim  = order % N_DIMS;
	const int second_dim = (order/N_DIMS) % N_DIMS;
	const int third_dim  = ((order/N_DIMS)/N_DIMS)  % N_DIMS;
	return (AcShape){
		get_size_from_dim(first_dim,dims),
		get_size_from_dim(second_dim,dims),
		get_size_from_dim(third_dim,dims),
		1
	};
}
AcShape
acGetReductionShape(const AcProfileType type, const AcMeshDims dims)
{
	const AcShape order_size = acGetTransposeBufferShape(
			acGetMeshOrderForProfile(type),
			dims.m1
			);
	if(type & ONE_DIMENSIONAL_PROFILE)
	{
		return
		{
			order_size.x - 2*NGHOST,
			order_size.y - 2*NGHOST,
			order_size.z - (type != PROFILE_Z)*2*NGHOST,
			order_size.w
		};
	}
	else if(type & TWO_DIMENSIONAL_PROFILE)
		return
		{
			order_size.x - 2*NGHOST,
			order_size.y,
			order_size.z,
			order_size.w
		};
	return order_size;
}
static Volume
get_volume_from_shape(const AcShape shape)
{
	return {shape.x,shape.y,shape.z};
}

AcBuffer
acBufferCreate(const AcShape shape, const bool on_device)
{
    AcBuffer buffer    = {.data = NULL, .count = acShapeSize(shape), .on_device = on_device, .shape = shape};
    const size_t bytes = sizeof(buffer.data[0]) * buffer.count;
    if (buffer.on_device) {
        ERRCHK_CUDA_ALWAYS(cudaMalloc((void**)&buffer.data, bytes));
    }
    else {
        buffer.data = (AcReal*)malloc(bytes);
    }
    ERRCHK_ALWAYS(buffer.data);
    return buffer;
}

void
acBufferDestroy(AcBuffer* buffer)
{
    if (buffer->on_device)
    {
        ERRCHK_CUDA_ALWAYS(cudaFree(buffer->data));
    }
    else
        free(buffer->data);
    buffer->data  = NULL;
    buffer->count = 0;
}

AcResult
acBufferMigrate(const AcBuffer in, AcBuffer* out)
{
    cudaMemcpyKind kind;
    if (in.on_device) {
        if (out->on_device)
            kind = cudaMemcpyDeviceToDevice;
        else
            kind = cudaMemcpyDeviceToHost;
    }
    else {
        if (out->on_device)
            kind = cudaMemcpyHostToDevice;
        else
            kind = cudaMemcpyHostToHost;
    }

    ERRCHK_ALWAYS(in.count == out->count);
    ERRCHK_CUDA_ALWAYS(cudaMemcpy(out->data, in.data, sizeof(in.data[0]) * in.count, kind));
    return AC_SUCCESS;
}

AcBuffer
acBufferCopy(const AcBuffer in, const bool on_device)
{
    AcBuffer cpu_buffer = acBufferCreate(in.shape, on_device);
    acBufferMigrate(in,&cpu_buffer);
    return cpu_buffer;
}

AcBuffer
acBufferRemoveHalos(const AcBuffer buffer_in, const int3 halo_sizes, const cudaStream_t stream)
{
	const AcShape res_shape = 
	{
		buffer_in.shape.x - 2*halo_sizes.x,
		buffer_in.shape.y - 2*halo_sizes.y,
		buffer_in.shape.z - 2*halo_sizes.z,
		1
	};	

	const AcShape in_offset = 
	{
		as_size_t(halo_sizes.x),
		as_size_t(halo_sizes.y),
		as_size_t(halo_sizes.z),
		0,
	};	

	const AcShape out_offset = 
	{
		0,
		0,
		0,
		0,
	};	

	const AcShape in_shape    = buffer_in.shape;
	const AcShape block_shape = buffer_in.shape;

	AcBuffer dst = acBufferCreate(res_shape,true);
    	acReindex(stream,buffer_in.data, in_offset, in_shape, dst.data, out_offset , dst.shape, block_shape);
	return dst;
}
static
AcBuffer
acBufferCreateTransposed(const AcBuffer src, const AcMeshOrder order)
{
	const Volume dims = get_volume_from_shape(src.shape);
	return acBufferCreate(acGetTransposeBufferShape(order,dims),true);
}

AcBuffer
acTransposeBuffer(const AcBuffer src, const AcMeshOrder order, const cudaStream_t stream)
{
	const Volume dims = get_volume_from_shape(src.shape);
	AcBuffer dst = acBufferCreateTransposed(src,order);
	acTranspose(order,src.data,dst.data,dims,stream);
	return dst;
}
