#define POINTS_PER_THREAD (2)
__global__ void
solve(const int3 start, const int3 end, VertexBufferArray vba)
{
	int tid_orig = threadIdx.x + blockIdx.x*POINTS_PER_THREAD*blockDim.x;
	__shared__ AcReal tpb_radius_sum;
	if(threadIdx.x == 0)
	        tpb_radius_sum = 0.0;
	__syncthreads();

	AcReal radius_sum = 0.0;
        //const AcReal global_radius = (*global_radius_in);
        //const AcReal global_radius_2 = (global_radius)*(global_radius);

	//#pragma unroll


}

