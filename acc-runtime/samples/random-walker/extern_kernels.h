#define POINTS_PER_THREAD (1)
#define ADVANCEMENTS_PER_RADIUS_UPDATE (1)
__global__ void
solve(const int3 start, const int3 end, VertexBufferArray vba)
{
	bool* exists = gmem_bool_arrays[(int)AC_exists];
	int tid_orig = threadIdx.x + blockIdx.x*POINTS_PER_THREAD*blockDim.x;
	__shared__ AcReal tpb_radius_sum;
	if(threadIdx.x == 0)
	        tpb_radius_sum = 0.0;
	__syncthreads();

	AcReal radius_sum = 0.0;
	const bool even_step =  (vba.kernel_input_params.solve.step_num % 2);

	AcReal global_radius     = even_step ? gmem_real_arrays[(int)global_radius_start][0] : gmem_real_arrays[(int)global_radius_tmp][0];
	AcReal global_radius_out = even_step ? gmem_real_arrays[(int)global_radius_tmp][0] : gmem_real_arrays[(int)global_radius_start][0];
        const AcReal global_radius_2 = (global_radius)*(global_radius);
	#pragma unroll
        for(int point_offset = 0; point_offset < POINTS_PER_THREAD; ++point_offset)
        {
                const int tid = tid_orig + point_offset*blockDim.x;
                double xcoord_local = vba.in[COORDS_X][tid];
                double ycoord_local = vba.in[COORDS_Y][tid];
                #pragma unroll
                for(int i = 0; i< ADVANCEMENTS_PER_RADIUS_UPDATE; ++i)
                {
                        const double randx =  random_uniform(tid); // random number for x dir movement
                        const double randy =  random_uniform(tid); // random number for y dir movement

                        xcoord_local += ((randx > 0.75) - (randx < 0.25))*dx;
                        ycoord_local += ((randy > 0.75) - (randy < 0.25))*dy;
                        xcoord_local += ((xcoord_local < -lengthx/ 2.0) - (xcoord_local > lengthx/ 2.0))*lengthx;
                        ycoord_local += ((ycoord_local < -lengthy/ 2.0) - (ycoord_local > lengthy/ 2.0))*lengthy;
                        // if atom is within radius, it is is assimilated to the precipitate which then grows
                //      if ((vba.in[COORDS_X][tid]*vba.in[COORDS_X][tid] + vba.in[COORDS_Y][tid]*vba.in[COORDS_Y][tid] < (*global_radius_in)*(*global_radius_in)) && exists[tid] == 1) {
                        const bool inside = ((vba.in[COORDS_X][tid]*vba.in[COORDS_X][tid]) + (vba.in[COORDS_Y][tid]*vba.in[COORDS_Y][tid]) < global_radius_2);
                        if (inside && exists[tid] == 1) { // if atom is within radius, it is is assimilated to the precipitate which then grows
                //      if ((vba.in[COORDS_X][tid]*vba.in[COORDS_X][tid] + vba.in[COORDS_Y][tid]*vba.in[COORDS_Y][tid] < (*global_radius_in)*(*global_radius_in)) && exists[tid] == 1) {
                            exists[tid] = 0;
                            // annulus of precipitate increases the same amount as the area of atom
                            const double  aeq = AC_REAL_PI; // coefficients of quadratic equation aeq*deltar^2+beq*deltar+c=0
                            const double  beq = 2 * AC_REAL_PI* global_radius;
                            //double  deltar = (-beq + sqrt(pow(beq, 2.0) - 4 * aeq * ceq)) / (2 * aeq); // solve deltar from quadratic eq.
                            const double  deltar = (-beq + sqrt(beq*beq - 4 * aeq * ceq)) / (2 * aeq); // solve deltar from quadratic eq.
                            //radius += deltar; // increase precipitate radius
                            radius_sum += deltar;
                        }
                }
                vba.out[COORDS_X][tid]  = xcoord_local;
		vba.out[COORDS_Y][tid]  = ycoord_local;
        }
	if(radius_sum != 0.0)
                atomicAdd(&tpb_radius_sum, radius_sum);
        __syncthreads();
        if(threadIdx.x == 0 && tpb_radius_sum != 0.0)
                atomicAdd(&global_radius_out, tpb_radius_sum);
}

