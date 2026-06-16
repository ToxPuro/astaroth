/*
    Copyright (C) 2014-2021, Johannes Pekkila, Miikka Vaisala.

    This file is part of Astaroth.

    Astaroth is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Astaroth is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.
*/
/**
    Running: mpirun -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"
#include "user_constants.h"

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>


static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
}
double
drand()
{
	return (double)(rand()) / (double)(rand());
}

#include "../../stdlib/grid.h"

int
main(void)
{
    atexit(acAbort);

    int nprocs, pid;
    MPI_Init(NULL,NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("poisson.conf", &info);

    acPushToConfig(info,AC_coordinate_system,AC_SPHERICAL_COORDINATES);
    acPushToConfig(info,AC_MPI_comm_strategy,AC_MPI_COMM_STRATEGY_DUP_WORLD);
    acPushToConfig(info,AC_proc_mapping_strategy,AC_PROC_MAPPING_STRATEGY_MORTON);
    acPushToConfig(info,AC_decompose_strategy,AC_DECOMPOSE_STRATEGY_MORTON);
    info.comm->handle = MPI_COMM_WORLD;
    acUpdateDecompositionParams(&info);
    //info[AC_coordinate_system] = AC_CARTESIAN_COORDINATES;
    //TP: this is said to be a good empirical value based on the eigenspectrum of the 3d laplacian matrix

    const int max_devices = 8;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetGridMeshDims(info[AC_ngrid].x,info[AC_ngrid].y,info[AC_ngrid].z, &info);
    acPushToConfig(info,AC_ds,
    (AcReal3){
	    1.0/info[AC_ngrid].x,
	    AC_REAL_PI/info[AC_ngrid].y,
	    (2.0*AC_REAL_PI)/info[AC_ngrid].z
    });
    acPushToConfig(info,AC_first_gridpoint,
    (AcReal3){
            4.0*info[AC_ds].x,
	    0.5*info[AC_ds].y,
	    0.5*info[AC_ds].z
    });
    acHostUpdateParams(&info);
    const int3 decomp = info[AC_domain_decomposition];
    acSetLocalMeshDims(info[AC_ngrid].x/decomp.x,info[AC_ngrid].y/decomp.y,info[AC_ngrid].z/decomp.z, &info);

    //ac_compute_power_law_mapping_x(&info,info[AC_power_law_mapping_exponent]);
    /**
    ac_compute_inv_r(&info);
    ac_compute_r(&info);
    **/

    ac_compute_inv_sin_theta(&info);
    ac_compute_cot_theta(&info);
    ac_compute_theta(&info);
    ac_compute_phi(&info);
    ac_compute_spherical_harmonics(&info);
    ac_compute_sin_theta(&info);
    ac_compute_sin_phi(&info);
    ac_compute_cos_phi(&info);
    ac_compute_phi(&info);
    ac_compute_exp_mapping_x(&info);

    const AcReal R = info[AC_r][info[AC_charge_radius_points]] + 0.5*info[AC_ds].x;
    const AcReal R_max = info[AC_r][info[AC_nlocal].x+NGHOST-1] + 0.5*info[AC_ds].x;
    const AcReal R_min = info[AC_r][NGHOST] - 0.5*info[AC_ds].x;
    //const AcReal R_max = info[AC_r][info[AC_nlocal].x+NGHOST-1] + 0.0*info[AC_ds].x;
    //const AcReal R_min = info[AC_r][NGHOST] - 0.0*info[AC_ds].x;
    /**
    long double res_x = 0.0;
    AcReal real_res_x = 0.0;
    printf("AC_nx: %d\n",info[AC_nlocal].x);
    for(int x = NGHOST; x < info[AC_nlocal].x+NGHOST; ++x)
    {
	    //if(x <= info[AC_charge_radius_points])
	    //{
		    const long double r = (long double)info[AC_r][x];
		    AcReal w = 1.0;
		    const int x_local = x-NGHOST;
		    if(x_local == 0 || x_local == info[AC_nlocal].x-1) w = (1.0/3.0);
		    else if(x_local % 2 == 1) w = 4.0/3.0;
		    else w = 2.0/3.0;

		    if(x_local == 0) w += 17.0/24.0;
		    if(x_local == 1) w -= 7.0/24.0;
		    if(x_local == 2) w += 1.0/12.0;

		    if(x_local == info[AC_nlocal].x-1) w += 17.0/24.0;
		    if(x_local == info[AC_nlocal].x-2) w -= 7.0/24.0;
		    if(x_local == info[AC_nlocal].x-3) w += 1.0/12.0;
		

		    const AcReal local_res = r*r*info[AC_ds].x;
		    res_x += r*r*info[AC_ds].x;
		    real_res_x += r*r*info[AC_ds].x*w;
	    //}
    }
    AcReal lower = 0.0;
    AcReal simpsons = 0.0;
    for(int y = NGHOST; y < info[AC_nlocal].y+NGHOST; ++y)
    {
	    //if(x <= info[AC_charge_radius_points])
	    //{
		    const AcReal theta = (long double)info[AC_theta][y];
		    const AcReal sin_theta = (long double)info[AC_sin_theta][y];
		    AcReal w = 1.0;
		    const int y_local = y-NGHOST;
		    if(y_local == 0 || y_local == info[AC_nlocal].y-1) w = (1.0/3.0);
		    else if(y_local % 2 == 1) w = 4.0/3.0;
		    else w = 2.0/3.0;

		    if(y_local == 0) w += 17.0/24.0;
		    if(y_local == 1) w -= 7.0/24.0;
		    if(y_local == 2) w += 1.0/12.0;

		    if(y_local == info[AC_nlocal].y-1) w += 17.0/24.0;
		    if(y_local == info[AC_nlocal].y-2) w -= 7.0/24.0;
		    if(y_local == info[AC_nlocal].y-3) w += 1.0/12.0;
		

		    lower += sin_theta*info[AC_ds].y;
		    simpsons += w*sin_theta*info[AC_ds].y;
	    //}
    }
    AcReal z_lower = 0.0;
    AcReal z_simpsons = 0.0;
    for(int z = NGHOST; z < info[AC_nlocal].z+NGHOST; ++z)
    {
	    //if(x <= info[AC_charge_radius_points])
	    //{
		    const AcReal phi = (long double)info[AC_phi][z];
		    AcReal w = 1.0;
		    const int z_local = z-NGHOST;
		    if(z_local == 0 || z_local == info[AC_nlocal].z-1) w = (1.0/3.0);
		    else if(z_local % 2 == 1) w = 4.0/3.0;
		    else w = 2.0/3.0;

		    if(z_local == 0) w += 17.0/24.0;
		    if(z_local == 1) w -= 7.0/24.0;
		    if(z_local == 2) w += 1.0/12.0;

		    if(z_local == info[AC_nlocal].z-1) w += 17.0/24.0;
		    if(z_local == info[AC_nlocal].z-2) w -= 7.0/24.0;
		    if(z_local == info[AC_nlocal].z-3) w += 1.0/12.0;
		

		    z_lower += phi*info[AC_ds].z;
		    z_simpsons += w*phi*info[AC_ds].z;
	    //}
    }
    const AcReal R3 = R*R*R;
    const AcReal R_max2 = R_max*R_max;
    const AcReal R_max3 = R_max*R_max*R_max;
    const AcReal R_min3 = R_min*R_min*R_min;
    const AcReal R_min2 = R_min*R_min;
    fprintf(stderr,"Full sphere should be: %14e,%14e\n",(R_max3-R_min3)/3.0,info[AC_ngrid].x*info[AC_ds].x);
    fprintf(stderr,"Hollow sphere should be: %14e,%14e,%14e\n",(R_max2-R_min2)/2.0,AcReal(res_x),real_res_x);
    fprintf(stderr,"Lower: %14e\n",lower);
    fprintf(stderr,"Simpsons: %14e\n",simpsons);
    fprintf(stderr,"Phi Lower: %14e\n",z_lower);
    fprintf(stderr,"Phi Simpsons: %14e\n",z_simpsons);
    fprintf(stderr,"Test : %14e\n",info[AC_nlocal].y*info[AC_ds].y);
    //exit(EXIT_FAILURE);
    **/

    /**
    AcReal res_z = 0.0;
    for(int z = NGHOST; z < info[AC_nlocal].z+NGHOST; ++z)
    {
		    const AcReal local_res = info[AC_ds].z;
		    res_z += local_res;
    }
    long double res = 0.0;
    AcReal res_y = 0.0;
    for(int y = NGHOST; y < info[AC_nlocal].y + NGHOST; ++y)
    {
        AcReal weight = 1.0;
        if (y == NGHOST || y == info[AC_nlocal].y + NGHOST - 1)
            weight = 0.5;
        res += info[AC_sin_theta][y] * info[AC_ds].y * weight;
	res_y += info[AC_ds].y;
    }
    fprintf(stderr, "Should be 2.0: %14e\n", AcReal(res));
    fprintf(stderr,"Should be length: %14e\n",res_x);
    fprintf(stderr,"Should be PI: %14e\n",res_y);
    fprintf(stderr,"First and last phi: %14e,%14e\n",info[AC_phi][NGHOST],info[AC_phi][info[AC_nlocal].z + NGHOST-1]);
    **/
    #if AC_RUNTIME_COMPILATION
    const char* build_str = "-DBUILD_SAMPLES=OFF -DDSL_MODULE_DIR=../../DSL -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DOPTIMIZE_INPUT_PARAMS=ON -DBUILD_ACM=OFF";
    acCompile(build_str,info);
    acLoadLibrary(stdout,info);
    acLoadUtils(stdout,info);
    #endif

    AcMesh model, candidate;
    acHostMeshCreate(info, &model);
    acHostMeshCreate(info, &candidate);
    acHostMeshRandomize(&model);
    acHostMeshRandomize(&candidate);

    acGridInit(info);
    acDeviceSetInput(acGridGetDevice(),AC_SOR_omega,1.8);
    //acPrintMeshInfo(acDeviceGetLocalConfig(acGridGetDevice()));
    //fflush(stdout);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(initcond),1);
    auto calculate_integrals_graph = acGetOptimizedDSLTaskGraph(calculate_integrals);
    for(int i= 0; i < 100; ++i)
    {
      const AcReal start_time = MPI_Wtime();
      acGridExecuteTaskGraph(calculate_integrals_graph,1);
      const AcReal end_time   = MPI_Wtime();
      fprintf(stderr,"Calculating integrals took: %.14e\n",end_time-start_time);
    }
    //TP: does not matter which harmonic we choose since all of them take the same amount of memory
    //
    const Volume nn = acGetLocalNN(info);
    AcReal full_size =    (AcReal)nn.x*nn.y*nn.z*sizeof(AcReal);
    AcReal reduced_size = (AcReal)acGetRealScratchpadSize(AC_upper_negative_MLM[0]);
    printf("Saving a factor of %14e memory for scratchpads for calculating harmonic expansion\n",full_size/reduced_size);
    return 0;
}

#else
int
main(void)
{
    printf("The library was built without MPI support, cannot run mpitest. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLES
