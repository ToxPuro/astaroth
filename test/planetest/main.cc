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
#include "user_builtin_non_scalar_constants.h"

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

#define DER2_3 (1. / 90.)
#define DER2_2 (-3. / 20.)
#define DER2_1 (3. / 2.)
#define DER2_0 (-49. / 18.)

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))
#define NUM_INTEGRATION_STEPS (100)

#include "math_utils.h"
#include "user_constants.h"

const int npointsx_grid = 100;
const int npointsy_grid = 100;
#if TWO_D == 0
const int npointsz_grid = 100;
#else
const int npointsz_grid = 1;
#endif

const int npointsx = npointsx_grid;
const int npointsy = npointsy_grid;
const int npointsz = npointsz_grid;

const int mpointsx = npointsx + 2*NGHOST;
const int mpointsy = npointsy + 2*NGHOST;
#if TWO_D == 0
const int mpointsz = npointsz + 2*NGHOST;
#else
const int mpointsz = npointsz;
#endif

//Triangle with all angles 60 degree
constexpr AcReal a=Lengthscale/(2.0*npointsx_grid); //length of one side of a triangle
//Workaround for NVIDIA sqrt not being constexpr
constexpr AcReal sqrt_3 = 1.7320508075688772;
constexpr AcReal h=sqrt_3/2*a; //height of the triangle
//height of the triangle
constexpr AcReal d=a;
constexpr AcReal totalheight=npointsy_grid*h;
constexpr AcReal totalwidth=npointsx_grid*a;
constexpr AcReal totaldepth =npointsz_grid*d;


static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
}
int 
IDX_WITH_HALO(const int i, const int j, const int k)
{
	return i + mpointsx*j + mpointsx*mpointsy*k;
}

int 
IDX_COMP_DOMAIN(const int i, const int j, const int k)
{
	const auto ghosts = acDeviceGetLocalConfig(acGridGetDevice())[AC_nmin];
	return IDX_WITH_HALO(ghosts.x+i, ghosts.y+j, ghosts.z+k);
}


int
main(void)
{

    MPI_Init(NULL,NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
#if AC_RUNTIME_COMPILATION
    if(pid == 0)
    {
    	AcReal real_arr[4];
    	int int_arr[2];
    	bool bool_arr[2] = {false,true};
    	for(int i = 0; i < 4; ++i)
    		real_arr[i] = -i;
    	for(int i = 0; i < 2; ++i)
    		int_arr[i] = i;
    	AcCompInfo info = acInitCompInfo();
    	acLoadCompInfo(AC_lspherical_coords,true,&info);
    	acLoadCompInfo(AC_runtime_int,1,&info);
    	acLoadCompInfo(AC_runtime_real,0.12345,&info);
    	acLoadCompInfo(AC_runtime_real3,{0.12345,0.12345,0.12345},&info);
    	acLoadCompInfo(AC_runtime_int3,{0,1,2},&info);
    	acLoadCompInfo(AC_runtime_real_arr,real_arr,&info);
    	acLoadCompInfo(AC_runtime_int_arr,int_arr,&info);
    	acLoadCompInfo(AC_runtime_bool_arr,bool_arr,&info);
    	const char* build_str = "-DUSE_HIP=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON";
    	acCompile(build_str,info);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    acLoadLibrary();
    acLoadUtils();
#endif
    atexit(acAbort);
    int retval = 0;




    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    info[AC_ds] = {a,h,d};
    const int max_devices = 2 * 2 * 4;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d). Please modify "
                "mpitest/main.cc to use a larger mesh.\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetMeshDims(npointsx_grid, npointsy_grid, npointsz_grid, &info);
    //acSetMeshDims(44, 44, 44, &info);
    //

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);

    // Load/Store
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        const AcResult res = acVerifyMesh("Load/Store", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);

    auto initialize = acGetDSLTaskGraph(AC_initialize);
    auto solve = acGetDSLTaskGraph(AC_solve);

    // Boundconds
    if (pid == 0)
        acHostMeshRandomize(&model);


    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_ALL);

    auto periodic = acGetDSLTaskGraph(bcs);
    acGridExecuteTaskGraphBase(periodic,1,true);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Periodic boundconds", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);
    acGridExecuteTaskGraph(initialize,1);
    acGridSynchronizeStream(STREAM_ALL);

    acGridExecuteTaskGraph(periodic,1);
    acGridSynchronizeStream(STREAM_ALL);

    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acHostMeshApplyPeriodicBounds(&candidate);
    std::vector<AcReal> y(npointsx_grid);
    acGridSynchronizeStream(STREAM_ALL);

    for (int i = 0; i < nsteps; ++i)
    {
	    acGridExecuteTaskGraph(solve, 1);
	    acGridSynchronizeStream(STREAM_ALL);
    }
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acGridSynchronizeStream(STREAM_ALL);
    acHostMeshApplyPeriodicBounds(&candidate);

    //TP: have to make a really imprecise test since periodic bcs cause drift from the actual analytical solution
    //TP: one can generate plot of comparison of the analytical solution compared to the numerical one with plot.py
    AcReal epsilon  = pow(10.0,-1.0);
    auto relative_diff = [](const auto a_val, const auto b_val)
    {
            const auto abs_diff = fabs(a_val-b_val);
            return  abs_diff/a_val;
    };
    auto in_eps_threshold = [&](const auto a_val, const auto b_val)
    {
            return relative_diff(a_val,b_val) < epsilon;
    };


    if(pid  == 0)
    {
            std::vector<AcReal> x(npointsx_grid);
            std::vector<AcReal> u(npointsx_grid);
            std::vector<AcReal> analytical(npointsx_grid);
	    FILE* fp_a = fopen("a.dat","w");
	    FILE* fp_u = fopen("u.dat","w");
	    FILE* fp_x = fopen("x.dat","w");
	    const int end = npointsx_grid;
            for(int i = 0; i< end; ++i)
            {
		const int idx = IDX_COMP_DOMAIN(i, npointsy_grid/2, npointsz_grid/2);
            	x[i] = candidate.vertex_buffer[AC_COORDS.x][idx];
                u[i] = candidate.vertex_buffer[U][idx];
                analytical[i] = candidate.vertex_buffer[SOLUTION][idx];

		fprintf(fp_a,"%.14e",analytical[i]);
		if(i < end -1 ) fprintf(fp_a,"%s",",");

		fprintf(fp_u,"%.14e",u[i]);
		if(i < end -1 ) fprintf(fp_u,"%s",",");

		fprintf(fp_x,"%.14e",x[i]);
		if(i < end -1 ) fprintf(fp_x,"%s",",");
            }
	    fclose(fp_a);
	    fclose(fp_x);
	    fclose(fp_u);
	    bool correct = true;
	    for(size_t i = npointsx_grid/4; i < (3*npointsx_grid)/4; ++i)
	    {
				const int idx = IDX_COMP_DOMAIN(i, npointsy_grid/2, npointsz_grid/2);
				const AcReal u_val = candidate.vertex_buffer[U][idx];
				const AcReal a_val = candidate.vertex_buffer[SOLUTION][idx];
				const bool in_eps = in_eps_threshold(u_val,a_val);
				correct &= in_eps;
				if(!in_eps) printf("U,A: %14e,%14e\n",u_val,a_val);
	    }
	    if(!correct) retval = AC_FAILURE;
    }

    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "MPITEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

    return EXIT_SUCCESS;
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
