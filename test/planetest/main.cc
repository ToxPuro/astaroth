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

const int npointsx_grid = 64*2;
const int npointsy_grid = 64*2;
const int npointsz_grid = 64*2;

const int npointsx = npointsx_grid;
const int npointsy = npointsy_grid;
const int npointsz = npointsz_grid;

const int mpointsx = npointsx + NGHOST;
const int mpointsy = npointsy + NGHOST;
const int mpointsz = npointsz + NGHOST;

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
	return IDX_WITH_HALO(NGHOST+i, NGHOST+j, NGHOST+k);
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

    info.real_params[AC_dsx] = a;
    info.real_params[AC_dsy] = h;
    info.real_params[AC_dsz] = d;

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

    std::vector<Field> all_fields {};
    for(int i = 0; i < NUM_VTXBUF_HANDLES; ++i) all_fields.push_back((Field)i);
    AcTaskGraph* initialize = acGridBuildTaskGraph({ 
		    acHaloExchange(all_fields),
		    acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
		    acCompute(KERNEL_initial_condition, all_fields)
    });

    AcTaskGraph* integrate = acGridBuildTaskGraph({ 
		    acHaloExchange(all_fields),
		    acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields),
		    acCompute(KERNEL_singlepass_solve, all_fields)
    });

    AcTaskGraph* periodic = acGridBuildTaskGraph({ 
		    acBoundaryCondition(BOUNDARY_XYZ, BOUNDCOND_PERIODIC, all_fields)
    });

    // Boundconds
    if (pid == 0)
        acHostMeshRandomize(&model);


    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_ALL);

    acGridPeriodicBoundconds(STREAM_ALL);
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
    acGridExecuteTaskGraph(initialize, 1);
    acGridSynchronizeStream(STREAM_ALL);

    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acGridSynchronizeStream(STREAM_ALL);

    auto IDX = [&](const int i, const int j, const int k)
    {
	    return acVertexBufferIdx(i,j,k,model.info);
    };

    auto calc_derxx = [&](const int i, const int j, const int k, const AcReal* arr, const AcReal dsx)
    {
	    const AcReal inv_dsx2 = (1.0/dsx)*(1.0/dsx);
	    return
		    arr[IDX(i-3,j,k)]*(inv_dsx2*DER2_3) +
		    arr[IDX(i-2,j,k)]*(inv_dsx2*DER2_2) +
		    arr[IDX(i-1,j,k)]*(inv_dsx2*DER2_1) +
		    arr[IDX(i,j,k)]  *(inv_dsx2*DER2_0) +
		    arr[IDX(i+1,j,k)]*(inv_dsx2*DER2_1) +
		    arr[IDX(i+2,j,k)]*(inv_dsx2*DER2_2) +
		    arr[IDX(i+3,j,k)]*(inv_dsx2*DER2_3);

    };

    auto calc_deryy = [&](const int i, const int j, const int k, const AcReal* arr, const AcReal dsy)
    {
	    const AcReal inv_dsy2 = (1.0/dsy)*(1.0/dsy);
	    return
		    arr[IDX(i,j-3,k)]*(inv_dsy2*DER2_3) +
		    arr[IDX(i,j-2,k)]*(inv_dsy2*DER2_2) +
		    arr[IDX(i,j-1,k)]*(inv_dsy2*DER2_1) +
		    arr[IDX(i,j,k)]  *(inv_dsy2*DER2_0) +
		    arr[IDX(i,j+1,k)]*(inv_dsy2*DER2_1) +
		    arr[IDX(i,j+2,k)]*(inv_dsy2*DER2_2) +
		    arr[IDX(i,j+3,k)]*(inv_dsy2*DER2_3);

    };

    auto calc_derzz = [&](const int i, const int j, const int k, const AcReal* arr, const AcReal dsz)
    {
	    const AcReal inv_dsz2 = (1.0/dsz)*(1.0/dsz);
	    return
		    arr[IDX(i,j,k-3)]*(inv_dsz2*DER2_3) +
		    arr[IDX(i,j,k-2)]*(inv_dsz2*DER2_2) +
		    arr[IDX(i,j,k-1)]*(inv_dsz2*DER2_1) +
		    arr[IDX(i,j,k)]  *(inv_dsz2*DER2_0) +
		    arr[IDX(i,j,k+1)]*(inv_dsz2*DER2_1) +
		    arr[IDX(i,j,k+2)]*(inv_dsz2*DER2_2) +
		    arr[IDX(i,j,k+3)]*(inv_dsz2*DER2_3);

    };


    const int nx_min = info.int_params[AC_nx_min];
    const int ny_min = info.int_params[AC_ny_min];
    const int nz_min = info.int_params[AC_nz_min];

    const int nx_max = info.int_params[AC_nx_max];
    const int ny_max = info.int_params[AC_ny_max];
    const int nz_max = info.int_params[AC_nz_max];
    AcReal* update = (AcReal*)malloc(sizeof(AcReal)*acVertexBufferSize(info));
    for (int i = 0; i < nsteps; ++i)
    {
	    acGridExecuteTaskGraph(integrate, 1);
	    acGridSynchronizeStream(STREAM_ALL);
	    //
    	    //acHostMeshApplyPeriodicBounds(&candidate);
            //for (int k = nz_min; k < nz_max; ++k) {
            //  for (int j = ny_min; j < ny_max; ++j) {
            //      for (int i = nx_min; i < nx_max; ++i) {
            //    	const int index = IDX(i,j,k);
            //    	const AcReal derxx = calc_derxx(i,j,k,candidate.vertex_buffer[U],model.info.real_params[AC_dsx]);
            //    	const AcReal deryy = calc_deryy(i,j,k,candidate.vertex_buffer[U],model.info.real_params[AC_dsy]);
            //    	const AcReal derzz = calc_derzz(i,j,k,candidate.vertex_buffer[U],model.info.real_params[AC_dsz]);
            //    	update[index] = D*(derxx + deryy + derzz);
            //      }
            //  }
            //}
            //for (int k = nz_min; k < nz_max; ++k) {
            //  for (int j = ny_min; j < ny_max; ++j) {
            //      for (int i = nx_min; i < nx_max; ++i) {
            //    	const int index = IDX(i,j,k);
            //    	candidate.vertex_buffer[U][index] += dt*update[index];
            //      }
            //  }
            //}
    }
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acGridSynchronizeStream(STREAM_ALL);
    acHostMeshApplyPeriodicBounds(&candidate);


    if(pid  == 0)
    {
	    const int npointsx_grid = candidate.info.int_params[AC_nxgrid];
	    const int npointsy_grid = candidate.info.int_params[AC_nygrid];
	    const int npointsz_grid = candidate.info.int_params[AC_nzgrid];
            std::vector<AcReal> x(npointsx_grid);
            std::vector<AcReal> u(npointsx_grid);
            std::vector<AcReal> a(npointsx_grid);
	    FILE* fp_a = fopen("a.dat","w");
	    FILE* fp_u = fopen("u.dat","w");
	    FILE* fp_x = fopen("x.dat","w");
            for(int i = 0; i< npointsx_grid; ++i)
            {
		const int idx = IDX_COMP_DOMAIN(i, npointsy_grid/2,npointsz_grid/2);
            	x[i] = candidate.vertex_buffer[COORDS_X][idx];
                u[i] = candidate.vertex_buffer[U][idx];
                a[i] = candidate.vertex_buffer[SOLUTION][idx];

		fprintf(fp_a,"%.14e",a[i]);
		if(i < npointsx_grid -1 ) fprintf(fp_a,"%s",",");

		fprintf(fp_u,"%.14e",u[i]);
		if(i < npointsx_grid -1 ) fprintf(fp_u,"%s",",");

		fprintf(fp_x,"%.14e",x[i]);
		if(i < npointsx_grid -1 ) fprintf(fp_x,"%s",",");

            }
	    fclose(fp_a);
	    fclose(fp_x);
	    fclose(fp_u);
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
