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

#define NUM_INTEGRATION_STEPS (1)
#define ROW_MAJOR_ORDER (0)

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

#if ROW_MAJOR_ORDER
#define TWO_D_ARR(i,j) [i][j]
#else
#define TWO_D_ARR(i,j) [j][i]
#endif

#if ROW_MAJOR_ORDER
#define THREE_D_ARR(i,j,k) [i][j][k]
#else
#define THREE_D_ARR(i,j,k) [k][j][i]
#endif

#if ROW_MAJOR_ORDER
#define FOUR_D_ARR(i,j,k,l) [i][j][k][l]
#else
#define FOUR_D_ARR(i,j,k,l) [l][k][j][i]
#endif

int
main(void)
{
    atexit(acAbort);
    int retval = 0;

    ac_MPI_Init();

    int nprocs, pid;
    MPI_Comm_size(acGridMPIComm(), &nprocs);
    MPI_Comm_rank(acGridMPIComm(), &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info[AC_host_has_row_memory_order] = ROW_MAJOR_ORDER;

    const int max_devices = 1;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    constexpr int nx = AC_nx_const;
    constexpr int ny = AC_ny_const;
    [[maybe_unused]] constexpr int nz = 4*9;
    //constexpr int nx = 2*9;
    //constexpr int ny = nx;
    //[[maybe_unused]] constexpr int nz = nx;
    acSetMeshDims(nx, ny, nz, &info);

    constexpr int mx = nx + 2*NGHOST;
    constexpr int my = ny + 2*NGHOST;
    constexpr int mz = nz + 2*NGHOST;
    //acSetMeshDims(44, 44, 44, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    const int test_int_arr[3] = {-3, 50, 42};
    AcReal test_arr[6];
    for(int i = 0; i < 6; ++i)
	    test_arr[i] = drand();
    const AcReal test_arr_2[3] = {1.0, -2.0, 3.0};
    info[AC_test_arr] = (AcReal*)test_arr;
    //these are read from config
    //info.int_arrays[AC_test_int_arr] = (int*)test_int_arr;
    //info.real_arrays[AC_test_arr_2] = (AcReal*)test_arr_2;
    int   global_arr[nx];
    float float_arr[nx];
    AcReal twoD_real_arr TWO_D_ARR(nx,ny);
    AcReal threeD_real_arr THREE_D_ARR(mx,my,mz);
    float  fourD_float_arr FOUR_D_ARR(mx,my,mz,3);

    for(int i = 0; i < nx; ++i)
    {
		    global_arr[i] = rand();
		    float_arr[i]  = (float)(1.0*rand())/(float)RAND_MAX;
    }
    for(int j = 0; j < ny; ++j)
    	for(int i = 0; i < nx; ++i)
		twoD_real_arr TWO_D_ARR(i,j) = (1.0*rand())/(AcReal)RAND_MAX;

    for(int k = 0; k < mz; ++k)
    	for(int j = 0; j < my; ++j)
    		for(int i = 0; i < mx; ++i)
		{
			threeD_real_arr THREE_D_ARR(i,j,k)  = (1.0*rand())/(AcReal)RAND_MAX;
			for(int l = 0; l < 3; ++l)
				fourD_float_arr FOUR_D_ARR(i,j,k,l) = (float)(1.0*rand())/(float)RAND_MAX;
		}
    info[AC_global_arr] = global_arr;
    info[AC_float_arr] = float_arr;
    info[AC_2d_reals]  = &twoD_real_arr[0][0];
    info[AC_2d_reals_dims_from_config]  = &twoD_real_arr[0][0];
    info[AC_3d_reals]  = &threeD_real_arr[0][0][0];
    info[AC_4d_float_arr]  = &fourD_float_arr[0][0][0][0];
    info[AC_dconst_int] = nx-1-info[AC_nmin].x;
    info[AC_4d_float_arr_out]  = &fourD_float_arr[0][0][0][0];
    acGridInit(info);

    Field all_fields[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields[i] = (Field)i;
    }
    AcTaskDefinition ops[] = {
	    acCompute(update,all_fields)
    };
    AcTaskGraph* graph = acGridBuildTaskGraph(ops);

    acGridExecuteTaskGraph(graph,3);

    // arr test
    if (pid == 0)
        acHostMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_DEFAULT);
    acDeviceLoad(acGridGetDevice(), STREAM_DEFAULT,info,AC_4d_float_arr_out);

    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    	acGridExecuteTaskGraph(graph,1);

    //acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acGridSynchronizeStream(STREAM_DEFAULT);

    const int nx_min = model.info[AC_nmin].x;
    const int nx_max = model.info[AC_nlocal_max].x;

    const int ny_min = model.info[AC_nmin].y;
    const int ny_max = model.info[AC_nlocal_max].y;

    const int nz_min = model.info[AC_nmin].z;
    const int nz_max = model.info[AC_nlocal_max].z;

    auto IDX = [&](const int i, const int j, const int k)
    {
	    return acVertexBufferIdx(i,j,k,model.info);
    };

    const auto ghosts = acDeviceGetLocalConfig(acGridGetDevice())[AC_nmin];
    for (int step_number = 0; step_number < NUM_INTEGRATION_STEPS; ++step_number) {

	//test arr with random compute
        for (int k = nz_min; k < nz_max; ++k) {
            for (int j = ny_min; j < ny_max; ++j) {
                for (int i = nx_min; i < nx_max; ++i) {
			[[maybe_unused]] int comp_x = i - ghosts.x;
			int comp_y = j - ghosts.y;
			[[maybe_unused]] int comp_z = j - ghosts.z;
			model.vertex_buffer[FIELD_X][IDX(i,j,k)] = test_int_arr[0]*(test_arr[0] + test_arr[3] + test_arr_2[0])*global_arr[i-info[AC_nmin].x];
			model.vertex_buffer[FIELD_Y][IDX(i,j,k)] = test_int_arr[1]*(test_arr[1] + test_arr[4] + test_arr_2[1] + twoD_real_arr TWO_D_ARR(info[AC_dconst_int],comp_y) + threeD_real_arr THREE_D_ARR(i,j,k));
			model.vertex_buffer[FIELD_Z][IDX(i,j,k)] = test_int_arr[2]*(test_arr[2] + test_arr[5] + test_arr_2[2])*((AcReal)fourD_float_arr FOUR_D_ARR(i,j,k,0) + (AcReal)fourD_float_arr FOUR_D_ARR(i,j,k,1)  + (AcReal)fourD_float_arr FOUR_D_ARR(i,j,k,2));

			fourD_float_arr FOUR_D_ARR(i,j,k,0) = (float) fourD_float_arr FOUR_D_ARR(i,j,k,0) * test_arr[1];
                }
            }
        }
    }

    const AcResult res = acVerifyMesh("arrays", model, candidate);
    if (res != AC_SUCCESS) {
        retval = res;
        WARNCHK_ALWAYS(retval);
    }

    fflush(stdout);
    int read_global_arr[nx];
    AcReal read_2d TWO_D_ARR(nx,ny);
    float read_fourD_float_arr FOUR_D_ARR(mx,my,mz,3);
    acDeviceStore(acGridGetDevice(), STREAM_DEFAULT, AC_global_arr, read_global_arr);
    acDeviceStore(acGridGetDevice(), STREAM_DEFAULT, AC_2d_reals, &read_2d[0][0]);
    acDeviceStore(acGridGetDevice(), STREAM_DEFAULT, AC_4d_float_arr_out, &read_fourD_float_arr[0][0][0][0]);
    bool arrays_are_the_same = true;
    for(int i = 0; i < info[AC_nlocal].x; ++i)
	    arrays_are_the_same &= (read_global_arr[i] == global_arr[i]);
    for(int i = 0; i < info[AC_nlocal].x; ++i)
    	for(int j = 0; j < info[AC_nlocal].y; ++j)
	    arrays_are_the_same &= (read_2d TWO_D_ARR(i,j) == twoD_real_arr TWO_D_ARR(i,j));
    bool updated_arrays_are_the_same = true;
    for(int k = 0; k < mz; ++k)
    	for(int j = 0; j < my; ++j)
    		for(int i = 0; i < mx; ++i)
			for(int l = 0; l < 3; ++l)
			{
				const bool eq = (read_fourD_float_arr FOUR_D_ARR(i,j,k,l) == fourD_float_arr FOUR_D_ARR(i,j,k,l));
				updated_arrays_are_the_same &= eq;
				if(!eq) fprintf(stderr,"different at %d,%d,%d,%d\n",i,j,k,l);
			}
    printf("LOAD STORE GMEM ARRAY... %s \n", arrays_are_the_same ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    printf("GMEM ARRAY UPDATE ... %s \n", updated_arrays_are_the_same ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "ARR_TEST complete: %s\n",
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
