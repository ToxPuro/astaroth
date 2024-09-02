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

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))
#define NUM_INTEGRATION_STEPS (2)

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

    const int max_devices = 1;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetMeshDims(2 * 9, 2 * 11, 4 * 7, &info);
    constexpr int nx = 2*9;
    constexpr int ny = 2*11;
    constexpr int nz = 4*7;

    constexpr int mx = 2*9  + 2*NGHOST;
    constexpr int my = 2*11 + 2*NGHOST;
    constexpr int mz = 4*7  + 2*NGHOST;
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
    info.real_arrays[AC_test_arr] = (AcReal*)test_arr;
    //these are read from config
    //info.int_arrays[AC_test_int_arr] = (int*)test_int_arr;
    //info.real_arrays[AC_test_arr_2] = (AcReal*)test_arr_2;
    int global_arr[nx];
#if AC_ROW_MAJOR_ORDER
    AcReal twoD_real_arr[nx][ny];
    AcReal threeD_real_arr[mx][my][mz];
#else
    AcReal twoD_real_arr[ny][nx];
    AcReal threeD_real_arr[mz][my][mx];
#endif

    for(int i = 0; i < nx; ++i)
		    global_arr[i] = 1;
    for(int j = 0; j < ny; ++j)
    	for(int i = 0; i < nx; ++i)
#if AC_ROW_MAJOR_ORDER
		twoD_real_arr[i][j] = (1.0*rand())/RAND_MAX;
#else
		twoD_real_arr[j][i] = (1.0*rand())/RAND_MAX;
#endif

    for(int k = 0; k < mz; ++k)
    	for(int j = 0; j < my; ++j)
    		for(int i = 0; i < mx; ++i)
#if AC_ROW_MAJOR_ORDER
			threeD_real_arr[i][j][k]  = (1.0*rand())/RAND_MAX;
#else
			threeD_real_arr[k][j][i]  = (1.0*rand())/RAND_MAX;
#endif
    info.int_arrays[AC_global_arr] = global_arr;
    info.real_arrays[AC_2d_reals]  = &twoD_real_arr[0][0];
    info.real_arrays[AC_3d_reals]  = &threeD_real_arr[0][0][0];
    info.int_params[AC_dconst_int] = nx-NGHOST_X;
    acGridInit(info);

    Field all_fields[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields[i] = (Field)i;
    }
    auto null_loader = [&](ParamLoadingInfo l){(void)l;};
    AcTaskDefinition ops[] = {
	    acComputeWithParams(KERNEL_test_arr,all_fields,null_loader)
    };
    AcTaskGraph* graph = acGridBuildTaskGraph(ops);

    acGridExecuteTaskGraph(graph,3);

    // arr test
    if (pid == 0)
        acHostMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);

    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    	acGridExecuteTaskGraph(graph,1);

    //acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);


    const int nx_min = model.info.int_params[AC_nx_min];
    const int nx_max = model.info.int_params[AC_nx_max];

    const int ny_min = model.info.int_params[AC_ny_min];
    const int ny_max = model.info.int_params[AC_ny_max];

    const int nz_min = model.info.int_params[AC_nz_min];
    const int nz_max = model.info.int_params[AC_nz_max];
    auto IDX = [&](const int i, const int j, const int k)
    {
	    return acVertexBufferIdx(i,j,k,model.info);
    };

    for (int step_number = 0; step_number < 1; ++step_number) {

	//test arr with random compute
        for (int k = nz_min; k < nz_max; ++k) {
            for (int j = ny_min; j < ny_max; ++j) {
                for (int i = nx_min; i < nx_max; ++i) {
			int comp_x = i - NGHOST_X;
			int comp_y = j - NGHOST_Y;
			int comp_z = j - NGHOST_Z;
			model.vertex_buffer[FIELD_X][IDX(i,j,k)] = test_int_arr[0]*(test_arr[0] + test_arr[3] + test_arr_2[0])*global_arr[i-NGHOST_X];
#if AC_ROW_MAJOR_ORDER
			model.vertex_buffer[FIELD_Y][IDX(i,j,k)] = test_int_arr[1]*(test_arr[1] + test_arr[4] + test_arr_2[1] + twoD_real_arr[info.int_params[AC_dconst_int]][comp_y] + threeD_real_arr[i][j][k]);
#else
			model.vertex_buffer[FIELD_Y][IDX(i,j,k)] = test_int_arr[1]*(test_arr[1] + test_arr[4] + test_arr_2[1] + twoD_real_arr[comp_y][info.int_params[AC_dconst_int]] + threeD_real_arr[k][j][i]);
#endif
			model.vertex_buffer[FIELD_Z][IDX(i,j,k)] = test_int_arr[2]*(test_arr[2] + test_arr[5] + test_arr_2[2]);
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
    acStoreUniform(AC_global_arr, read_global_arr, get_array_length(AC_global_arr,model.info));
    bool arrays_are_the_same = true;
    for(int i = 0; i < info.int_params[AC_nx]; ++i)
	    arrays_are_the_same &= (read_global_arr[i] == global_arr[i]);
    printf("LOAD STORE GMEM ARRAY... %s \n", arrays_are_the_same ? AC_GRN "OK! " AC_COL_RESET : AC_RED "FAIL! " AC_COL_RESET);
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
