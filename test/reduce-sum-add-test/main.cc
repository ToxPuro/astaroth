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
#define FIVE_D_ARR(i,j,k,l,n) [n][l][k][j][i]
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

    acGridInit(info);
    acHostUpdateParams(&info);

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(add_step),1);

    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(init),1);

    bool correct = true;
    for(int i = 1; i <= 10; ++i)
    {
      acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(add_step),1);
      const AcReal f_sum = acDeviceGetOutput(acGridGetDevice(),AC_F_sum);
      const AcReal f_global_sum = acDeviceGetOutput(acGridGetDevice(),AC_F_global_sum);
      const AcReal true_val = info[AC_nlocal].x*info[AC_nlocal].y*info[AC_nlocal].z*i*1.0;
      const AcReal true_global_val = true_val*nprocs;
      fprintf(stderr,"%.14e vs. %.14e\n",f_sum,true_val);
      fprintf(stderr,"%.14e vs. %.14e\n",f_global_sum,true_global_val);
      fprintf(stderr,"\n");
      correct &= (f_sum == true_val);
      correct &= (f_global_sum == true_global_val);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if(!correct) retval = AC_FAILURE;
    if (pid == 0)
        fprintf(stderr, "REDUCE_SUM_ADD_TEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

    return retval == AC_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
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
