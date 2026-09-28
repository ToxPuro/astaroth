/*
    Copyright (C) 2026, Touko Puro

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
    Tests that carrying the halo sync state between compute steps (AC_carry_halo_sync_between_compute_steps)
    gives the same results as exchanging the halos in every compute step, and that it actually skips exchanges.

    Running: mpirun -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED

#include <mpi.h>
#include <fstream>
#include <string>

static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

static void
execute(const AcDSLTaskGraph steps, const size_t n_iterations = 1)
{
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(steps), n_iterations);
    acGridSynchronizeStream(STREAM_ALL);
}

static void
run_sequence(AcMeshInfo info, const bool carry_halo_sync, const AcMesh model, AcMesh* after_rounds, AcMesh* candidate)
{
    acPushToConfig(info, AC_carry_halo_sync_between_compute_steps, carry_halo_sync);
    acGridInit(info);
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_ALL);
    for (int round = 0; round < 3; ++round) {
        execute(step_a);
        execute(step_b); // F exchanged by step_a
        execute(step_a); // F exchanged by step_b
        execute(step_w, 2); // Assumes F is in sync but writes F: must not skip the exchange in the second iteration
        execute(step_b);
    }
    acGridStoreMesh(STREAM_DEFAULT, after_rounds);
    acGridSynchronizeStream(STREAM_ALL);
    // Writing outside of task graphs invalidates the halos (model has random halos)
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_ALL);
    execute(step_b);
    acGridStoreMesh(STREAM_DEFAULT, candidate);
    acGridSynchronizeStream(STREAM_ALL);
    acGridQuit();
}

// Returns whether some task graph built for step_b did not exchange any halos
static bool
step_b_skipped_halo_exchange(const char* log_path)
{
    std::ifstream log(log_path);
    std::string line;
    bool in_step_b = false, has_halo = false, skipped = false;
    while (std::getline(log, line)) {
        if (line.find(" Ops:") != std::string::npos) {
            if (in_step_b && !has_halo) skipped = true;
            in_step_b = line.rfind("step_b Ops:", 0) == 0;
            has_halo  = false;
        }
        if (line.rfind("Halo(", 0) == 0) has_halo = true;
    }
    if (in_step_b && !has_halo) skipped = true;
    return skipped;
}

int
main(void)
{
    atexit(acAbort);
    int retval = 0;

    int nprocs, pid;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    srand(321654987);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acPushToConfig(info, AC_MPI_comm_strategy, AC_MPI_COMM_STRATEGY_DUP_WORLD);
    info.comm->handle = MPI_COMM_WORLD;
    acSetGridMeshDims(16, 16, 16, &info);
    const int3 decomp = acDecompose(nprocs, info);
    acSetLocalMeshDims(16 / decomp.x, 16 / decomp.y, 16 / decomp.z, &info);

    // Global meshes: acGridLoadMesh scatters from and acGridStoreMesh gathers to the root
    AcMesh model, with_carry, without_carry, with_carry_rounds, without_carry_rounds;
    acHostGridMeshCreate(info, &model);
    acHostGridMeshCreate(info, &with_carry);
    acHostGridMeshCreate(info, &without_carry);
    acHostGridMeshCreate(info, &with_carry_rounds);
    acHostGridMeshCreate(info, &without_carry_rounds);
    acHostMeshRandomize(&model);

    const char* log_path = "taskgraph_log.txt";
    if (pid == 0) remove(log_path);
    MPI_Barrier(MPI_COMM_WORLD);

    run_sequence(info, false, model, &without_carry_rounds, &without_carry);
    if (pid == 0) remove(log_path);
    MPI_Barrier(MPI_COMM_WORLD);
    run_sequence(info, true, model, &with_carry_rounds, &with_carry);

    if (pid == 0) {
        AcResult res = acVerifyMesh("Carrying halo sync between compute steps", without_carry_rounds, with_carry_rounds);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
        res = acVerifyMesh("Writes outside of task graphs invalidate the halos", without_carry, with_carry);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
        if (!step_b_skipped_halo_exchange(log_path)) {
            fprintf(stderr, "No task graph of step_b skipped the halo exchange of F\n");
            retval = AC_FAILURE;
        }
    }
    MPI_Bcast(&retval, 1, MPI_INT, 0, MPI_COMM_WORLD);

    acHostMeshDestroy(&model);
    acHostMeshDestroy(&with_carry);
    acHostMeshDestroy(&without_carry);
    acHostMeshDestroy(&with_carry_rounds);
    acHostMeshDestroy(&without_carry_rounds);

    MPI_Finalize();
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "HALO SYNC CARRY TEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");

    return retval == AC_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

#else
int
main(void)
{
    printf("The library was built without MPI support, cannot run the test. Rebuild Astaroth with "
           "cmake -DMPI_ENABLED=ON .. to enable.\n");
    return EXIT_FAILURE;
}
#endif // AC_MPI_ENABLED
