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
#include "../../stdlib/reduction.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED
#include "grid_detail.h"

#include <mpi.h>
#include <vector>

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
}

int
main(int argc, char* argv[])
{


    const size_t nx = argc >  1 ? (size_t)atoi(argv[1]) : 2*9;
    const size_t ny = argc >  2 ? (size_t)atoi(argv[2]) : 2*11;
    const size_t nz = argc >  3 ? (size_t)atoi(argv[3]) : 4*7;
    const size_t NUM_INTEGRATION_STEPS = argc >  4 ? (size_t)atoi(argv[4]) : 100;
    MPI_Init(NULL,NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    AcMeshInfo info = acInitInfo();
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info.comm = MPI_COMM_WORLD;

    acSetGridMeshDims(nx, ny, nz, &info);
    //TP: this is because of backwards compatibility
    acSetLocalMeshDims(nx, ny, nz, &info);

    acPushToConfig(info,AC_proc_mapping_strategy, (int)AcProcMappingStrategy::Linear);
    acPushToConfig(info,AC_decompose_strategy,    (int)AcDecomposeStrategy::Morton);
    acPushToConfig(info,AC_MPI_comm_strategy,    (int)AcMPICommStrategy::DuplicateMPICommWorld);

#if AC_RUNTIME_COMPILATION
    AcReal real_arr[4];
    int int_arr[2];
    bool bool_arr[2] = {false,true};
    for(int i = 0; i < 4; ++i)
    	real_arr[i] = -i;
    for(int i = 0; i < 2; ++i)
    	int_arr[i] = i;
    acLoadCompInfo(AC_lspherical_coords,true,&info.run_consts);
    acLoadCompInfo(AC_runtime_int,0,&info.run_consts);
    acLoadCompInfo(AC_runtime_real,0.12345,&info.run_consts);
    acLoadCompInfo(AC_runtime_real3,{0.12345,0.12345,0.12345},&info.run_consts);
    acLoadCompInfo(AC_runtime_int3,{0,1,2},&info.run_consts);
    acLoadCompInfo(AC_runtime_real_arr,real_arr,&info.run_consts);
    acLoadCompInfo(AC_runtime_int_arr,int_arr,&info.run_consts);
    acLoadCompInfo(AC_runtime_bool_arr,bool_arr,&info.run_consts);
    const char* build_str = "-DOPTIMIZE_FIELDS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_MODEL=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_ACM=OFF";
    info.runtime_compilation_log_dst = "ac_compilation_log";
    acCompile(build_str,info);
    acLoadLibrary(stdout);
    acLoadUtils(stdout);
#endif
    atexit(acAbort);
    int retval = 0;




    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc

    const int max_devices = 2 * 2 * 4;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d). Please modify "
                "mpitest/main.cc to use a larger mesh.\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    AcMesh model, candidate;
    if (pid == 0) {
        acHostGridMeshCreate(info, &model);
        acHostGridMeshCreate(info, &candidate);
        acHostGridMeshRandomize(&model);
        acHostGridMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    AcReal* gmem_arr = (AcReal*)malloc(sizeof(AcReal)*100);
    memset(gmem_arr,0,sizeof(AcReal)*100);
    info[AC_real_gmem_arr] = gmem_arr;
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

    // Boundconds
    if (pid == 0)
        acHostGridMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
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

    // DSL Boundconds
    // TP: Works only for a single proc but that is sufficient for now
    if(nprocs == 1)
    {
    	if (pid == 0)
    	    acHostGridMeshRandomize(&model);

    	const auto periodic = acGetDSLTaskGraph(boundconds);
    	acGridLoadMesh(STREAM_DEFAULT, model);
    	acGridSynchronizeStream(STREAM_DEFAULT);
    	acGridExecuteTaskGraphBase(periodic,1,true);
    	acGridSynchronizeStream(STREAM_DEFAULT);
    	acGridStoreMesh(STREAM_DEFAULT, &candidate);
    	acGridSynchronizeStream(STREAM_DEFAULT);
    	if (pid == 0) {
    	    acHostMeshApplyPeriodicBounds(&model);
    	    const AcResult res = acVerifyMesh("DSL Periodic boundconds", model, candidate);
    	    if (res != AC_SUCCESS) {
    	        retval = res;
    	        WARNCHK_ALWAYS(retval);
    	    }
    	}
    	fflush(stdout);
    }
    //// Dryrun
    const AcReal dt = (AcReal)FLT_EPSILON;

    acGridIntegrate(STREAM_DEFAULT, dt);
    acGridSynchronizeStream(STREAM_DEFAULT);

    AcTaskGraph* dsl_graph = acGetDSLTaskGraph(AC_test_rhs);
    acGridExecuteTaskGraph(dsl_graph,1);
    acGridSynchronizeStream(STREAM_DEFAULT);


    // Integration
    if (pid == 0)
        acHostGridMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_DEFAULT);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_DEFAULT);

    // Device integrate
    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        acGridIntegrate(STREAM_DEFAULT, dt);

    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);

        // Host integrate
        for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
            acHostIntegrateStep(model, dt);

        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("Integration", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);

    acGridSynchronizeStream(STREAM_DEFAULT);

    // Integration
    if (pid == 0)
        acHostGridMeshRandomize(&model);

    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_DEFAULT);
    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_DEFAULT);

    // Device integrate
    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    	acGridExecuteTaskGraph(dsl_graph,3);

    acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        acHostMeshApplyPeriodicBounds(&model);

        // Host integrate
        for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
            acHostIntegrateStep(model, dt);

        acHostMeshApplyPeriodicBounds(&model);
        const AcResult res = acVerifyMesh("DSL ComputeSteps", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }
    fflush(stdout);

    // Scalar reductions
    if (pid == 0) {
        printf("---Test: Scalar reductions---\n");
        acHostGridMeshRandomize(&model);
        acHostMeshApplyPeriodicBounds(&model);
    }
    fflush(stdout);
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridPeriodicBoundconds(STREAM_DEFAULT);

    const AcReduction scal_reductions[] = {RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS,
                                             RTYPE_RMS_EXP};
    for (size_t i = 0; i < ARRAY_SIZE(scal_reductions); ++i) { // NOTE: not using NUM_RTYPES here
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        const auto reduction = scal_reductions[i];

        AcReal candval;
        acGridReduceScal(STREAM_DEFAULT, reduction, v0, &candval);

        if (pid == 0) {
            const AcReal modelval = acHostReduceScal(model, reduction, v0);

            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceScal(model, RTYPE_MAX, v0);
            error.minimum_magnitude = acHostReduceScal(model, RTYPE_MIN, v0);

            if (!acEvalError(reduction.name, error)) {
                fprintf(stderr, "Scalar %s: cand %g model %g\n", reduction.name, (double)candval, (double)modelval);
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }
    fflush(stdout);

    // Vector reductions
    if (pid == 0) {
        printf("---Test: Vector reductions---\n");
    }
    fflush(stdout);

    const AcReduction vec_reductions[] = {RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS,
                                            RTYPE_RMS_EXP};
    for (size_t i = 0; i < ARRAY_SIZE(vec_reductions); ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        const VertexBufferHandle v1 = (VertexBufferHandle)1;
        const VertexBufferHandle v2 = (VertexBufferHandle)2;
        AcReal candval;

        const auto reduction = vec_reductions[i];
        acGridReduceVec(STREAM_DEFAULT, reduction, v0, v1, v2, &candval);
        if (pid == 0) {
            const AcReal modelval = acHostReduceVec(model, reduction, v0, v1, v2);

            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVec(model, RTYPE_MAX, v0, v1, v2);
            error.minimum_magnitude = acHostReduceVec(model, RTYPE_MIN, v0, v1, v1);

            if (!acEvalError(reduction.name, error)) {
                fprintf(stderr, "Vector %s: cand %g model %g\n", reduction.name, (double)candval, (double)modelval);
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }
    fflush(stdout);

    if (pid == 0) {
        printf("---Test: Alfven reductions---\n");
    }
    fflush(stdout);

    const AcReduction alf_reductions[] = {RTYPE_ALFVEN_MAX, RTYPE_ALFVEN_MIN, RTYPE_ALFVEN_RMS};
    for (size_t i = 0; i < ARRAY_SIZE(alf_reductions); ++i) { // NOTE: 2 instead of NUM_RTYPES
        const VertexBufferHandle v0 = (VertexBufferHandle)0;
        const VertexBufferHandle v1 = (VertexBufferHandle)1;
        const VertexBufferHandle v2 = (VertexBufferHandle)2;
        const VertexBufferHandle v3 = (VertexBufferHandle)3;
        AcReal candval;

        const auto reduction = alf_reductions[i];
        acGridReduceVecScal(STREAM_DEFAULT, reduction, v0, v1, v2, v3, &candval);
        if (pid == 0) {
            const AcReal modelval = acHostReduceVecScal(model, reduction, v0, v1, v2, v3);

            Error error             = acGetError(modelval, candval);
            error.maximum_magnitude = acHostReduceVecScal(model, RTYPE_ALFVEN_MAX, v0, v1, v2, v3);
            error.minimum_magnitude = acHostReduceVecScal(model, RTYPE_ALFVEN_MIN, v0, v1, v1, v3);

            if (!acEvalError(reduction.name, error)) {
                fprintf(stderr, "Alfven %s: cand %g model %g\n", reduction.name, (double)candval, (double)modelval);
                retval = AC_FAILURE;
                WARNCHK_ALWAYS(retval);
            }
        }
    }
    fflush(stdout);

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
