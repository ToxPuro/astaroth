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
#include "../../stdlib/reduction.h"

#if AC_MPI_ENABLED

#include <mpi.h>
#include <vector>

#define NUM_INTEGRATION_STEPS (1)
#include "user_non_scalar_constants.h"


#define DER2_3 (1. / 90.)
#define DER2_2 (-3. / 20.)
#define DER2_1 (3. / 2.)
#define DER2_0 (-49. / 18.)

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
main(int argc, char* argv[])
{
    atexit(acAbort);
    int retval = 0;

    MPI_Init(NULL,NULL);

    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("PC-AC.conf", &info);
    printf("MAX DIM: %d\n",info[AC_nlocal_max_dim]);
    printf("SPHERICAL: %d\n",info[AC_lspherical_coords__mod__cdata]);
    //printf("SHOCK ASKED FOR: %d\n",info.run_consts.config.bool_arrays[AC_lpencil__mod__cdata][i_shock__mod__cparam]);
    //printf("CHIT  IS NULL: %d\n",info[AC_chit_prof_stored__mod__energy] == NULL);
    //printf("GRAVX IS NULL: %d\n",info[AC_gravx_xpencil__mod__gravity] == NULL);
    //exit(EXIT_SUCCESS);

    /**
    int n_active = 0;
    const bool* lpencil = info.run_consts.config[AC_lpencil__mod__cdata];
    printf("LPENCIL IS NULL: %d\n",lpencil == NULL);
    printf("DEL2U asked: %d\n",lpencil[i_del2u__mod__cparam-1]);
    printf("EQUIDIST LOADED: %d\n",info.run_consts.is_loaded.bool3_params[AC_lequidist__mod__cdata]);
    for(int i = 0; i < 270; ++i) n_active += lpencil[i];
    printf("%d pencils active\n",n_active);
    **/

    finalized = true;
    info.comm = MPI_COMM_WORLD;

#if AC_RUNTIME_COMPILATION
#if AC_USE_HIP
    const char* build_str = "-DUSE_HIP=ON  -DOPTIMIZE_FIELDS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DMAX_THREADS_PER_BLOCK=512 -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../../test/pc-transpiled/DSL";
#else
    const char* build_str = "-DUSE_HIP=OFF  -DOPTIMIZE_FIELDS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DMAX_THREADS_PER_BLOCK=512 -DBUILD_TESTS=ON -DDSL_MODULE_DIR=../../test/pc-transpiled/DSL";
#endif
    acCompile(build_str,"pc-transpiled",info);
    acLoadLibrary();
    acLoadUtils();
#endif

    const int max_devices = 8;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d) this test works only with a single device\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    acGridInit(info);

    acGridAccessMeshOnDiskSynchronousDistributed(F_UU.x, "PC-AC-data","initial-condition", ACCESS_READ);
    acGridAccessMeshOnDiskSynchronousDistributed(F_UU.y, "PC-AC-data","initial-condition", ACCESS_READ);
    acGridAccessMeshOnDiskSynchronousDistributed(F_UU.z, "PC-AC-data","initial-condition", ACCESS_READ);

    acGridAccessMeshOnDiskSynchronousDistributed(F_AVEC.x, "PC-AC-data","initial-condition", ACCESS_READ);
    acGridAccessMeshOnDiskSynchronousDistributed(F_AVEC.y, "PC-AC-data","initial-condition", ACCESS_READ);
    acGridAccessMeshOnDiskSynchronousDistributed(F_AVEC.z, "PC-AC-data","initial-condition", ACCESS_READ);

    acGridAccessMeshOnDiskSynchronousDistributed(RHO, "PC-AC-data","initial-condition", ACCESS_READ);

    acGridAccessMeshOnDiskSynchronousDistributed(SS, "PC-AC-data","initial-condition", ACCESS_READ);

    acGridSynchronizeStream(STREAM_ALL);

    {
    	AcReal uu_rms;
    	AcReal3 uu_sum;
    	AcReal  ss_sum;
    	// Calculate rms, min and max from the velocity vector field
    	acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, F_UU.x, F_UU.y, F_UU.z, &uu_rms);
    	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, F_UU.x,&uu_sum.x);
    	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, F_UU.y,&uu_sum.y);
    	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, F_UU.z,&uu_sum.z);
    	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, SS,&ss_sum);
    	acGridSynchronizeStream(STREAM_ALL);
    	fprintf(stderr,"UU_RMS: %14e\n",uu_rms);
    	fprintf(stderr,"UUX_SUM: %14e\n",uu_sum.x);
    	fprintf(stderr,"UUY_SUM: %14e\n",uu_sum.y);
    	fprintf(stderr,"UUZ_SUM: %14e\n",uu_sum.z);
    	fprintf(stderr,"SS_SUM: %14e\n",ss_sum);
    	fflush(stderr);
    }

    std::vector<Field> all_fields;
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        all_fields.push_back((Field)i);
    }
    AcTaskGraph* graph = acGetDSLTaskGraph(rhs);

    for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
    {

        acGridExecuteTaskGraph(graph,1);
        acGridSynchronizeStream(STREAM_ALL);
    }
    acGridSynchronizeStream(STREAM_ALL);
    acGridFinalizeReduceLocal(graph);
    acGridSynchronizeStream(STREAM_ALL);
    printf("DF RHO SUM: %14e\n",acDeviceGetOutput(acGridGetDevice(),AC_df_rho_sum));

    acGridSynchronizeStream(STREAM_ALL);
    //{
    //	AcReal rho_sum;
    //	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, TEST_1,&rho_sum);
    //	acGridSynchronizeStream(STREAM_ALL);
    //	printf("TEST_1 SUM: %14e\n",rho_sum);

    //	AcReal aax_sum;
    //	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, TEST_2,&aax_sum);
    //	acGridSynchronizeStream(STREAM_ALL);
    //	printf("TEST_2 SUM: %14e\n",aax_sum);

    //	AcReal aay_sum;
    //	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, TEST_3,&aay_sum);
    //	acGridSynchronizeStream(STREAM_ALL);
    //	printf("TEST_3 SUM: %14e\n",aay_sum);

    //	AcReal aaz_sum;
    //	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, TEST_4 ,&aaz_sum);
    //	acGridSynchronizeStream(STREAM_ALL);
    //	printf("TEST_4 SUM: %14e\n",aaz_sum);
    //}

    {
    	AcReal uu_rms;
    	AcReal3 uu_sum;
    	// Calculate rms, min and max from the velocity vector field
    	acGridReduceVec(STREAM_DEFAULT, RTYPE_RMS, F_UU.x, F_UU.y, F_UU.z, &uu_rms);
    	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, F_UU.x,&uu_sum.x);
    	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, F_UU.y,&uu_sum.y);
    	acGridReduceScal(STREAM_DEFAULT, RTYPE_SUM, F_UU.z,&uu_sum.z);
    	acGridSynchronizeStream(STREAM_ALL);
    	fprintf(stderr,"UU_RMS: %14e\n",uu_rms);
    	fprintf(stderr,"UU_SUM: %14e\n",uu_sum.x + uu_sum.y + uu_sum.z);
    	fflush(stderr);
    }

    //acGridPeriodicBoundconds(STREAM_DEFAULT);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    AcReal test_sum_1 = 0.0;
    AcReal test_sum_2 = 0.0;
    AcReal test_sum_3 = 0.0;
    AcReal test_sum_4 = 0.0;
    AcReal test_sum_5 = 0.0;
    for(int y = NGHOST; y < 32+NGHOST; ++y)
    {
    	for(int z = NGHOST; z < 32+NGHOST; ++z)
	{
    		for(int x = NGHOST; x < 32+NGHOST; ++x)
    		{
			test_sum_1 = std::max(test_sum_1,fabs(candidate.vertex_buffer[TEST_1][acVertexBufferIdx(x,y,z,model.info)]));
			test_sum_2 = std::max(test_sum_2,fabs(candidate.vertex_buffer[TEST_2][acVertexBufferIdx(x,y,z,model.info)]));
			test_sum_3 = std::max(test_sum_3,fabs(candidate.vertex_buffer[TEST_3][acVertexBufferIdx(x,y,z,model.info)]));
			test_sum_4 = std::max(test_sum_4,fabs(candidate.vertex_buffer[TEST_4][acVertexBufferIdx(x,y,z,model.info)]));
			test_sum_5 = std::max(test_sum_5,fabs(candidate.vertex_buffer[TEST_5][acVertexBufferIdx(x,y,z,model.info)]));
		}
	}
    }
    fprintf(stderr,"TEST_SUM_1: %14e\n",test_sum_1);
    fprintf(stderr,"TEST_SUM_2: %14e\n",test_sum_2);
    fprintf(stderr,"TEST_SUM_3: %14e\n",test_sum_3);
    fprintf(stderr,"TEST_SUM_4: %14e\n",test_sum_4);
    fprintf(stderr,"TEST_SUM_5: %14e\n",test_sum_5);
    fflush(stderr);
    //acHostMeshApplyPeriodicBounds(&candidate);
    //acDeviceSwapBuffer(acGridGetDevice(), UU);
    //acDeviceStoreMesh(acGridGetDevice(),STREAM_DEFAULT,&candidate);



    if (pid == 0) {
        acHostMeshDestroy(&model);
        acHostMeshDestroy(&candidate);
    }

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "PC-TRANSPILED complete: %s\n",
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
