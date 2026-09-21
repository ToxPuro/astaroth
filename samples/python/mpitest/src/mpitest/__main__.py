#!/bin/env python3

# Copyright (C) 2014-2026, Johannes Pekkila, Miikka Vaisala, Ondřej Míchal.
#
# This file is part of Astaroth.
#
# Astaroth is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Astaroth is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Astaroth.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import pathlib
import sys
import unittest
from typing import Final, Optional

import mpitest.astaroth as ac
from mpi4py import MPI

BUILD_STR: Final[str] = (
    "-DOPTIMIZE_FIELDS=ON -DOPTIMIZE_INPUT_PARAMS=ON -DELIMINATE_CONDITIONALS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_ACM=OFF"
)
MAX_DEVICES: Final[int] = 2 * 2 * 4

args: Optional[argparse.Namespace] = None
model: Optional[ac.Mesh] = None
candidate: Optional[ac.Mesh] = None

def print0(*args, **kwargs):
    if MPI.COMM_WORLD.rank != 0:
        return
    print(*args, **kwargs)


def verify_mesh(msg: str, reference: ac.Mesh, candidate: ac.Mesh, rank: int) -> bool:
    success: bool = False
    if rank == 0:
        success = ac.verify_mesh(msg, reference, candidate) == ac.Result.ac_success
    return MPI.COMM_WORLD.bcast(success, root=0)


def setUpModule():
    info: ac.AcMeshInfo = ac.init_info()
    ac.load_config(str(args.config), info)

    # FIXME: Make the acPushToConfig work similarly to acDeviceSetInput. Needs
    # overloads.
    # acPushToConfig(info,AC_proc_mapping_strategy, AC_PROC_MAPPING_STRATEGY_LINEAR);
    # acPushToConfig(info,AC_decompose_strategy,    AC_DECOMPOSE_STRATEGY_MORTON);
    # acPushToConfig(info,AC_MPI_comm_strategy,     AC_MPI_COMM_STRATEGY_DUP_WORLD);
    ac.communicator_set_communicator(info.comm, MPI.COMM_WORLD)

    nprocs: int = MPI.COMM_WORLD.size
    rank: int = MPI.COMM_WORLD.rank

    assert nprocs <= MAX_DEVICES, (
        "The number of devices (resp. processes) is higher than the maximum permitted number."
    )

    decomp: ac.int3 = ac.decompose(nprocs, info)

    ac.set_grid_mesh_dims(*args.dims, info)
    ac.set_local_mesh_dims(
        ac.int3(
            args.dims[0] / decomp.x, args.dims[1] / decomp.y, args.dims[2] / decomp.z
        ),
        info,
    )

    # AcReal real_arr[4];
    # bool bool_arr[2] = {false,true};
    # for(int i = 0; i < 4; ++i)
    # 	real_arr[i] = -i;
    #
    # int int_arr[2];
    # for(int i = 0; i < 2; ++i)
    # 	int_arr[i] = i;
    # acLoadCompInfo(AC_lspherical_coords,true,&info.run_consts);
    # acLoadCompInfo(AC_runtime_int,0,&info.run_consts);
    # acLoadCompInfo(AC_runtime_real,0.12345,&info.run_consts);
    # acLoadCompInfo(AC_runtime_real3,{0.12345,0.12345,0.12345},&info.run_consts);
    # acLoadCompInfo(AC_runtime_int3,{0,1,2},&info.run_consts);
    # acLoadCompInfo(AC_runtime_real_arr,real_arr,&info.run_consts);
    # acLoadCompInfo(AC_runtime_int_arr,int_arr,&info.run_consts);
    # acLoadCompInfo(AC_runtime_bool_arr,bool_arr,&info.run_consts);
    ac.compile(BUILD_STR, info)
    ac.load_library(info)
    ac.load_utils(info)

    ac.srand(321654987)

    # CPU alloc
    model: ac.Mesh = ac.Mesh()
    candidate: ac.Mesh = ac.Mesh()
    if rank == 0:
        ac.host_grid_mesh_create(info, model)
        ac.host_grid_mesh_create(info, candidate)

    # // GPU alloc & compute
    # AcReal* gmem_arr = (AcReal*)malloc(sizeof(AcReal)*100);
    # memset(gmem_arr,0,sizeof(AcReal)*100);
    # info[AC_real_gmem_arr] = gmem_arr;
    ac.grid_init(info)


def tearDownModule():
    ac.grid_quit()


class MPITestInit(unittest.TestCase):
    def setUp(self):
        self.nprocs: int = MPI.COMM_WORLD.size
        self.rank: int = MPI.COMM_WORLD.rank

        global model, candidate
        self.model = model
        self.candidate = candidate
        ac.host_grid_mesh_randomize(self.candidate)

        self.grid_device = ac.grid_get_device()

    def tearDown(self):
        pass

    def test_load_store(self):
        if self.rank == 0:
            ac.store_config(
                ac.device_get_local_config(self.grid_device), "mpitest.conf"
            )

        ac.grid_load_mesh(ac.Stream.default, self.model)
        ac.grid_store_mesh(ac.Stream.default, self.candidate)

        self.assertTrue(
            verify_mesh("Load/Store", self.model, self.candidate, self.rank)
        )

    def test_write_read(self):
        ac.grid_write_mesh_to_disk_launch("snapshot", "0")
        ac.grid_disk_access_sync()

        # std::vector<Field> io_fields{};
        # for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        #     io_fields.push_back(Field(i));
        # }
        # const size_t num_io_fields = io_fields.size();
        # for (size_t i = 0; i < num_io_fields; ++i)
        #     acGridAccessMeshOnDiskSynchronous(io_fields[i], "snapshot", "0", ACCESS_READ);

        ac.grid_periodic_boundconds(ac.Stream.default)
        ac.grid_store_mesh(ac.Stream.default, self.candidate)

        self.assertTrue(
            verify_mesh("Write/Read", self.model, self.candidate, self.rank)
        )

    def test_periodic_boundcond(self):
        print0(
            f"Using CUDA-AWARE MPI: {ac.device_get_local_config(self.grid_device)[ac.BoolParam.ac_use_cuda_aware_mpi]}",
            file=sys.stderr,
        )
        ac.grid_halo_exchange()
        ac.grid_load_mesh(ac.Stream.default, self.model)
        ac.grid_periodic_boundconds(ac.Stream.default)
        ac.grid_store_mesh(ac.Stream.default, self.candidate)
        if self.rank == 0:
            ac.host_mesh_apply_periodic_bounds(self.model)
        self.assertTrue(
            verify_mesh("Periodic boundconds", self.model, self.candidate, self.rank)
        )

    @unittest.skip("not imlemented")
    def test_dsl_periodic_boundcond(self):
        # const auto periodic = acGetOptimizedDSLTaskGraph(boundconds);
        # acGridLoadMesh(STREAM_DEFAULT, model);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        # acGridExecuteTaskGraphBase(periodic,1,true);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        # acGridStoreMesh(STREAM_DEFAULT, &candidate);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        # if (pid == 0) {
        #     acHostMeshApplyPeriodicBounds(&model);
        #     const AcResult res = acVerifyMesh("DSL Periodic boundconds", model, candidate);
        #     if (res != AC_SUCCESS) {
        #         retval = res;
        #         WARNCHK_ALWAYS(retval);
        #     }
        # }
        pass


class MPITestMain(unittest.TestCase):
    def setUp(self):
        self.nprocs: int = MPI.COMM_WORLD.size
        self.rank: int = MPI.COMM_WORLD.rank

        global model, candidate
        self.model = model
        self.candidate = candidate
        ac.host_grid_mesh_randomize(self.candidate)

        self.grid_device = ac.grid_get_device()

    def tearDown(self):
        pass

    @unittest.skip("not implemented")
    def test_integration(self):
        # // Dryrun
        # const AcReal dt = (AcReal)FLT_EPSILON;
        #
        # acGridIntegrate(STREAM_DEFAULT, dt);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        #
        # for(int substep = 0; substep < 3; ++substep)
        # {
        #   acDeviceSetInput(acGridGetDevice(),AC_SUBSTEP,(AC_SUBSTEP_NUMBER)substep);
        #   AcTaskGraph* dsl_graph = acGetOptimizedDSLTaskGraph(AC_rhs_substep);
        #   acGridExecuteTaskGraph(dsl_graph,1);
        #   acGridSynchronizeStream(STREAM_DEFAULT);
        # }
        #
        #
        # // Integration
        # if (pid == 0)
        #     acHostGridMeshRandomize(&model);
        #
        # acGridLoadMesh(STREAM_DEFAULT, model);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        # acGridPeriodicBoundconds(STREAM_DEFAULT);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        #
        # // Device integrate
        # for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        #     acGridIntegrate(STREAM_DEFAULT, dt);
        #
        # acGridPeriodicBoundconds(STREAM_DEFAULT);
        # acGridStoreMesh(STREAM_DEFAULT, &candidate);
        # if (pid == 0) {
        #     acHostMeshApplyPeriodicBounds(&model);
        #
        #     // Host integrate
        #     for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        #         acHostIntegrateStep(model, dt);
        #
        #     acHostMeshApplyPeriodicBounds(&model);
        #     const AcResult res = acVerifyMeshWithMaximumError("Integration", model, candidate,max_ulp_error);
        #     if (res != AC_SUCCESS) {
        #         retval = res;
        #         WARNCHK_ALWAYS(retval);
        #     }
        # }
        pass

    @unittest.skip("not implemented")
    def test_integration_2(self):
        # // Integration
        # if (pid == 0)
        #     acHostGridMeshRandomize(&model);
        #
        # acGridLoadMesh(STREAM_DEFAULT, model);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        # acGridPeriodicBoundconds(STREAM_DEFAULT);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        #
        # // Device integrate
        # for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        #     acGridIntegrate(STREAM_DEFAULT, dt);
        #
        # acGridPeriodicBoundconds(STREAM_DEFAULT);
        # acGridStoreMesh(STREAM_DEFAULT, &candidate);
        # if (pid == 0) {
        #     acHostMeshApplyPeriodicBounds(&model);
        #
        #     // Host integrate
        #     for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        #         acHostIntegrateStep(model, dt);
        #
        #     acHostMeshApplyPeriodicBounds(&model);
        #     const AcResult res = acVerifyMeshWithMaximumError("Integration", model, candidate,max_ulp_error);
        #     if (res != AC_SUCCESS) {
        #         retval = res;
        #         WARNCHK_ALWAYS(retval);
        #     }
        # }
        # fflush(stdout);
        #
        # acGridSynchronizeStream(STREAM_DEFAULT);
        #
        # // Integration
        # if (pid == 0)
        #     acHostGridMeshRandomize(&model);
        #
        # acGridLoadMesh(STREAM_DEFAULT, model);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        # acGridPeriodicBoundconds(STREAM_DEFAULT);
        # acGridSynchronizeStream(STREAM_DEFAULT);
        #
        # // Device integrate
        # for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        # {
        # for(int substep = 0; substep < 3; ++substep)
        # {
        #     const AcReal start_time = MPI_Wtime();
        #     acDeviceSetInput(acGridGetDevice(),AC_SUBSTEP,(AC_SUBSTEP_NUMBER)substep);
        #             AcTaskGraph* dsl_graph = acGetOptimizedDSLTaskGraph(AC_rhs_substep);
        #             acGridExecuteTaskGraph(dsl_graph,1);
        #     const AcReal end_time = MPI_Wtime();
        #     fprintf(stderr,"Substep %d took %.14e seconds\n",substep,end_time-start_time);
        # }
        # }
        #
        # acGridPeriodicBoundconds(STREAM_DEFAULT);
        # acGridStoreMesh(STREAM_DEFAULT, &candidate);
        # if (pid == 0) {
        #     acHostMeshApplyPeriodicBounds(&model);
        #
        #     // Host integrate
        #     for (size_t i = 0; i < NUM_INTEGRATION_STEPS; ++i)
        #         acHostIntegrateStep(model, dt);
        #
        #     acHostMeshApplyPeriodicBounds(&model);
        #     const AcResult res = acVerifyMeshWithMaximumError("DSL ComputeSteps", model, candidate,max_ulp_error);
        #     if (res != AC_SUCCESS) {
        #         retval = res;
        #         WARNCHK_ALWAYS(retval);
        #     }
        # }
        pass

    @unittest.skip("not implemented")
    def test_reductions_scalar(self):
        # // Scalar reductions
        # if (pid == 0) {
        #     printf("---Test: Scalar reductions---\n");
        #     acHostGridMeshRandomize(&model);
        #     acHostMeshApplyPeriodicBounds(&model);
        # }
        # fflush(stdout);
        # acGridLoadMesh(STREAM_DEFAULT, model);
        # acGridPeriodicBoundconds(STREAM_DEFAULT);
        #
        # const AcReduction scal_reductions[] = {RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS,
        #                                          RTYPE_RMS_EXP};
        # for (size_t i = 0; i < ARRAY_SIZE(scal_reductions); ++i) { // NOTE: not using NUM_RTYPES here
        #     const VertexBufferHandle v0 = (VertexBufferHandle)0;
        #     const auto reduction = scal_reductions[i];
        #
        #     AcReal candval;
        #     acGridReduceScal(STREAM_DEFAULT, reduction, v0, &candval);
        #
        #     if (pid == 0) {
        #         const AcReal modelval = acHostReduceScal(model, reduction, v0);
        #
        #         Error error             = acGetError(modelval, candval);
        #         error.maximum_magnitude = acHostReduceScal(model, RTYPE_MAX, v0);
        #         error.minimum_magnitude = acHostReduceScal(model, RTYPE_MIN, v0);
        #
        #         if (!acEvalErrorWithMaximumError(reduction.name, error, max_ulp_error)) {
        #             fprintf(stderr, "Scalar %s: cand %g model %g\n", reduction.name, (double)candval, (double)modelval);
        #             retval = AC_FAILURE;
        #             WARNCHK_ALWAYS(retval);
        #         }
        #     }
        # }
        # fflush(stdout);
        pass

    @unittest.skip("not implemented")
    def test_reductions_vector(self):
        # // Vector reductions
        # if (pid == 0) {
        #     printf("---Test: Vector reductions---\n");
        # }
        # fflush(stdout);
        #
        # const AcReduction vec_reductions[] = {RTYPE_MAX, RTYPE_MIN, RTYPE_SUM, RTYPE_RMS,
        #                                         RTYPE_RMS_EXP};
        # for (size_t i = 0; i < ARRAY_SIZE(vec_reductions); ++i) { // NOTE: 2 instead of NUM_RTYPES
        #     const VertexBufferHandle v0 = (VertexBufferHandle)0;
        #     const VertexBufferHandle v1 = (VertexBufferHandle)1;
        #     const VertexBufferHandle v2 = (VertexBufferHandle)2;
        #     AcReal candval;
        #
        #     const auto reduction = vec_reductions[i];
        #     acGridReduceVec(STREAM_DEFAULT, reduction, v0, v1, v2, &candval);
        #     if (pid == 0) {
        #         const AcReal modelval = acHostReduceVec(model, reduction, v0, v1, v2);
        #
        #         Error error             = acGetError(modelval, candval);
        #         error.maximum_magnitude = acHostReduceVec(model, RTYPE_MAX, v0, v1, v2);
        #         error.minimum_magnitude = acHostReduceVec(model, RTYPE_MIN, v0, v1, v1);
        #
        #         if (!acEvalErrorWithMaximumError(reduction.name, error, max_ulp_error)) {
        #             fprintf(stderr, "Vector %s: cand %g model %g\n", reduction.name, (double)candval, (double)modelval);
        #             retval = AC_FAILURE;
        #             WARNCHK_ALWAYS(retval);
        #         }
        #     }
        # }
        # fflush(stdout);
        pass

    @unittest.skip("not implemented")
    def test_reductions_alfven(self):
        # if (pid == 0) {
        #     printf("---Test: Alfven reductions---\n");
        # }
        # fflush(stdout);
        #
        # const AcReduction alf_reductions[] = {RTYPE_ALFVEN_MAX, RTYPE_ALFVEN_MIN, RTYPE_ALFVEN_RMS};
        # for (size_t i = 0; i < ARRAY_SIZE(alf_reductions); ++i) { // NOTE: 2 instead of NUM_RTYPES
        #     const VertexBufferHandle v0 = (VertexBufferHandle)0;
        #     const VertexBufferHandle v1 = (VertexBufferHandle)1;
        #     const VertexBufferHandle v2 = (VertexBufferHandle)2;
        #     const VertexBufferHandle v3 = (VertexBufferHandle)3;
        #     AcReal candval;
        #
        #     const auto reduction = alf_reductions[i];
        #     acGridReduceVecScal(STREAM_DEFAULT, reduction, v0, v1, v2, v3, &candval);
        #     if (pid == 0) {
        #         const AcReal modelval = acHostReduceVecScal(model, reduction, v0, v1, v2, v3);
        #
        #         Error error             = acGetError(modelval, candval);
        #         error.maximum_magnitude = acHostReduceVecScal(model, RTYPE_ALFVEN_MAX, v0, v1, v2, v3);
        #         error.minimum_magnitude = acHostReduceVecScal(model, RTYPE_ALFVEN_MIN, v0, v1, v1, v3);
        #
        #         if (!acEvalErrorWithMaximumError(reduction.name, error,max_ulp_error)) {
        #             fprintf(stderr, "Alfven %s: cand %g model %g\n", reduction.name, (double)candval, (double)modelval);
        #             retval = AC_FAILURE;
        #             WARNCHK_ALWAYS(retval);
        #         }
        #     }
        # }
        # fflush(stdout);
        pass


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser("mpitest")

    parser.add_argument(
        "-c", "--config", type=pathlib.Path, default=ac.get_default_config()
    )
    parser.add_argument(
        "dimensions",
        nargs=3,
        type=int,
        default=[2 * 9, 2 * 11, 4 * 7],
    )
    parser.add_argument(
        "--integration-steps",
        required=False,
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-ulp-error",
        required=False,
        type=int,
        default=5,
    )

    global args
    args = parser.parse_args()

    args.config = args.config.absolute()

    unittest.main()


if __name__ == "__main__":
    main()
