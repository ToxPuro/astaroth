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

import pathlib
import statistics
import sys
import time
from contextlib import AbstractContextManager
from enum import StrEnum
from typing import Final, Optional

import benchmark.astaroth as ac
from mpi4py import MPI

BUILD_STR: Final[str] = (
    "-DOPTIMIZE_FIELDS=ON -DOPTIMIZE_INPUT_PARAMS=ON -DELIMINATE_CONDITIONALS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_ACM=OFF"
)
BENCHMARK_NUM_ITERATIONS: Final[int] = 100
VERIFICATION_NUM_ITERATIONS: Final[int] = 10
NS_MS_FACTOR: Final[int] = 1_000_000


def print0(*args, **kwargs):
    if MPI.COMM_WORLD.rank != 0:
        return
    print(*args, **kwargs)


class TestType(StrEnum):
    STRONG_SCALING = "strong"
    WEAK_SCALING = "weak"


class TimedBlock(AbstractContextManager):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __enter__(self):
        ac.grid_synchronize_stream(ac.Stream.all)

        self.start: int = time.perf_counter_ns()

        return self

    def __exit__(self, *exc_details):
        ac.grid_synchronize_stream(ac.Stream.all)

        self.end: int = time.perf_counter_ns()
        self.duration: float = (self.end - self.start) / NS_MS_FACTOR

        rank: int = MPI.COMM_WORLD.rank

        durations = MPI.COMM_WORLD.gather(self.duration, root=0)
        if not durations:
            return

        if self.name and rank == 0:
            if len(durations) > 1:
                mean = statistics.mean(durations)
                variance = statistics.pvariance(durations, mean)

                print0(
                    f"{self.name}: Time elapsed (mean) {mean:.5f} (+-{variance:.5f}) ms"
                )
            else:
                print0(f"{self.name}: Time elapsed {self.duration:.5f} ms")


def integrate(dt: float):
    ac.device_set_input(ac.grid_get_device(), ac.RealInputParam.ac_dt, dt)

    for substep in range(0, 3):
        ac.device_set_input(
            ac.grid_get_device(),
            ac.AC_SUBSTEP_NUMBERInputParam.ac_substep,
            ac.AC_SUBSTEP_NUMBER(substep),
        )

        graph = ac.get_optimized_dsl_task_graph_base(ac.DSLTaskGraph.ac_rhs_substep)

        with TimedBlock(f"Substep {substep}"):
            ac.grid_execute_task_graph(graph, 1)


def run_benchmark(
    config: pathlib.Path, test_type: TestType, dims: list[int], verify: bool
) -> bool:
    info: ac.AcMeshInfo = ac.init_info()
    ac.load_config(str(config), info)
    ac.host_update_params(info)

    # FIXME: Make the acPushToConfig work similarly to acDeviceSetInput. Needs
    # overloads.
    # acPushToConfig(info,AC_proc_mapping_strategy, AC_PROC_MAPPING_STRATEGY_MORTON);
    # acPushToConfig(info,AC_decompose_strategy,    AC_DECOMPOSE_STRATEGY_MORTON);
    # acPushToConfig(info,AC_MPI_comm_strategy,     AC_MPI_COMM_STRATEGY_DUP_WORLD);
    ac.communicator_set_communicator(info.comm, MPI.COMM_WORLD)

    ac.compile(BUILD_STR, info)
    ac.load_library(info)
    ac.load_utils(info)

    nprocs: int = MPI.COMM_WORLD.Get_size()
    rank: int = MPI.COMM_WORLD.Get_rank()

    print0(f"Benchmark mesh dimensions: ({dims})")

    decomp: ac.int3 = ac.decompose(nprocs, info)
    if test_type == TestType.STRONG_SCALING:
        print0("Running strong scaling benchmarks.", file=sys.stderr)

        ngrid: ac.int3 = ac.int3(*dims)
        nlocal: ac.int3 = ac.int3(
            int(dims[0] / decomp.x), int(dims[1] / decomp.y), int(dims[2] / decomp.z)
        )
    else:  # TestType.WEAK_SCALING
        print0("Running weak scaling benchmarks.", file=sys.stderr)

        ngrid: ac.int3 = ac.int3(
            dims[0] * decomp.x, dims[1] * decomp.y, dims[2] * decomp.z
        )
        nlocal: ac.int3 = ac.int3(*dims)
    ac.set_grid_mesh_dims(ngrid.x, ngrid.y, ngrid.z, info)
    ac.set_local_mesh_dims(nlocal.x, nlocal.y, nlocal.z, info)

    ac.grid_init(info)
    ac.grid_randomize()

    # Constant timestep
    dt: float = sys.float_info.epsilon

    print0("Dry run:", file=sys.stderr)
    ac.device_set_input(ac.grid_get_device(), ac.RealInputParam.ac_current_time, 0)
    integrate(dt)

    if verify:
        print0("\nVerification:", file=sys.stderr)

        model: ac.Mesh = ac.Mesh()
        candidate: ac.Mesh = ac.Mesh()

        if rank == 0:
            ac.host_grid_mesh_create(info, model)
            ac.host_grid_mesh_create(info, candidate)
            ac.host_grid_mesh_randomize(model)

        ac.grid_load_mesh(ac.Stream.default, model)
        ac.grid_synchronize_stream(ac.Stream.default)

        ac.grid_periodic_boundconds(ac.Stream.default)
        ac.grid_synchronize_stream(ac.Stream.default)

        for i in range(VERIFICATION_NUM_ITERATIONS):
            integrate(dt)

            print0(f"Host integration step {i}")
            if rank == 0:
                ac.host_mesh_apply_periodic_bounds(model)
                ac.host_integrate_step(model, dt)

        ac.grid_periodic_boundconds(ac.Stream.default)
        ac.grid_store_mesh(ac.Stream.default, candidate)
        ac.grid_synchronize_stream(ac.Stream.all)

        print0("Verifying...", file=sys.stderr)
        success: bool = False
        if rank == 0:
            ac.host_mesh_apply_periodic_bounds(model)

            # FIXME: This would look better if AcResult was used for making
            # exceptions.
            success = (
                ac.verify_mesh("Integration", model, candidate) == ac.Result.ac_success
            )
            # FIXME: Get rid of explicit memory management. Python can cleanup
            # automatically.
            ac.host_mesh_destroy(model)
            ac.host_mesh_destroy(candidate)

        success = MPI.COMM_WORLD.bcast(success, root=0)
        if not success:
            print0(
                "Failures found, benchmark invalid. Skipping the rest.", file=sys.stderr
            )
            return False

        print0("Verification done - everything OK", file=sys.stderr)

    print0("\nWarmup:")
    for i in range(0, 5):
        integrate(dt)

    print0("\nBenchmark:")
    results: list[float] = []
    for i in range(BENCHMARK_NUM_ITERATIONS):
        ac.grid_synchronize_stream(ac.Stream.all)
        with TimedBlock() as timed_block:
            integrate(dt)
        ac.grid_synchronize_stream(ac.Stream.all)

        results.append(timed_block.duration)

    if rank == 0:
        results.sort()

        def print_report(nth_percentile: float) -> None:
            print0(
                f"Integration step time {results[int(nth_percentile * BENCHMARK_NUM_ITERATIONS)]}"
                f" ({100 * nth_percentile}th percentile)"
            )

        print_report(0.5)
        print_report(0.9)

    MPI.COMM_WORLD.Barrier()

    print0("\nSanity performance check:")

    mesh_dims: ac.MeshDims = ac.get_mesh_dims(info)

    with TimedBlock("acGridPeriodicBoundconds"):
        # Causes segfaults on ac.grid_quit() when deallocating reals scratchpads
        # ac.device_periodic_boundconds(ac.grid_get_device(), ac.Stream.default, mesh_dims.m0, mesh_dims.m1)
        pass

    with TimedBlock("acGridIntegrate"):
        integrate(dt)

    # with TimedBlock("acGridReduceScal"):
    #    candval: float = 0
    # FIXME: Unsure how to work with the parameters. RTYPE_SUM comes from stlib/reductions.h and is not always
    # included (WHYYYYYYY?!)
    # ac.grid_reduce_scal(ac.Stream.default, RTYPE_SUM, ac.Field(0), candval)

    # with TimedBlock("acGridReduceVec"):
    # FIXME: Unsure how to work with the parameters.
    # ac.grid_reduce_vec(ac.Stream.default, RTYPE_SUM, ac.Field(0), ac.Field(1), ac.Field(2), candval);

    ac.grid_quit()
    ac.mpi_finalize()

    return True
