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
import sys
import time
from contextlib import AbstractContextManager
from enum import StrEnum
from typing import Final, Optional

import pyastaroth_benchmark.astaroth as ac
from mpi4py import MPI

NUM_ITERATIONS: Final[int] = 100
NS_MS_FACTOR: Final[int] = 1_000_000


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

        if self.name:
            print(f"{self.name}: Time elapsed: {self.duration} ms")


def integrate(dt: float):
    # FIXME: Device-layer API is not bound.
    """
    acDeviceSetInput(acGridGetDevice(),AC_dt,dt);
    """

    for substep in range(0, 3):
        # FIXME: Device-layer API is not bound.
        """
        acDeviceSetInput(acGridGetDevice(),AC_SUBSTEP,(AC_SUBSTEP_NUMBER)substep);
        """
        graph = ac.grid_get_optimized_dsl_task_graph(ac.DSLTaskGraph.ac_rhs_substep)

        with TimedBlock("Substep"):
            # FIXME: Did not get bound
            """
            ac.grid_execute_task_graph(graph, 1)
            """
            pass


def run_benchmark(
    config: pathlib.Path, test_type: TestType, dims: list[int], verify: bool
):
    info: ac.AcMeshInfo = ac.init_info()
    ac.load_config(str(config), info)
    ac.communicator_set_communicator(info.comm, MPI.COMM_WORLD)
    ac.host_update_params(info)

    build_str: str = "-DOPTIMIZE_FIELDS=ON -DOPTIMIZE_INPUT_PARAMS=ON -DELIMINATE_CONDITIONALS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_ACM=OFF"

    ac.compile(build_str, info)
    ac.load_library(info)
    ac.load_utils(info)

    nprocs: int = MPI.COMM_WORLD.Get_size()
    rank: int = MPI.COMM_WORLD.Get_rank()

    decomp: int3 = ac.decompose(nprocs, info)
    if test_type == TestType.STRONG_SCALING:
        # nx, ny, nz -> dimensions
        print("Running strong scaling benchmarks.")

        ac.push_to_config_int3(info, ac.Int3Param.ac_ngrid, decomp)

        nlocal: ac.int3 = ac.int3(
            int(dims[0] / decomp.x), int(dims[1] / decomp.y), int(dims[2] / decomp.z)
        )
        ac.push_to_config_int3(info, ac.Int3Param.ac_nlocal, nlocal)
    else:  # TestType.WEAK_SCALING
        print("Running weak scaling benchmarks.")
        """
        acPushToConfig(info, AC_ngrid, (int3){decomp.x * nx, decomp.y * ny, decomp.z * nz});
        acPushToConfig(info, AC_nlocal, (int3){nx, ny, nz});
        """

    ac.host_update_params(info)

    ac.grid_init(info)
    ac.grid_randomize()

    # FIXME: Most macros do not get bound by default.
    """
    const AcReal dt = (AcReal)FLT_EPSILON;
    """
    dt: float = sys.float_info.epsilon

    # FIXME: The Device-layer API has problems to be bound because of the opaque types.
    """
    // Dryrun
    acDeviceSetInput(acGridGetDevice(), AC_current_time, 0.0);
    """
    integrate(dt)

    if verify:
        model: ac.Mesh = ac.Mesh()
        candidate: ac.Mesh = ac.Mesh()

        if rank == 0:
            ac.host_grid_mesh_create(info, model)
            ac.host_grid_mesh_create(info, candidate)
            ac.host_grid_mesh_randomize(model)
            ac.host_grid_mesh_randomize(candidate)
        else:
            ac.grid_load_mesh(
                ac.Stream._0, model
            )  # FIXME: The STREAM_DEFAULT constant is not bound."

        ac.grid_synchronize_stream(
            ac.Stream._0
        )  # FIXME: The STREAM_DEFAULT constant is not bound."
        ac.grid_periodic_boundconds(
            ac.Stream._0
        )  # FIXME: The STREAM_DEFAULT constant is not bound."
        ac.grid_synchronize_stream(
            ac.Stream._0
        )  # FIXME: The STREAM_DEFAULT constant is not bound."

        for i in range(0, 10):
            integrate(dt)
            if rank == 0:
                print(f"Host integration step {i}")
                ac.host_mesh_apply_periodic_bounds(model)
                ac.host_integrate_step(model, dt)

        ac.grid_periodic_boundconds(
            ac.Stream._0
        )  # FIXME: The STREAM_DEFAULT constant is not bound."
        ac.grid_store_mesh(
            ac.Stream._0, candidate
        )  # FIXME: The STREAM_DEFAULT constant is not bound."
        ac.grid_periodic_boundconds(
            ac.Stream._0
        )  # FIXME: The STREAM_DEFAULT constant is not bound."
        ac.grid_synchronize_stream(
            ac.Stream._0
        )  # FIXME: The STREAM_DEFAULT constant is not bound."

        if rank == 0:
            ac.host_mesh_apply_periodic_bounds(model)
            print("Verifying...")

            # FIXME: This would look better if AcResult was used for making
            # exceptions.
            success = (
                ac.verify_mesh("Integration", model, candidate)
                == ac.AcResult.ac_success
            )
            # FIXME: Get rid of explicit memory management. Python can cleanup
            # automatically.
            ac.host_mesh_destroy(model)
            ac.host_mesh_destroy(candidate)
            if not success:
                print("Failures found, benchmark invalid. Skipping.", file=sys.stderr)
                return 1

            print("Verification done - everything OK")

    # Warmup
    for i in range(0, 5):
        integrate(dt)

    # Benchmark
    results: list[float] = []
    for i in range(NUM_ITERATIONS):
        with TimedBlock() as timed_block:
            integrate(dt)
        results.append(timed_block.duration)

    if rank == 0:
        results.sort()

        def print_report(nth_percentile: float) -> None:
            print(
                f"Integration step time {results[int(nth_percentile * NUM_ITERATIONS)]}"
                f" ({100 * nth_percentile}th percentile)"
            )

        print_report(0.5)
        print_report(0.9)

    MPI.COMM_WORLD.Barrier()

    if rank == 0:
        print("Sanity performance check:")

    mesh_dims: ac.MeshDims = ac.get_mesh_dims(info)

    with TimedBlock("acGridPeriodicBoundconds"):
        # FIXME: The Device-layer API has problems to be bound because of the opaque types.
        """
        acDevicePeriodicBoundconds(acGridGetDevice(), STREAM_DEFAULT, mesh_dims.m0, mesh_dims.m1);
        """

    with TimedBlock("acGridIntegrate"):
        integrate(dt)

    with TimedBlock("acGridReduceScal"):
        candval: float = 0
        # FIXME: Unsure how to work with the parameters.
        """
        ac.grid_reduce_scal(ac.Stream._0, RTYPE_SUM, (Field)0, candval);
        """

    with TimedBlock("acGridReduceVec"):
        # FIXME: Unsure how to work with the parameters.
        """
        ac.grid_reduce_vec(ac.Stream._0, RTYPE_SUM, (Field)0, (Field)1, (Field)2, &candval);
        """

    ac.grid_quit()
    ac.mpi_finalize()
