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
import sys
import time
from contextlib import AbstractContextManager
from enum import StrEnum
from typing import Final, Optional

import pyastaroth as core
import pyastaroth.grid as grid
import pyastaroth.utils as utils
from mpi4py import MPI

CONST_NUM_ITERATIONS: Final[int] = 100
CONST_NS_MS_FACTOR: Final[int] = 1_000_000

class TestType(StrEnum):
    STRONG_SCALING = "strong"
    WEAK_SCALING = "weak"


class TimedBlock(AbstractContextManager):
    def __init__(self, name: Optional[str] = None):
        self.name = name

    def __enter__(self):
        grid.grid_synchronize_stream(core.Stream.all)

        self.start: int = time.perf_counter_ns()

    def __exit__(self, *exc_details):
        grid.grid_synchronize_stream(core.Stream.all)

        self.end: int = time.perf_counter_ns()
        self.duration: int = (self.end - self.start) * CONST_NS_MS_FACTOR

        if self.name:
            print(f"{name}: Time elapsed: {self.duration} ms")


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
        grid.get_optimized_dsl_task_graph(core.AcDSLTaskGraph.ac_rhs_substep)
        start: float = MPI.Wtime()
        
        # FIXME: Did not get bound
        """
        acGridExecuteTaskGraph(dsl_graph,1);
        """
        end: float = MPI.Wtime()

        print(f"Substep took {end-start}", file=sys.stderr)

def main():
    parser : argparse.ArgumentParser = argparse.ArgumentParser("benchmark")

    parser.add_argument("dimensions", nargs=3, type=int)
    parser.add_argument(
        "-t",
        "--type",
        required=False,
        type=TestType,
        choices=list(TestType),
        default=TestType.STRONG_SCALING,
    )
    parser.add_argument(
        "--verify",
        required=False,
        action="store_true",
    )

    args: argparse.Namespace = parser.parse_args()

    grid.mpi_init()

    nprocs: int = grid.comm_size()
    rank: int = grid.comm_rank()

    info: core.AcMeshInfo = core.init_info()
    utils.ac_load_config("", info)
    core.host_update_params(info)

    # FIXME: Cannot be bound using litgen and is not in the C API.
    """
    acPushToConfig(info,AC_proc_mapping_strategy, AC_PROC_MAPPING_STRATEGY_MORTON);
    acPushToConfig(info,AC_decompose_strategy,    AC_DECOMPOSE_STRATEGY_MORTON);
    acPushToConfig(info,AC_MPI_comm_strategy,     AC_MPI_COMM_STRATEGY_DUP_WORLD);
    """

    info.comm.handle = MPI.COMM_WORLD

    build_str: str = "-DOPTIMIZE_FIELDS=ON -DOPTIMIZE_INPUT_PARAMS=ON -DELIMINATE_CONDITIONALS=ON -DOPTIMIZE_ARRAYS=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_SHARED_LIBS=ON -DMPI_ENABLED=ON -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_ACM=OFF"
    core.compile(build_str, info)

    # FIXME: Does not show up in the bindings. Why? And can this be made
    # automatic? Can we have a acIsLoaded() function so that we don't have to
    # rely on macros to discern whether runtime compilation is needed?
    """
    core.load_library()
    core.load_utils()
    """

    decomp: int3 = core.decompose(nprocs, info)
    if args.type == TestType.STRONG_SCALING:
        print("Running strong scaling benchmarks.");

        """
        acPushToConfig(info, AC_ngrid, (int3){nx, ny, nz});
        acPushToConfig(info, AC_nlocal,
                       (int3){nx / decomp.x, ny / decomp.y, nz / decomp.z});
        """
    else: # TestType.WEAK_SCALING
        print("Running weak scaling benchmarks.");

        """
        acPushToConfig(info, AC_ngrid, (int3){decomp.x * nx, decomp.y * ny, decomp.z * nz});
        acPushToConfig(info, AC_nlocal, (int3){nx, ny, nz});
        """

    core.host_update_params(info)

    grid.grid_init(info)
    grid.grid_randomize()

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

    if args.verify:
        model: AcMesh = core.AcMesh()
        candidate: AcMesh = core.AcMesh()

        if rank == 0:
            core.host_grid_mesh_create(info, model)
            core.host_grid_mesh_create(info, candidate)
            core.host_grid_mesh_randomize(model)
            core.host_grid_mesh_randomize(candidate)
        else:
            grid.grid_load_mesh(core.Stream._0, model) # FIXME: The STREAM_DEFAULT constant is not bound."

        grid.grid_synchronize_stream(core.Stream._0) # FIXME: The STREAM_DEFAULT constant is not bound."
        grid.grid_periodic_boundconds(core.Stream._0) # FIXME: The STREAM_DEFAULT constant is not bound."
        grid.grid_synchronize_stream(core.Stream._0) # FIXME: The STREAM_DEFAULT constant is not bound."

        for i in range(0, 10):
            integrate(dt)
            if rank == 0:
                print(f"Host integration step {i}")
                utils.host_mesh_apply_periodic_bounds(model)
                utils.host_integrate_step(model, dt)

        grid.grid_periodic_boundconds(core.Stream._0) # FIXME: The STREAM_DEFAULT constant is not bound."
        grid.grid_store_mesh(core.Stream._0, candidate) # FIXME: The STREAM_DEFAULT constant is not bound."
        grid.grid_periodic_boundconds(core.Stream._0) # FIXME: The STREAM_DEFAULT constant is not bound."
        grid.grid_synchronize_stream(core.Stream._0) # FIXME: The STREAM_DEFAULT constant is not bound."

        if rank == 0:
            utils.host_mesh_apply_periodic_bounds(model)
            print("Verifying...")

            # FIXME: This would look better if AcResult was used for making
            # exceptions.
            success = utils.verify_mesh("Integration", model, candidate) == core.AcResult.ac_success
            # FIXME: Get rid of explicit memory management. Python can cleanup
            # automatically.
            core.host_mesh_destroy(model)
            core.host_mesh_destroy(candidate)
            if not success:
                print("Failures found, benchmark invalid. Skipping.", file=sys.stderr)
                return 1

            print("Verification done - everything OK")

    # Warmup
    for i in range(0, 5):
        integrate(dt)

    # Benchmark
    results: list[float] = []
    for i in range(CONST_NUM_ITERATIONS):
        with TimedBlock() as timed_block:
            integrate(dt)
        results.append(timed_block.duration)

    if rank == 0:
        results.sort()

        def print_report(nth_percentile: float) -> None:
            print(
                f"Integration step time {results[int(nth_percentile * CONST_NUM_ITERATIONS)]}"
                f" ({100 * nth_percentile}th percentile)"
            )

        print_report(0.5)
        print_report(0.9)

    MPI.COMM_WORLD.Barrier()

    if rank == 0:
        print("Sanity performance check:")

    dims: core.AcMeshDims = core.get_mesh_dims(info)

    with TimedBlock("acGridPeriodicBoundconds"):
        # FIXME: The Device-layer API has problems to be bound because of the opaque types.
        """
        acDevicePeriodicBoundconds(acGridGetDevice(), STREAM_DEFAULT, dims.m0, dims.m1);
        """

    with TimedBlock("acGridIntegrate"):
        integrate(dt)

    with TimedBlock("acGridReduceScal"):
        candval: float = 0
        # FIXME: Unsure how to work with the parameters.
        """
        grid.grid_reduce_scal(core.Stream._0, RTYPE_SUM, (Field)0, candval);
        """

    with TimedBlock("acGridReduceVec"):
        # FIXME: Unsure how to work with the parameters.
        """
        grid.grid_reduce_vec(core.Stream._0, RTYPE_SUM, (Field)0, (Field)1, (Field)2, &candval);
        """

    grid.grid_quit()
    grid.mpi_finalize()

if __name__ == "__main__":
    main()
