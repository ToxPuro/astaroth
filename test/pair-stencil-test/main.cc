/*
    Copyright (C) 2014-2026, Johannes Pekkila, Touko Puro, Miikka Vaisala.

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
    Tests PairSum stencils: pair_stencil(a,b) = sum_i w_i * a(p_i) * b(p_i)
    and the staggered derivatives of the product of two fields of general_derivs.h
    (derx_shifted(a,b) etc.) against the reference formula:

        coef = [9./8.,-1./24.]
        c1x=coef(1)/dx; c2x=coef(2)/dx
        dex_dx = c1x*(a(i)*b(i)-a(i-1)*b(i-1)) + c2x*(a(i+1)*b(i+1)-a(i-2)*b(i-2))

    Running: mpirun -np <num processes> <executable>
*/
#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED

#include <cmath>
#include <functional>
#include <mpi.h>

static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

// Same points ([z][y][x] offsets) and coefficients as the stencils in DSL/solver.ac
struct StencilPoint {
    int z, y, x;
    AcReal coeff;
};
static const StencilPoint points[] = {
    {0, 0, 0, 2.0},  {0, 0, -3, -0.5}, {0, 0, 1, 1.5},   {0, -2, 0, 0.25},
    {0, 3, 0, -1.25}, {-1, 0, 0, 0.75}, {3, 0, 0, -2.5}, {1, -1, 2, 0.125},
};

int
main(void)
{
    atexit(acAbort);
    int retval = 0;

    int nprocs, pid;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    // Set random seed for reproducibility
    srand(321654987);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    acPushToConfig(info, AC_MPI_comm_strategy, AC_MPI_COMM_STRATEGY_DUP_WORLD);
    // Different spacing in every direction to catch mixed up directions in the derivatives
    acPushToConfig(info, AC_ds, (AcReal3){0.5, 0.25, 0.125});
    info.comm->handle = MPI_COMM_WORLD;

    acSetGridMeshDims(16, 16, 16, &info);
    const int3 decomp = acDecompose(nprocs, info);
    acSetLocalMeshDims(16 / decomp.x, 16 / decomp.y, 16 / decomp.z, &info);

    // Global meshes: acGridLoadMesh scatters from and acGridStoreMesh gathers to the root
    AcMesh model, candidate;
    acHostGridMeshCreate(info, &model);
    acHostGridMeshCreate(info, &candidate);
    acHostMeshRandomize(&model);

    acGridInit(info);
    const AcReal3 ds = acGridGetLocalMeshInfo()[AC_ds];
    acGridLoadMesh(STREAM_DEFAULT, model);
    acGridSynchronizeStream(STREAM_ALL);
    acGridExecuteTaskGraph(acGetOptimizedDSLTaskGraph(pair_step), 1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    acGridSynchronizeStream(STREAM_ALL);
    acGridQuit();

    if (pid == 0) {
        const AcMeshDims dims = acGetGridMeshDims(info);
        // Periodic wrap in global coordinates since the halos of the model are random
        auto wrap = [](const int i, const size_t n0, const size_t nn) {
            return (int)n0 + (((i - (int)n0) % (int)nn) + (int)nn) % (int)nn;
        };
        auto at = [&](const AcReal* buf, const int i, const int j, const int k) {
            return buf[acGridVertexBufferIdx(wrap(i, dims.n0.x, dims.nn.x),
                                             wrap(j, dims.n0.y, dims.nn.y),
                                             wrap(k, dims.n0.z, dims.nn.z), info)];
        };
        // b == NULL: ordinary single-field stencil
        auto reference = [&](const AcReal* a, const AcReal* b, const int i, const int j,
                             const int k) {
            long double res = 0;
            for (const auto& p : points) {
                const long double va = at(a, i + p.x, j + p.y, k + p.z);
                const long double vb = b ? at(b, i + p.x, j + p.y, k + p.z) : 1.0L;
                res += p.coeff * va * vb;
            }
            return (AcReal)res;
        };
        // dir: the unit offset of the derivative direction
        auto shifted_reference = [&](const int3 dir, const AcReal spacing, const int i,
                                     const int j, const int k) {
            const long double c1 = (9.0L / 8.0L) / spacing;
            const long double c2 = (-1.0L / 24.0L) / spacing;
            auto ab = [&](const int shift) {
                const int ii = i + shift * dir.x, jj = j + shift * dir.y, kk = k + shift * dir.z;
                return (long double)at(model.vertex_buffer[LDAP2MU], ii, jj, kk) *
                       (long double)at(model.vertex_buffer[EX], ii, jj, kk);
            };
            return (AcReal)(c1 * (ab(0) - ab(-1)) + c2 * (ab(1) - ab(-2)));
        };

        // Tolerances relative to the magnitude of the sum of |w_i * a * b| (random values are in [0,1])
        AcReal coeff_sum = 0;
        for (const auto& p : points)
            coeff_sum += std::fabs(p.coeff);
        const AcReal shifted_coeff_sum = 2 * (9.0 / 8.0 + 1.0 / 24.0);
        const AcReal* a = model.vertex_buffer[A];
        const AcReal* b = model.vertex_buffer[B];

        const struct {
            const char* name;
            Field out;
            std::function<AcReal(int, int, int)> ref;
            AcReal coeff_sum;
        } cases[] = {
            {"pair_stencil(A,B)", OUT_AB, [&](int i, int j, int k) { return reference(a, b, i, j, k); }, coeff_sum},
            {"pair_stencil(B,A)", OUT_BA, [&](int i, int j, int k) { return reference(b, a, i, j, k); }, coeff_sum},
            {"pair_stencil(A,A)", OUT_AA, [&](int i, int j, int k) { return reference(a, a, i, j, k); }, coeff_sum},
            {"plain_stencil(A)", OUT_PLAIN, [&](int i, int j, int k) { return reference(a, NULL, i, j, k); }, coeff_sum},
            {"derx_shifted(LDAP2MU,EX)", OUT_X,
             [&](int i, int j, int k) { return shifted_reference((int3){1, 0, 0}, ds.x, i, j, k); },
             shifted_coeff_sum / ds.x},
            {"dery_shifted(LDAP2MU,EX)", OUT_Y,
             [&](int i, int j, int k) { return shifted_reference((int3){0, 1, 0}, ds.y, i, j, k); },
             shifted_coeff_sum / ds.y},
            {"derz_shifted(LDAP2MU,EX)", OUT_Z,
             [&](int i, int j, int k) { return shifted_reference((int3){0, 0, 1}, ds.z, i, j, k); },
             shifted_coeff_sum / ds.z},
        };

        for (const auto& c : cases) {
            const AcReal epsilon = 64 * AC_REAL_EPSILON * c.coeff_sum;
            AcReal max_err = 0;
            size_t num_wrong = 0;
            for (size_t k = dims.n0.z; k < dims.n1.z; ++k) {
                for (size_t j = dims.n0.y; j < dims.n1.y; ++j) {
                    for (size_t i = dims.n0.x; i < dims.n1.x; ++i) {
                        const AcReal ref = c.ref(i, j, k);
                        const AcReal val =
                            candidate.vertex_buffer[c.out][acGridVertexBufferIdx(i, j, k, info)];
                        const AcReal err = std::fabs(ref - val);
                        if (!(err <= epsilon)) {
                            if (num_wrong < 10)
                                fprintf(stderr, "%s WRONG at (%zu,%zu,%zu): %.14e vs. %.14e\n",
                                        c.name, i, j, k, (double)val, (double)ref);
                            ++num_wrong;
                        }
                        if (err > max_err || std::isnan(err))
                            max_err = err;
                    }
                }
            }
            fprintf(stderr, "%s: max abs error %g (tolerance %g), %zu wrong points\n", c.name,
                    (double)max_err, (double)epsilon, num_wrong);
            if (num_wrong)
                retval = AC_FAILURE;
        }
    }
    MPI_Bcast(&retval, 1, MPI_INT, 0, MPI_COMM_WORLD);

    acHostMeshDestroy(&model);
    acHostMeshDestroy(&candidate);

    MPI_Finalize();
    finalized = true;

    if (pid == 0)
        fprintf(stderr, "PAIR STENCIL TEST complete: %s\n",
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
