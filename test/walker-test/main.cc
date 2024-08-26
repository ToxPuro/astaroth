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
#include "matplotlib-cpp/matplotlibcpp.h"
namespace plt = matplotlibcpp;


#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(*arr))
#define NUM_INTEGRATION_STEPS (100)

static bool finalized = false;

#include <stdlib.h>
void
acAbort(void)
{
    if (!finalized)
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
}
#include "user_constants.h"

constexpr AcReal boxxlength = lsf * sf * 10 * dx; // set the boxlength of one box xdir
constexpr AcReal boxylength = lsf * sf * 10 * dy; // set the boxlength of one box ydir
constexpr int boxesx = (int) (lengthx / boxxlength); // calculate number of boxes in x dir
constexpr int boxesy = (int) (lengthy / boxylength); // calculate number of boxes in y dir
int natoms[boxesx-1][boxesy-1]; // number of atoms in each box
// // coordinates of box centers
AcReal boxcentersx[boxesx-1][boxesy-1];
AcReal boxcentersy[boxesx-1][boxesy-1];

constexpr int
int_pow(const int base, const int exponent)
{
        int res = base;
        for(int i = 1; i < exponent; ++i)
                res *= base;
        return res;

}


int
main(void)
{

    MPI_Init(NULL,NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    int retval = 0;




    // Set random seed for reproducibility
    srand(321654987);

    // CPU alloc
    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);
    info.real_params[AC_dsx] = dx;
    info.real_params[AC_dsx] = dy;
    info.int_params[AC_proc_mapping_strategy] = (int)AcProcMappingStrategy::Linear;

    const int max_devices = 1;
    if (nprocs > max_devices) {
        fprintf(stderr,
                "Cannot run autotest, nprocs (%d) > max_devices (%d). Please modify "
                "mpitest/main.cc to use a larger mesh.\n",
                nprocs, max_devices);
        MPI_Abort(acGridMPIComm(), EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    acSetMeshDims(npoints_x, npoints_y, 1, &info);
    //acSetMeshDims(44, 44, 44, &info);

    AcMesh model, candidate;
    if (pid == 0) {
        acHostMeshCreate(info, &model);
        acHostMeshCreate(info, &candidate);
        acHostMeshRandomize(&model);
        acHostMeshRandomize(&candidate);
    }

    // GPU alloc & compute
    bool exists[npoints] = {true};
    info.bool_arrays[AC_exists] = exists;
    AcReal global_radius_host[1] = {init_radius};
    info.real_arrays[global_radius_start] = global_radius_host;
    info.real_arrays[global_radius_tmp]   = global_radius_host;
    acGridInit(info);

    // Load/Store
    //acGridLoadMesh(STREAM_DEFAULT, model);
    acDeviceLoadMesh(acGridGetDevice(), STREAM_DEFAULT,model);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &candidate);
    if (pid == 0) {
        const AcResult res = acVerifyMesh("Load/Store", model, candidate);
        if (res != AC_SUCCESS) {
            retval = res;
            WARNCHK_ALWAYS(retval);
        }
    }

    std::array<Field,NUM_VTXBUF_HANDLES> fields_array = get_vtxbuf_handles();
    std::vector<Field> all_fields(fields_array.begin(), fields_array.end());
    int steps = 0;
    auto loader = [&](auto p)
    {
	    p.params -> solve.step_num = steps;
    };
    AcTaskGraph* solve_graph = acGridBuildTaskGraph({
		    	acCompute(KERNEL_solve,all_fields,loader)
		    });


    //const std::vector<int> simlen = {int_pow(10,3), int_pow(10,4)};
    const std::vector<int> simlen = {int_pow(10,3)};
    //const std::vector<int> simlen = {int_pow(10,3),int_pow(10,4)};
    std::vector<std::vector<double>> atoms_per_area;
    std::vector<std::vector<double>> box_center_x;
    for(size_t i = 0; i < simlen.size(); ++i)
    {
    	std::vector<double> tmp(boxesx-1);
    	atoms_per_area.push_back(tmp);
    	box_center_x.push_back(tmp);
    }
    for (size_t ww = 0; ww < simlen.size(); ww++) 
    {
	++steps;
        const int nsteps = (ww == 0) ? simlen[ww] : simlen[ww]-simlen[ww-1];
	for(int step = 0; step < nsteps; ++step)
		acGridExecuteTaskGraph(solve_graph,1);
	acGridSynchronizeStream(STREAM_ALL);
    	acGridStoreMesh(STREAM_DEFAULT, &model);
	acGridSynchronizeStream(STREAM_ALL);
	acStoreUniform(AC_exists, exists, get_array_length(AC_exists,model.info));
	acGridSynchronizeStream(STREAM_ALL);
    }
    if (pid == 0)
        fprintf(stderr, "MPITEST complete: %s\n",
                retval == AC_SUCCESS ? "No errors found" : "One or more errors found");
    exit(EXIT_SUCCESS);
    /**

	 // calculate how many atoms are in each box (to get densities by dividing the number of atoms by the box size if necessary)
        for (int ii = 0; ii < boxesx - 1; ii++) {
            for (int jj = 0; jj < boxesy - 1; jj++) {
                natoms[ii][jj] = 0;
                for (int uu = 0; uu < npoints; uu++) {
                    const double boxcenterx = boxcentersx[ii][jj];
                    const double boxcentery = boxcentersy[ii][jj];
                    int inside_box = ((model.vertex_buffer[COORDS_X][uu] > boxcenterx - boxxlength / 2.0 && model.vertex_buffer[COORDS_X][uu] < boxcenterx + boxxlength / 2.0) && (model.vertex_buffer[COORDS_Y][uu] > boxcentery - boxylength / 2.0 && model.vertex_buffer[COORDS_Y][uu] < boxcentery + boxylength / 2.0));
                    natoms[ii][jj] += inside_box*(exists[uu] == 1);
                }
            }
        }
        for (int ii = 0; ii < boxesx-1; ii++) { // print the number of atoms in the boxes aligned in horizontal direction through the middle of the sim. domain in y dir
            printf("%d,",natoms[ii][boxesy/2]);
        }
        printf("\n");
        for (int ii = 0; ii < boxesx - 1; ii++)
        {
                atoms_per_area[ww][ii] = natoms[ii][boxesy/2]/(boxxlength*boxylength);
                box_center_x[ww][ii] = boxcentersx[ii][boxesy/2];
        }
    }
    for(size_t i = 0; i < simlen.size(); ++i)
    {
    	char name_buffer[4000];
    	sprintf(name_buffer, "%d steps",simlen[i]);
    	plt::named_plot(name_buffer,box_center_x[i], atoms_per_area[i]);
    }
    plt::xlabel("x coordinate [a]");
    plt::ylabel("Atoms/area [1/$a^2$]");
    plt::legend();

    char name_buffer[4000];
    sprintf(name_buffer,"res_%d",sf);
    plt::save(name_buffer);

    plt::show();

    acGridQuit();
    ac_MPI_Finalize();
    fflush(stdout);
    finalized = true;
    

    return EXIT_SUCCESS;
    **/
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
