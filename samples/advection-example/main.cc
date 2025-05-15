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

#include <mpi.h>
#include <vector>
#include <string>

int
main()
{
    // CPU alloc
    AcMeshInfo info;
    acLoadConfig("advec.conf", &info);
    info[AC_ds] = 
	    (AcReal3)
	    {
	    	info[AC_len].x/info[AC_ngrid].x,
	    	info[AC_len].y/info[AC_ngrid].x,
	    	info[AC_len].z/info[AC_ngrid].z
	    };

    acHostUpdateParams(&info);
    // GPU alloc & compute;
    acGridInit(info);

    AcTaskGraph* init_graph = acGetDSLTaskGraph(initial_condition);
    AcTaskGraph* update_graph = acGetDSLTaskGraph(euler_update);


    AcReal simulation_time =  0.0;
    acGridExecuteTaskGraph(init_graph,1);
    int i = 0;
    while(simulation_time < info[AC_max_time])
    {
	if((i % info[AC_slice_steps]) == 0) acGridWriteSlicesToDiskSynchronous("slices",i,simulation_time);
        acGridExecuteTaskGraph(update_graph,1);
	simulation_time += info[AC_dt];
	++i;
    }
    if((i % info[AC_slice_steps]) != 0) acGridWriteSlicesToDiskSynchronous("slices",i,simulation_time);
    acGridQuit();
    printf("Test simulation done!\n");
    return EXIT_SUCCESS;
}

