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
    //AC_ds is a real3 declaration and a entry is generated for it in the info object
    //updating that entry is the way to get the values to the device.
    //
    //Here we calculate the grid spacings based on the length and number of points.
    //These are specified from the configuration file which is loaded with the above call of acLoadConfig
    acPushToConfig(info,AC_ds,
	    (AcReal3)
	    {
	    	info[AC_len].x/info[AC_ngrid].x,
	    	info[AC_len].y/info[AC_ngrid].x,
	    	info[AC_len].z/info[AC_ngrid].z
	    };

    //This function call is needed to get the default values of the variables as defined in the DSL
    //If you upload some values with acPushToConfig those take precedence over the default values of the DSL
    acHostUpdateParams(&info);

    //Allocates buffers on the device, setups MPI and all other initialization
    acGridInit(info);

    //We now generate handles out of the ComputeSteps defined in the DSL
    //TaskGraphs are the objects to be invoked which include the required bcs and communications, which are added to the compute steps as needed by the operations
    AcTaskGraph* init_graph = acGetDSLTaskGraph(initial_condition);
    AcTaskGraph* update_graph = acGetDSLTaskGraph(euler_update);


    AcReal simulation_time =  0.0;

    //This executes the sequence of steps in the compute steps
    acGridExecuteTaskGraph(init_graph,1);
    int i = 0;
    while(simulation_time < info[AC_max_time])
    {
	//Save the XY slices of the buffers from the GPU to the disk
	//Here is an example where it is nice to define DSL variables which are only used on the host:
	//This way we can give the value of AC_slice_steps from the configuration file!
	if((i % info[AC_slice_steps]) == 0) acGridWriteSlicesToDiskSynchronous("slices",i,simulation_time);
        acGridExecuteTaskGraph(update_graph,1);

	//The info entry of AC_dt has the calculated default value based on the DSL equation
	simulation_time += info[AC_dt];
	++i;
    }
    if((i % info[AC_slice_steps]) != 0) acGridWriteSlicesToDiskSynchronous("slices",i,simulation_time);
    acGridQuit();
    printf("Test simulation done!\n");
    return EXIT_SUCCESS;
}

