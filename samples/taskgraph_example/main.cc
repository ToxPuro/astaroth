#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED
#include <mpi.h>
#include <iostream>

int
main(void)
{
    MPI_Init(NULL, NULL);
    int nprocs, pid;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);

    AcMeshInfo info;
    acLoadConfig(AC_DEFAULT_CONFIG, &info);

    AcMesh mesh;
    if (pid == 0) {
        acHostMeshCreate(info, &mesh);
        acHostMeshRandomize(&mesh);
    }

    acGridInit(info);

    //Example: This does the same as acGridIntegrate()
    
    std::cout << "Initializing variables"<< std::endl;
    //First we define what variables we're using.
    //This parameter is a c-style array but only works with c++ at the moment
    //(the interface relies on templates for safety and array type deduction).
    VertexBufferHandle all_variables[] = { VTXBUF_LNRHO,
                                           VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                                           VTXBUF_AX, VTXBUF_AY, VTXBUF_AZ,
                                           VTXBUF_ENTROPY };

    std::cout << "Generating graph"<< std::endl;
    //Build a task graph consisting of:
    // - a halo exchange with periodic boundconds for all variables
    // - a calculation of the solve kernel touching all variables
    //
    //This function call generates tasks for each subregions in the domain
    //and figures out the dependencies between the tasks.
    TaskGraph* hc_graph = acGridBuildTaskGraph({
                            HaloExchange(Boundconds_Periodic, all_variables),
                            Compute(Kernel_solve, all_variables)
                          });

    //We can build multiple TaskGraphs, the MPI requests will not collide
    //because MPI tag space has been partitioned into ranges that each HaloExchange step uses.
    TaskGraph* h3_graph = acGridBuildTaskGraph({
                            HaloExchange(Boundconds_Periodic, all_variables),
                            HaloExchange(Boundconds_Periodic, all_variables),
                            HaloExchange(Boundconds_Periodic, all_variables)
                          });
    
    std::cout << "Loading mesh" <<std::endl;
    acGridLoadMesh(STREAM_DEFAULT, mesh);
    std::cout << "Setting time delta"<< std::endl;
    //Set the time delta
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, FLT_EPSILON);
    acGridSynchronizeStream(STREAM_DEFAULT);
    
    std::cout << "Executing taskgraph Halo->Compute for 3 iterations"<< std::endl;
    //Execute the task graph for three iterations.
    acGridExecuteTaskGraph(hc_graph, 3);

    std::cout << "Executing taskgraph Halo->Halo->Halo for 10 iterations"<< std::endl;
    //Execute the task graph for ten iterations.
    acGridExecuteTaskGraph(h3_graph, 10);
    //End example
    
    std::cout << "Destroying grid"<< std::endl;
    acGridDestroyTaskGraph(hc_graph);
    acGridDestroyTaskGraph(h3_graph);
    acGridQuit();
    MPI_Finalize();
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
