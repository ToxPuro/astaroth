#include "astaroth.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED
#include <iostream>
#include <mpi.h>

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define RESET "\x1B[0m"

bool
test_mirror(AcReal* field, int3 direction, int3 dims, int3 domain_start, int3 ghost_start, AcMeshInfo info)
{

    for (int x = 0; x < dims.x; x++){
        for (int y = 0; y < dims.y; y++){
            for (int z = 0; z < dims.z; z++){
                int3 dom = int3{domain_start.x+x,domain_start.y+y,domain_start.z+z};
                int3 ghost = int3{
                                (direction.x ==0)? dom.x: ghost_start.x+dims.x-x-1,
                                (direction.y ==0)? dom.y: ghost_start.y+dims.y-y-1,
                                (direction.z ==0)? dom.z: ghost_start.z+dims.z-z-1
                            };
                int idx_dom = acVertexBufferIdx(dom.x, dom.y, dom.z, info);
                int idx_ghost = acVertexBufferIdx(ghost.x, ghost.y, ghost.z, info);

                if (field[idx_dom] != field[idx_ghost]){
                    printf("%sERROR%s:Symmetric boundconds not satisfied.\n",RED,RESET);
                    printf("%d (%3d,%3d,%3d) != (%3d,%3d,%3d) %d\n",idx_dom, dom.x,dom.y,dom.z,ghost.x,ghost.y,ghost.z,idx_ghost);
                    printf("\t %f != %f\n",field[idx_dom], field[idx_ghost]);
                    return false;
                }

            }
        }
    }

    return true;
}

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

    acGridLoadMesh(STREAM_DEFAULT, mesh);

    // Set the time delta
    acGridLoadScalarUniform(STREAM_DEFAULT, AC_dt, FLT_EPSILON);
    acGridSynchronizeStream(STREAM_DEFAULT);

    VertexBufferHandle all_fields[NUM_VTXBUF_HANDLES] = {VTXBUF_LNRHO, VTXBUF_UUX, VTXBUF_UUY, VTXBUF_UUZ,
                                                         VTXBUF_AX,    VTXBUF_AY,  VTXBUF_AZ,  VTXBUF_ENTROPY};

    TaskGraph* symmetric_bc_graph = acGridBuildTaskGraph({HaloExchange(Boundconds_Symmetric, all_fields)});

    acGridExecuteTaskGraph(symmetric_bc_graph, 1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &mesh);

    const int3 nn = (int3){
        info.int_params[AC_nx],
        info.int_params[AC_ny],
        info.int_params[AC_nz],
    };
   
    bool passed = true;
    printf("\nTesting boundconds.\n");
    /*Symmetric boundconds*/
    printf("---Face symmetry---\n");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++){
        bool good = true;
        printf("\t%-14s ",vtxbuf_names[i]);
        //faces
        good &= test_mirror(mesh.vertex_buffer[i], int3{1,0,0}, int3{NGHOST, nn.y,nn.z}, int3{nn.x-1,NGHOST,NGHOST}, int3{nn.x+NGHOST,NGHOST,NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{0,1,0}, int3{nn.x,NGHOST,nn.z}, int3{NGHOST,nn.y-1,NGHOST}, int3{NGHOST,nn.y+NGHOST,NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{0,0,1}, int3{nn.x,nn.y,NGHOST}, int3{NGHOST,NGHOST,nn.z-1}, int3{NGHOST,NGHOST,nn.z+NGHOST}, info);

        good &= test_mirror(mesh.vertex_buffer[i], int3{-1,0,0}, int3{NGHOST,nn.y,nn.z}, int3{NGHOST+1,NGHOST,NGHOST}, int3{0,NGHOST,NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{0,-1,0}, int3{nn.x,NGHOST,nn.z}, int3{NGHOST,NGHOST+1,NGHOST}, int3{NGHOST,0,NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{0,0,-1}, int3{nn.x,nn.y,NGHOST}, int3{NGHOST,NGHOST,NGHOST+1}, int3{NGHOST,NGHOST,0}, info);
        passed &= good;
        if (good) {
            printf("%sOK!%s\n",GRN,RESET);
        }
    }
    printf("---Edge symmetry---\n");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++){
        bool good = true;
        printf("\t%-14s ",vtxbuf_names[i]);
        //edges
        good &= test_mirror(mesh.vertex_buffer[i], int3{1,1,0}, int3{NGHOST,NGHOST,nn.z}, int3{nn.x-1,nn.y-1,NGHOST}, int3{nn.x+NGHOST,nn.y+NGHOST,NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{1,0,1}, int3{NGHOST,nn.y,NGHOST}, int3{nn.x-1,NGHOST,nn.z-1}, int3{nn.x+NGHOST,NGHOST,nn.z+NGHOST}, info); 
        good &= test_mirror(mesh.vertex_buffer[i], int3{0,1,1}, int3{nn.x,NGHOST,NGHOST}, int3{NGHOST,nn.y-1,nn.z-1}, int3{NGHOST,nn.y+NGHOST,nn.z+NGHOST}, info);
        
        good &= test_mirror(mesh.vertex_buffer[i], int3{1,-1,0}, int3{NGHOST,NGHOST,nn.z}, int3{nn.x-1,NGHOST+1,NGHOST}, int3{nn.x+NGHOST,0,NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{1,0,-1}, int3{NGHOST,nn.y,NGHOST}, int3{nn.x-1,NGHOST,NGHOST+1}, int3{nn.x+NGHOST,NGHOST,0}, info); 
        good &= test_mirror(mesh.vertex_buffer[i], int3{0,1,-1}, int3{nn.x,NGHOST,NGHOST}, int3{NGHOST,nn.y-1,NGHOST+1}, int3{NGHOST,nn.y+NGHOST,0}, info); 

        good &= test_mirror(mesh.vertex_buffer[i], int3{-1,1,0}, int3{NGHOST,NGHOST,nn.z}, int3{NGHOST+1,nn.y-1,NGHOST}, int3{0,nn.y+NGHOST,NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{-1,0,1}, int3{NGHOST,nn.y,NGHOST}, int3{NGHOST+1,NGHOST,nn.z-1}, int3{0,NGHOST,nn.z+NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{0,-1,1}, int3{nn.x,NGHOST,NGHOST}, int3{NGHOST,NGHOST+1,nn.z-1}, int3{NGHOST,0,nn.z+NGHOST}, info);

        good &= test_mirror(mesh.vertex_buffer[i], int3{-1,-1,0}, int3{NGHOST,NGHOST,nn.z}, int3{NGHOST+1,NGHOST+1,NGHOST}, int3{0,0,NGHOST}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{-1,0,-1}, int3{NGHOST,nn.y,NGHOST}, int3{NGHOST+1,NGHOST,NGHOST+1}, int3{0,NGHOST,0}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{0,-1,-1}, int3{nn.x,NGHOST,NGHOST}, int3{NGHOST,NGHOST+1,NGHOST+1}, int3{NGHOST,0,0}, info);
        if (good) {
            printf("%sOK!%s\n",GRN,RESET);
        }        
        passed &= good;
    }
    printf("---Corner symmetry---\n");
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++){
        bool good = true;
        printf("\t%-14s ",vtxbuf_names[i]);
        //corners
        good &= test_mirror(mesh.vertex_buffer[i], int3{ 1, 1, 1}, int3{NGHOST,NGHOST,NGHOST}, int3{nn.x-1,nn.y-1,nn.z-1}, int3{nn.x+NGHOST,nn.y+NGHOST,nn.z+NGHOST}, info);

        good &= test_mirror(mesh.vertex_buffer[i], int3{ 1, 1,-1}, int3{NGHOST,NGHOST,NGHOST}, int3{nn.x-1,nn.y-1,NGHOST+1}, int3{nn.x+NGHOST,nn.y+NGHOST,0}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{ 1,-1, 1}, int3{NGHOST,NGHOST,NGHOST}, int3{nn.x-1,NGHOST+1,nn.z-1}, int3{nn.x+NGHOST,0,nn.z+NGHOST}, info); 
        good &= test_mirror(mesh.vertex_buffer[i], int3{-1, 1, 1}, int3{NGHOST,NGHOST,NGHOST}, int3{NGHOST+1,nn.y-1,nn.z-1}, int3{0,nn.y+NGHOST,nn.z+NGHOST}, info);

        good &= test_mirror(mesh.vertex_buffer[i], int3{-1, 1,-1}, int3{NGHOST,NGHOST,NGHOST}, int3{NGHOST+1,nn.y-1,NGHOST+1}, int3{0,nn.y+NGHOST,0}, info);
        good &= test_mirror(mesh.vertex_buffer[i], int3{ 1,-1,-1}, int3{NGHOST,NGHOST,NGHOST}, int3{nn.x-1,NGHOST+1,NGHOST+1}, int3{nn.x+NGHOST,0,0}, info); 
        good &= test_mirror(mesh.vertex_buffer[i], int3{-1,-1, 1}, int3{NGHOST,NGHOST,NGHOST}, int3{NGHOST+1,NGHOST+1,nn.z-1}, int3{0,0,nn.z+NGHOST}, info);

        good &= test_mirror(mesh.vertex_buffer[i], int3{-1,-1,-1}, int3{NGHOST,NGHOST,NGHOST}, int3{NGHOST+1,NGHOST+1,NGHOST+1}, int3{0,0,0}, info);
        if (good) {
            printf("%sOK!%s\n",GRN,RESET);
        }        
        passed &= good;
    }

    /*As more boundary conditions are added, add tests for them here*/

    acGridDestroyTaskGraph(symmetric_bc_graph);
    acGridQuit();
    MPI_Finalize();

    printf("\nSymmetric boundary condition test: ");
    if (passed) {
        printf("%sPASSED%s\n",GRN,RESET);
        return 0;
    }else {
        printf("%sFAILED%s\n",RED,RESET);
        return 1;
    }

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
