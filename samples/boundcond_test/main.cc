#include "astaroth.h"
#include "astaroth_debug.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED
#include <cstring>
#include <iostream>
#include <mpi.h>

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define RESET "\x1B[0m"

#define debug_bc_errors 1
#define debug_bc_values 0





bool
test_simple_bc(AcMesh mesh, int3 direction, int3 dims, int3 domain_start, int3 ghost_start,
            AcMeshInfo info, AcReal offset)
{
    bool passed = true;

    int col_width[4] = {0, 0, 0, 0};

    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        int s_len = strlen(vtxbuf_names[i]);
        if (col_width[i % 4] < s_len) {
            col_width[i % 4] = s_len;
        }
    }

    AcReal epsilon = 0.00000000000001;
    printf("\n(%2d,%2d,%d):\n\t", direction.x, direction.y, direction.z);
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
        bool first_error = true;
        AcReal* field    = mesh.vertex_buffer[i];
        size_t errors    = 0;
        size_t total     = 0;
        for (int x = 0; x < dims.x; x++) {
            for (int y = 0; y < dims.y; y++) {
                for (int z = 0; z < dims.z; z++) {
                    int3 dom   = int3{domain_start.x + x, domain_start.y + y, domain_start.z + z};
                    int3 ghost = int3{(direction.x == 0) ? dom.x : ghost_start.x + dims.x - x - 1,
                                      (direction.y == 0) ? dom.y : ghost_start.y + dims.y - y - 1,
                                      (direction.z == 0) ? dom.z : ghost_start.z + dims.z - z - 1};

                    int idx_dom   = acVertexBufferIdx(dom.x, dom.y, dom.z, info);
                    int idx_ghost = acVertexBufferIdx(ghost.x, ghost.y, ghost.z, info);
                    total++;

                    if ((field[idx_dom] + offset < field[idx_ghost] - epsilon) ||
                        (field[idx_dom] + offset > field[idx_ghost] + epsilon)) {
                        errors++;
#if debug_bc_errors
                        if (first_error) {
                            first_error = false;
                            printf("\n\t%sERROR%s at boundary (%2d,%2d,%2d):", RED, RESET,
                                   direction.x, direction.y, direction.z);
                            printf("domain[(%3d,%3d,%3d)] = %f != ghost[(%3d,%3d,%3d)] = %f\n",
                                   dom.x, dom.y, dom.z, (double)field[idx_dom], ghost.x, ghost.y,
                                   ghost.z, (double)field[idx_ghost]);
                            // printf("\n\tdomain[%d]!= ghost[%d]",idx_dom, idx_ghost);
                            fflush(stdout);
                        }
#endif
                    }
                    else {
#if debug_bc_values
                        printf("domain[(%3d,%3d,%3d)] = %f != ghost[(%3d,%3d,%3d)] = %f\n", dom.x,
                               dom.y, dom.z, field[idx_dom], ghost.x, ghost.y, ghost.z,
                               field[idx_ghost]);
#endif
                    }
                }
            }
        }
        passed &= (errors == 0);
        printf("%*s: %s %lu/%lu. %s", col_width[i % 4], vtxbuf_names[i], (errors > 0 ? RED : GRN),
               (total - errors), total, RESET);
        if ((i + 1) % 4 == 0) {
            printf("\n\t");
        }
    }
    printf("\n");
    return passed;
}

bool
test_mirror(AcMesh mesh, int3 direction, int3 dims, int3 domain_start, int3 ghost_start,
            AcMeshInfo info){
    return test_simple_bc(mesh,direction,dims,domain_start, ghost_start, info, 0);
}



int
main(void)
{
    int ret_val = 0;

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

    VertexBufferHandle all_fields[NUM_VTXBUF_HANDLES];
    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++) {
         all_fields[i] = (VertexBufferHandle)i;
    }

    AcMeshInfo submesh_info = info;
    const int3 nn           = int3{
        (int)(info.int_params[AC_nx]),
        (int)(info.int_params[AC_ny]),
        (int)(info.int_params[AC_nz]),
    };

    AcTaskGraph* symmetric_bc_graph = acGridBuildTaskGraph(
        {
         acHaloExchange(all_fields),                                              //0
         acBoundaryCondition(BOUNDARY_XYZ, AC_BOUNDCOND_PERIODIC, all_fields),    //1
         acCompute(KERNEL_solve, all_fields),                                     //2
         acCompute(KERNEL_solve, all_fields),                                     //3
         acCompute(KERNEL_solve, all_fields),                                     //4
         acHaloExchange(all_fields),                                              //5
         acBoundaryCondition(BOUNDARY_X, AC_BOUNDCOND_SYMMETRIC, all_fields),     //6
         acBoundaryCondition(BOUNDARY_Y, AC_BOUNDCOND_SYMMETRIC, all_fields),     //7
         acBoundaryCondition(BOUNDARY_Z, AC_BOUNDCOND_SYMMETRIC, all_fields)});   //8


    acGridExecuteTaskGraph(symmetric_bc_graph, 1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &mesh);

    bool faces_passed   = true;
    bool edges_passed   = true;
    bool corners_passed = true;

    if (pid == 0) {
        // acGraphPrintDependencies(symmetric_bc_graph);
#if 1

        printf("\nTesting boundconds.\n");
        // Symmetric boundconds
        printf("---Face symmetry---\n");
        // faces
        faces_passed &= test_mirror(mesh, int3{1, 0, 0}, int3{NGHOST, nn.y, nn.z},
                                    int3{nn.x - 1, NGHOST, NGHOST},
                                    int3{nn.x + NGHOST, NGHOST, NGHOST}, submesh_info);
        faces_passed &= test_mirror(mesh, int3{0, 1, 0}, int3{nn.x, NGHOST, nn.z},
                                    int3{NGHOST, nn.y - 1, NGHOST},
                                    int3{NGHOST, nn.y + NGHOST, NGHOST}, submesh_info);
        faces_passed &= test_mirror(mesh, int3{0, 0, 1}, int3{nn.x, nn.y, NGHOST},
                                    int3{NGHOST, NGHOST, nn.z - 1},
                                    int3{NGHOST, NGHOST, nn.z + NGHOST}, submesh_info);

        faces_passed &= test_mirror(mesh, int3{-1, 0, 0}, int3{NGHOST, nn.y, nn.z},
                                    int3{NGHOST + 1, NGHOST, NGHOST}, int3{0, NGHOST, NGHOST},
                                    submesh_info);
        faces_passed &= test_mirror(mesh, int3{0, -1, 0}, int3{nn.x, NGHOST, nn.z},
                                    int3{NGHOST, NGHOST + 1, NGHOST}, int3{NGHOST, 0, NGHOST},
                                    submesh_info);
        faces_passed &= test_mirror(mesh, int3{0, 0, -1}, int3{nn.x, nn.y, NGHOST},
                                    int3{NGHOST, NGHOST, NGHOST + 1}, int3{NGHOST, NGHOST, 0},
                                    submesh_info);

        printf("---Edge symmetry---\n");
        // edges
        edges_passed &= test_mirror(mesh, int3{1, 1, 0}, int3{NGHOST, NGHOST, nn.z},
                                    int3{nn.x - 1, nn.y - 1, NGHOST},
                                    int3{nn.x + NGHOST, nn.y + NGHOST, NGHOST}, submesh_info);
        edges_passed &= test_mirror(mesh, int3{1, 0, 1}, int3{NGHOST, nn.y, NGHOST},
                                    int3{nn.x - 1, NGHOST, nn.z - 1},
                                    int3{nn.x + NGHOST, NGHOST, nn.z + NGHOST}, submesh_info);
        edges_passed &= test_mirror(mesh, int3{0, 1, 1}, int3{nn.x, NGHOST, NGHOST},
                                    int3{NGHOST, nn.y - 1, nn.z - 1},
                                    int3{NGHOST, nn.y + NGHOST, nn.z + NGHOST}, submesh_info);

        edges_passed &= test_mirror(mesh, int3{1, -1, 0}, int3{NGHOST, NGHOST, nn.z},
                                    int3{nn.x - 1, NGHOST + 1, NGHOST},
                                    int3{nn.x + NGHOST, 0, NGHOST}, submesh_info);
        edges_passed &= test_mirror(mesh, int3{1, 0, -1}, int3{NGHOST, nn.y, NGHOST},
                                    int3{nn.x - 1, NGHOST, NGHOST + 1},
                                    int3{nn.x + NGHOST, NGHOST, 0}, submesh_info);
        edges_passed &= test_mirror(mesh, int3{0, 1, -1}, int3{nn.x, NGHOST, NGHOST},
                                    int3{NGHOST, nn.y - 1, NGHOST + 1},
                                    int3{NGHOST, nn.y + NGHOST, 0}, submesh_info);

        edges_passed &= test_mirror(mesh, int3{-1, 1, 0}, int3{NGHOST, NGHOST, nn.z},
                                    int3{NGHOST + 1, nn.y - 1, NGHOST},
                                    int3{0, nn.y + NGHOST, NGHOST}, submesh_info);
        edges_passed &= test_mirror(mesh, int3{-1, 0, 1}, int3{NGHOST, nn.y, NGHOST},
                                    int3{NGHOST + 1, NGHOST, nn.z - 1},
                                    int3{0, NGHOST, nn.z + NGHOST}, submesh_info);
        edges_passed &= test_mirror(mesh, int3{0, -1, 1}, int3{nn.x, NGHOST, NGHOST},
                                    int3{NGHOST, NGHOST + 1, nn.z - 1},
                                    int3{NGHOST, 0, nn.z + NGHOST}, submesh_info);

        edges_passed &= test_mirror(mesh, int3{-1, -1, 0}, int3{NGHOST, NGHOST, nn.z},
                                    int3{NGHOST + 1, NGHOST + 1, NGHOST}, int3{0, 0, NGHOST},
                                    submesh_info);
        edges_passed &= test_mirror(mesh, int3{-1, 0, -1}, int3{NGHOST, nn.y, NGHOST},
                                    int3{NGHOST + 1, NGHOST, NGHOST + 1}, int3{0, NGHOST, 0},
                                    submesh_info);
        edges_passed &= test_mirror(mesh, int3{0, -1, -1}, int3{nn.x, NGHOST, NGHOST},
                                    int3{NGHOST, NGHOST + 1, NGHOST + 1}, int3{NGHOST, 0, 0},
                                    submesh_info);

        printf("---Corner symmetry---\n");
        corners_passed = true;
        // corners
        corners_passed &= test_mirror(mesh, int3{1, 1, 1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{nn.x - 1, nn.y - 1, nn.z - 1},
                                      int3{nn.x + NGHOST, nn.y + NGHOST, nn.z + NGHOST},
                                      submesh_info);

        corners_passed &= test_mirror(mesh, int3{1, 1, -1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{nn.x - 1, nn.y - 1, NGHOST + 1},
                                      int3{nn.x + NGHOST, nn.y + NGHOST, 0}, submesh_info);
        corners_passed &= test_mirror(mesh, int3{1, -1, 1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{nn.x - 1, NGHOST + 1, nn.z - 1},
                                      int3{nn.x + NGHOST, 0, nn.z + NGHOST}, submesh_info);
        corners_passed &= test_mirror(mesh, int3{-1, 1, 1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{NGHOST + 1, nn.y - 1, nn.z - 1},
                                      int3{0, nn.y + NGHOST, nn.z + NGHOST}, submesh_info);

        corners_passed &= test_mirror(mesh, int3{-1, 1, -1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{NGHOST + 1, nn.y - 1, NGHOST + 1},
                                      int3{0, nn.y + NGHOST, 0}, submesh_info);
        corners_passed &= test_mirror(mesh, int3{1, -1, -1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{nn.x - 1, NGHOST + 1, NGHOST + 1},
                                      int3{nn.x + NGHOST, 0, 0}, submesh_info);
        corners_passed &= test_mirror(mesh, int3{-1, -1, 1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{NGHOST + 1, NGHOST + 1, nn.z - 1},
                                      int3{0, 0, nn.z + NGHOST}, submesh_info);

        corners_passed &= test_mirror(mesh, int3{-1, -1, -1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{NGHOST + 1, NGHOST + 1, NGHOST + 1}, int3{0, 0, 0},
                                      submesh_info);
#endif
        // As more boundary conditions are added, add tests for them here
    } // if pid == 0

    acGridDestroyTaskGraph(symmetric_bc_graph);

    if (pid == 0) {
        printf("\nSymmetric boundary condition test:\n");
        if (faces_passed) {
            printf("\t%sFaces PASSED%s\n", GRN, RESET);
        }
        else {
            printf("\t%sFaces FAILED%s\n", RED, RESET);
        }
        if (edges_passed) {
            printf("\t%sEdges PASSED%s\n", GRN, RESET);
        }
        else {
            printf("\t%sEdges FAILED%s\n", RED, RESET);
        }
        if (corners_passed) {
            printf("\t%sCorners PASSED%s\n", GRN, RESET);
        }
        else {
            printf("\t%sCorners FAILED%s\n", RED, RESET);
        }

        if (faces_passed && edges_passed && corners_passed) {
            printf("\n%sPASSED%s\n", GRN, RESET);
            ret_val = 0;
        }
        else {
            printf("\n%sFAILED%s\n", RED, RESET);
            ret_val = 1;
        }
    }

//Dummy test for arbitrary calculations, tests that the dependencies work
    AcTaskGraph* add_one_bc_graph = acGridBuildTaskGraph(
        {
         acHaloExchange(all_fields),                                            //0
         acBoundaryCondition(BOUNDARY_XYZ, AC_BOUNDCOND_PERIODIC, all_fields),  //1
         acCompute(KERNEL_solve, all_fields),                                   //2
         acCompute(KERNEL_solve, all_fields),                                   //3
         acCompute(KERNEL_solve, all_fields),                                   //4
         acHaloExchange(all_fields),                                            //5
         acBoundaryCondition(BOUNDARY_X, AC_BOUNDCOND_ADD_ONE, all_fields),     //6
         acBoundaryCondition(BOUNDARY_Y, AC_BOUNDCOND_ADD_TWO, all_fields),     //7
         acBoundaryCondition(BOUNDARY_Z, AC_BOUNDCOND_ADD_FOUR, all_fields)});   //8

    acGridExecuteTaskGraph(add_one_bc_graph, 1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &mesh);

    faces_passed   = true;
    edges_passed   = true;
    corners_passed = true;
 
 
    if (pid == 0) {
        // acGraphPrintDependencies(symmetric_bc_graph);
#if 1

        printf("\nTesting arbitrary.\n");

        // Symmetric boundconds
        printf("---Face ghost zones---\n");
        // faces
        faces_passed &= test_simple_bc(mesh, int3{1, 0, 0}, int3{NGHOST, nn.y, nn.z},
                                    int3{nn.x - 1, NGHOST, NGHOST},
                                    int3{nn.x + NGHOST, NGHOST, NGHOST}, submesh_info,1.0);
        faces_passed &= test_simple_bc(mesh, int3{0, 1, 0}, int3{nn.x, NGHOST, nn.z},
                                    int3{NGHOST, nn.y - 1, NGHOST},
                                    int3{NGHOST, nn.y + NGHOST, NGHOST}, submesh_info,2.0);
        faces_passed &= test_simple_bc(mesh, int3{0, 0, 1}, int3{nn.x, nn.y, NGHOST},
                                    int3{NGHOST, NGHOST, nn.z - 1},
                                    int3{NGHOST, NGHOST, nn.z + NGHOST}, submesh_info,4.0);

        faces_passed &= test_simple_bc(mesh, int3{-1, 0, 0}, int3{NGHOST, nn.y, nn.z},
                                    int3{NGHOST + 1, NGHOST, NGHOST}, int3{0, NGHOST, NGHOST},
                                    submesh_info,1.0);
        faces_passed &= test_simple_bc(mesh, int3{0, -1, 0}, int3{nn.x, NGHOST, nn.z},
                                    int3{NGHOST, NGHOST + 1, NGHOST}, int3{NGHOST, 0, NGHOST},
                                    submesh_info,2.0);
        faces_passed &= test_simple_bc(mesh, int3{0, 0, -1}, int3{nn.x, nn.y, NGHOST},
                                    int3{NGHOST, NGHOST, NGHOST + 1}, int3{NGHOST, NGHOST, 0},
                                    submesh_info,4.0);

        printf("---Edge ghost zones---\n");
        // edges
        edges_passed &= test_simple_bc(mesh, int3{1, 1, 0}, int3{NGHOST, NGHOST, nn.z},
                                    int3{nn.x - 1, nn.y - 1, NGHOST},
                                    int3{nn.x + NGHOST, nn.y + NGHOST, NGHOST}, submesh_info,3.0);
        edges_passed &= test_simple_bc(mesh, int3{1, 0, 1}, int3{NGHOST, nn.y, NGHOST},
                                    int3{nn.x - 1, NGHOST, nn.z - 1},
                                    int3{nn.x + NGHOST, NGHOST, nn.z + NGHOST}, submesh_info,5.0);
        edges_passed &= test_simple_bc(mesh, int3{0, 1, 1}, int3{nn.x, NGHOST, NGHOST},
                                    int3{NGHOST, nn.y - 1, nn.z - 1},
                                    int3{NGHOST, nn.y + NGHOST, nn.z + NGHOST}, submesh_info,6.0);

        edges_passed &= test_simple_bc(mesh, int3{1, -1, 0}, int3{NGHOST, NGHOST, nn.z},
                                    int3{nn.x - 1, NGHOST + 1, NGHOST},
                                    int3{nn.x + NGHOST, 0, NGHOST}, submesh_info,3.0);
        edges_passed &= test_simple_bc(mesh, int3{1, 0, -1}, int3{NGHOST, nn.y, NGHOST},
                                    int3{nn.x - 1, NGHOST, NGHOST + 1},
                                    int3{nn.x + NGHOST, NGHOST, 0}, submesh_info,5.0);
        edges_passed &= test_simple_bc(mesh, int3{0, 1, -1}, int3{nn.x, NGHOST, NGHOST},
                                    int3{NGHOST, nn.y - 1, NGHOST + 1},
                                    int3{NGHOST, nn.y + NGHOST, 0}, submesh_info,6.0);

        edges_passed &= test_simple_bc(mesh, int3{-1, 1, 0}, int3{NGHOST, NGHOST, nn.z},
                                    int3{NGHOST + 1, nn.y - 1, NGHOST},
                                    int3{0, nn.y + NGHOST, NGHOST}, submesh_info,3.0);
        edges_passed &= test_simple_bc(mesh, int3{-1, 0, 1}, int3{NGHOST, nn.y, NGHOST},
                                    int3{NGHOST + 1, NGHOST, nn.z - 1},
                                    int3{0, NGHOST, nn.z + NGHOST}, submesh_info,5.0);
        edges_passed &= test_simple_bc(mesh, int3{0, -1, 1}, int3{nn.x, NGHOST, NGHOST},
                                    int3{NGHOST, NGHOST + 1, nn.z - 1},
                                    int3{NGHOST, 0, nn.z + NGHOST}, submesh_info,6.0);

        edges_passed &= test_simple_bc(mesh, int3{-1, -1, 0}, int3{NGHOST, NGHOST, nn.z},
                                    int3{NGHOST + 1, NGHOST + 1, NGHOST}, int3{0, 0, NGHOST},
                                    submesh_info, 3.0);
        edges_passed &= test_simple_bc(mesh, int3{-1, 0, -1}, int3{NGHOST, nn.y, NGHOST},
                                    int3{NGHOST + 1, NGHOST, NGHOST + 1}, int3{0, NGHOST, 0},
                                    submesh_info, 5.0);
        edges_passed &= test_simple_bc(mesh, int3{0, -1, -1}, int3{nn.x, NGHOST, NGHOST},
                                    int3{NGHOST, NGHOST + 1, NGHOST + 1}, int3{NGHOST, 0, 0},
                                    submesh_info, 6.0);

        printf("---Corner ghost zones---\n");
        corners_passed = true;
        // corners
        corners_passed &= test_simple_bc(mesh, int3{1, 1, 1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{nn.x - 1, nn.y - 1, nn.z - 1},
                                      int3{nn.x + NGHOST, nn.y + NGHOST, nn.z + NGHOST},
                                      submesh_info, 7.0);

        corners_passed &= test_simple_bc(mesh, int3{1, 1, -1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{nn.x - 1, nn.y - 1, NGHOST + 1},
                                      int3{nn.x + NGHOST, nn.y + NGHOST, 0}, submesh_info, 7.0);
        corners_passed &= test_simple_bc(mesh, int3{1, -1, 1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{nn.x - 1, NGHOST + 1, nn.z - 1},
                                      int3{nn.x + NGHOST, 0, nn.z + NGHOST}, submesh_info, 7.0);
        corners_passed &= test_simple_bc(mesh, int3{-1, 1, 1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{NGHOST + 1, nn.y - 1, nn.z - 1},
                                      int3{0, nn.y + NGHOST, nn.z + NGHOST}, submesh_info, 7.0);

        corners_passed &= test_simple_bc(mesh, int3{-1, 1, -1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{NGHOST + 1, nn.y - 1, NGHOST + 1},
                                      int3{0, nn.y + NGHOST, 0}, submesh_info, 7.0);
        corners_passed &= test_simple_bc(mesh, int3{1, -1, -1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{nn.x - 1, NGHOST + 1, NGHOST + 1},
                                      int3{nn.x + NGHOST, 0, 0}, submesh_info, 7.0);
        corners_passed &= test_simple_bc(mesh, int3{-1, -1, 1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{NGHOST + 1, NGHOST + 1, nn.z - 1},
                                      int3{0, 0, nn.z + NGHOST}, submesh_info, 7.0);

        corners_passed &= test_simple_bc(mesh, int3{-1, -1, -1}, int3{NGHOST, NGHOST, NGHOST},
                                      int3{NGHOST + 1, NGHOST + 1, NGHOST + 1}, int3{0, 0, 0},
                                      submesh_info, 7.0);
#endif
        // As more boundary conditions are added, add tests for them here
    } // if pid == 0

    acGridDestroyTaskGraph(add_one_bc_graph);

    if (pid == 0) {
        printf("\nAdd one boundary condition test:\n");
        if (faces_passed) {
            printf("\t%sFaces PASSED%s\n", GRN, RESET);
        }
        else {
            printf("\t%sFaces FAILED%s\n", RED, RESET);
        }
        if (edges_passed) {
            printf("\t%sEdges PASSED%s\n", GRN, RESET);
        }
        else {
            printf("\t%sEdges FAILED%s\n", RED, RESET);
        }
        if (corners_passed) {
            printf("\t%sCorners PASSED%s\n", GRN, RESET);
        }
        else {
            printf("\t%sCorners FAILED%s\n", RED, RESET);
        }

        if (faces_passed && edges_passed && corners_passed) {
            printf("\n%sPASSED%s\n", GRN, RESET);
            ret_val = (ret_val == 0)?0:1;
        }
        else {
            printf("\n%sFAILED%s\n", RED, RESET);
            ret_val = 1;
        }
    }  

    //End
    acGridQuit();
    MPI_Finalize();

    return ret_val;
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
