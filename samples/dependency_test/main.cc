#include "astaroth.h"
#include "astaroth_debug.h"
#include "astaroth_utils.h"
#include "errchk.h"

#if AC_MPI_ENABLED
#include <iostream>
#include <mpi.h>
#include <cstring>

#define RED "\x1B[31m"
#define GRN "\x1B[32m"
#define RESET "\x1B[0m"

bool
test_for_value(AcMesh mesh, const AcMeshInfo info, int3 start, int3 dims, const AcReal expect)
{
    bool passed = true;

    int col_width[4] = {0,0,0,0};

    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++){
        int s_len = strlen(vtxbuf_names[i]);
        if (col_width[i%4] < s_len){
            col_width[i%4] = s_len;
        }
    }

    for (int i = 0; i < NUM_VTXBUF_HANDLES; i++){
        AcReal* field = mesh.vertex_buffer[i];
        size_t errors = 0;
        size_t total = 0;
        for (int x = 0; x < dims.x; x++){
            for (int y = 0; y < dims.y; y++){
                for (int z = 0; z < dims.z; z++){
                    int3 dom = int3{start.x+x,start.y+y,start.z+z};
                    int idx_dom = acVertexBufferIdx(dom.x, dom.y, dom.z, info);
                    total++;
                    if (field[idx_dom] != expect){
                        errors++;
                        #if 1
                        if (x == y && x == z && x == NGHOST){
                            printf("\n\t%s(%2d,%2d,%2d)=%f%s:",RED,dom.x,dom.y,dom.z, field[idx_dom], RESET);
                            fflush(stdout);
                        }
                        #endif
                    }

                    

                }
            }
        }
        passed &= (errors == 0);
        printf("%*s: %s %lu/%lu. %s", col_width[i%4], vtxbuf_names[i], (errors > 0 ? RED:GRN), (total-errors), total, RESET);
        if ((i+1)%4 == 0) {
            printf("\n\t");
        }
    }
    printf("\n");
    return passed;
}

bool
test_graph(AcMesh mesh, const AcMeshInfo info, const AcTaskGraph* graph, const char* name, const AcReal expect)
{
    //acHostMeshRandomize(&mesh);
    acGridExecuteTaskGraph(graph, 1);
    acGridSynchronizeStream(STREAM_ALL);
    acGridStoreMesh(STREAM_DEFAULT, &mesh);
    int pid;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    if (pid == 0) {
        //acGraphPrintDependencies(symmetric_bc_graph);
        const int3 nn = int3{
            (int)(info.int_params[AC_nx]),
            (int)(info.int_params[AC_ny]),
            (int)(info.int_params[AC_nz]),
        };

        printf("\nTesting TaskGraph %s, expecting %f\n", name, expect);
        bool success= test_for_value(mesh, info, int3{NGHOST,NGHOST,NGHOST}, int3{nn.x, nn.y,nn.z}, expect);
        printf("TaskGraph %s test:",name);
        if (success) {
            printf("Success!\n");
        } else {
            printf("Fail!\n");
        }
    } //if pid == 0
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

    AcTaskGraph* one_kernel = acGridBuildTaskGraph({
                                Compute(Kernel_set_one, all_fields)
                                });
    test_graph(mesh, info, one_kernel, "Single kernel", 1.0);
    acGridDestroyTaskGraph(one_kernel);


    AcTaskGraph* two_kernels = acGridBuildTaskGraph({
                                Compute(Kernel_set_zero, all_fields),
                                Compute(Kernel_set_one, all_fields)
                                });
    test_graph(mesh, info, two_kernels, "Two kernels", 1.0);
    acGridDestroyTaskGraph(two_kernels);


    AcTaskGraph* three_kernels = acGridBuildTaskGraph({
                                Compute(Kernel_set_zero, all_fields),
                                Compute(Kernel_set_one, all_fields),
                                Compute(Kernel_set_two, all_fields)
                                });
    test_graph(mesh, info, three_kernels, "Three kernels", 2.0);
    acGridDestroyTaskGraph(three_kernels);

    acGridQuit();
    MPI_Finalize();
    
    if (pid == 0) {}
    return 0;
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
