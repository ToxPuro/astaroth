#include <mpi.h>
#include <cstdio>

int
main(int argc, char *argv[])
{
    MPI_Init(nullptr,nullptr);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    printf("World size %d\n", world_size);

    MPI_Finalize();
}


