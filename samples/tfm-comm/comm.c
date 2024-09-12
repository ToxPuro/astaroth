#include "comm.h"

#define SUCCESS (0)
#define FAILURE (-1)

#include <mpi.h>
#include <stdio.h>

int
test(void)
{
    printf("Hello from comm_init\n");
    return SUCCESS;
}