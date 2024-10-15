#include "comm.h"

#include "dynarr.h"
#include "errchk.h"
#include "math_utils.h"
#include "ntuple.h"
#include "partition.h"
#include "segment.h"

#include <stdio.h>

#include <mpi.h>

int
acCommTest(void)
{
    int errcount = 0;
    errcount += test_get_errorcode_description();
    errcount += test_ntuple();

    // TODO: return errcounts from test_math_utils
    test_math_utils();
    test_segment();
    test_dynarr();
    test_partition();
    // test_mpi_utils();
    // test_pack();

    void* ptr = NULL;
    // CHECK_OK(ptr);
    // if (CHECK_OK(ptr, "PTR alloc failed\n"))
    //     printf("Ok!\n");
    // else
    //     printf("Failure!\n");

    return errcount;
}
