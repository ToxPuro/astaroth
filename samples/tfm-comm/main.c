#include <stdio.h>
#include <stdlib.h>

#include "comm.h"
#include "decomp.h"
#include "math_utils.h"
#include "print.h"

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof(arr[0]))

int
main(int argc, char* argv[])
{
    printf("Hello\n");

    const size_t nprocs_per_node        = 8;
    const size_t nnodes                 = 2;
    const size_t global_dims[]          = {128, 128, 128};
    const size_t partitions_per_layer[] = {nprocs_per_node, nnodes};
    const size_t ndims                  = ARRAY_SIZE(global_dims);
    const size_t nlayers                = ARRAY_SIZE(partitions_per_layer);
    AcDecompositionInfo info            = acDecompositionInfoCreate(ndims, global_dims, nlayers,
                                                                    partitions_per_layer);
    acDecompositionInfoPrint(info);

    for (size_t i = 0; i < prod(info.ndims, info.global_decomposition); ++i) {

        int64_t pid[info.ndims];
        acGetPid3D(i, info, info.ndims, pid);

        size_t row_wise_i = to_linear(info.ndims, pid, info.global_decomposition);

        printf("%zu -> %zu", i, row_wise_i);
        acPrintArray_size_t("", info.ndims, pid);
    }

    acDecompositionInfoDestroy(&info);

    return EXIT_SUCCESS;
}