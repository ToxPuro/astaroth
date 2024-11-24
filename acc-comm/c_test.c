#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "acm.h"
#include "errchk_print.h"

#define ERRCHK_ACM(errcode)                                                                        \
    do {                                                                                           \
        const ACM_Errorcode _tmp_acm_api_errcode_ = (errcode);                                     \
        if (_tmp_acm_api_errcode_ != ACM_ERRORCODE_SUCCESS) {                                      \
            errchk_print_error(__func__, __FILE__, __LINE__, #errcode,                             \
                               ACM_Get_errorcode_description(errcode));                            \
            errchk_print_stacktrace();                                                             \
            MPI_Abort(MPI_COMM_WORLD, -1);                                                         \
        }                                                                                          \
    } while (0)

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

static void
print_array(const char* label, const size_t count, const uint64_t* arr)
{
    printf("%s: { ", label);
    for (size_t i = 0; i < count; ++i)
        printf("%" PRIu64 " ", arr[i]);
    printf("}\n");
}

#define PRINT_ARRAY_DEBUG(count, arr) (print_array(#arr, (count), (arr)))

int
main(void)
{
    ERRCHK_ACM(ACM_MPI_Init_funneled());

    const uint64_t global_nn[] = {128, 128, 128};
    const size_t ndims         = ARRAY_SIZE(global_nn);

    MPI_Comm cart_comm;
    ERRCHK_ACM(ACM_MPI_Cart_comm_create(MPI_COMM_WORLD, ndims, global_nn, &cart_comm));

    uint64_t local_nn[ndims] = {0};
    ERRCHK_ACM(ACM_Get_local_nn(cart_comm, ndims, global_nn, local_nn));

    uint64_t decomp[ndims] = {0};
    ERRCHK_ACM(ACM_Get_decomposition(cart_comm, ndims, decomp));

    uint64_t global_nn_offset[ndims] = {0};
    ERRCHK_ACM(ACM_Get_global_nn_offset(cart_comm, ndims, global_nn, global_nn_offset));

    PRINT_ARRAY_DEBUG(ndims, global_nn);
    PRINT_ARRAY_DEBUG(ndims, local_nn);
    PRINT_ARRAY_DEBUG(ndims, decomp);
    PRINT_ARRAY_DEBUG(ndims, global_nn_offset);

    ERRCHK_ACM(ACM_MPI_Finalize());
    return EXIT_SUCCESS;
}
