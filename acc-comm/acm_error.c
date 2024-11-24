#include "acm_error.h"

#include <stdio.h>
#include <stdlib.h>

static const char* const errorcode_descriptions[ACM_NUM_ERRORCODES] = {
    [ACM_ERRORCODE_SUCCESS]           = "Success",
    [ACM_ERRORCODE_GENERIC_FAILURE]   = "Generic failure",
    [ACM_ERRORCODE_TEST_FAILURE]      = "Module test failed",
    [ACM_ERRORCODE_NOT_IMPLEMENTED]   = "Not implemented",
    [ACM_ERRORCODE_MPI_FAILURE]       = "MPI failure",
    [ACM_ERRORCODE_INPUT_FAILURE]     = "Invalid input passed to the library",
    [ACM_ERRORCODE_UNSUPPORTED_NDIMS] = "Supplied ndims not supported with function",
};

const char*
ACM_Get_errorcode_description(const ACM_Errorcode code)
{
    if (code < ACM_ERRORCODE_SUCCESS || code >= ACM_NUM_ERRORCODES) {
        return "Invalid error code";
    }
    else {
        if (errorcode_descriptions[code] == NULL)
            return "Errorcode description not defined";
        else
            return errorcode_descriptions[code];
    }
}

int
ACM_Test_get_errorcode_description(void)
{
    int errcount = 0;

    for (int i = ACM_ERRORCODE_SUCCESS; i < ACM_NUM_ERRORCODES; ++i) {
        if (!errorcode_descriptions[i]) {
            fprintf(stderr, "errorcode_descriptions[%d] was not defined", i);
            ++errcount;
        }
    }

    return errcount;
}
