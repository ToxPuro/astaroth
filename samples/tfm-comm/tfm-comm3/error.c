#include "error.h"

#include <stdlib.h>

#include "errchk.h"

static const char* errorcode_descriptions[NUM_ERRORCODES] = {
    [ERRORCODE_SUCCESS]         = "Success",
    [ERRORCODE_GENERIC_FAILURE] = "Generic failure",
    [ERRORCODE_TEST_FAILURE]    = "Module test failed",
};

const char*
get_errorcode_description(const ErrorCode code)
{
    if (code < ERRORCODE_SUCCESS || code >= NUM_ERRORCODES) {
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
test_get_errorcode_description(void)
{
    int errcount = 0;

    for (int i = ERRORCODE_SUCCESS; i < NUM_ERRORCODES; ++i) {
        ERRCHK(errorcode_descriptions[i] != NULL);
        if (errorcode_descriptions[i] == NULL) {
            ++errcount;
        }
    }

    return errcount;
}
