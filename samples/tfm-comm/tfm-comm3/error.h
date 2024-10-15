#pragma once

typedef enum {
    ERRORCODE_SUCCESS,
    ERRORCODE_GENERIC_FAILURE,
    ERRORCODE_TEST_FAILURE,
    ERRORCODE_NOT_IMPLEMENTED,
    ERRORCODE_MPI_FAILURE,
    ERRORCODE_INVALID_NDIMS,
    NUM_ERRORCODES,
} ErrorCode;

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the description corresponding to the error code */
const char* get_errorcode_description(const ErrorCode code);

/** Returns 0 on success. Otherwise returns the number of undefined errorcodes. */
int test_get_errorcode_description(void);

#ifdef __cplusplus
}
#endif
