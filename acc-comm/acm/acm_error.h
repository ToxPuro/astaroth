#pragma once

typedef enum {
    ACM_ERRORCODE_SUCCESS,
    ACM_ERRORCODE_GENERIC_FAILURE,
    ACM_ERRORCODE_TEST_FAILURE,
    ACM_ERRORCODE_NOT_IMPLEMENTED,
    ACM_ERRORCODE_MPI_FAILURE,
    ACM_ERRORCODE_INPUT_FAILURE,
    ACM_ERRORCODE_UNSUPPORTED_NDIMS,
    ACM_NUM_ERRORCODES,
} ACM_Errorcode;

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the description corresponding to the error code */
const char* ACM_Get_errorcode_description(const ACM_Errorcode code);

/** Returns 0 on success. Otherwise returns the number of undefined errorcodes. */
int ACM_Test_get_errorcode_description(void);

#ifdef __cplusplus
}
#endif
