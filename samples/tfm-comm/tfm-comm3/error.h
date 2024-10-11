#pragma once

typedef enum {
    ERRORCODE_SUCCESS,
    ERRORCODE_GENERIC_FAILURE,
    ERRORCODE_TEST_FAILURE,
    NUM_ERRORCODES,
} ErrorCode;

#ifdef __cplusplus
extern "C" {
#endif

/** Returns the description corresponding to the error code */
const char* get_errorcode_description(const ErrorCode code);

/** Returns 0 on success and the number of errors encountered otherwise */
int test_get_errorcode_description(void);

#ifdef __cplusplus
}
#endif
