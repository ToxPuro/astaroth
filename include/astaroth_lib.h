#ifndef AC_LIB_H
#define AC_LIB_H

#include <dlfcn.h>
#include <stdio.h>

#include "acc_runtime.h"
#include "func_define.h"

#define LOAD_DSYM(FUNC_NAME)                           \
    *(void**)(&FUNC_NAME) = dlsym(handle, #FUNC_NAME); \
    if (FUNC_NAME == NULL)                             { \
                                  fprintf(stderr, "Fatal error: was not able to load function " #FUNC_NAME "\n"); \
        abort(); \
}

AC_BEGIN_C_DECLARATIONS

typedef void* AcLibHandle;


AcLibHandle
acLibLoadLibrary(const AcMeshInfo* info, const char* lib_path, const char* lib_filename,
                 int* counter);


AC_END_C_DECLARATIONS

#endif
