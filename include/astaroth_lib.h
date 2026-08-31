#ifndef AC_LIB_H
#define AC_LIB_H

#include <dlfcn.h>

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

static AcLibHandle UNUSED astarothLibHandle = NULL, utilsLibHandle = NULL, kernelsLibHandle = NULL;

static AcLibHandle
acLibLoadLibrary(const AcMeshInfo* info, const char* lib_path, const char* lib_filename,
                 int* counter)
{
#ifdef __APPLE__
    const char* lib_extension = "dylib";
#else
    const char* lib_extension = "so";
#endif

    char* original_lib_path = NULL;
    if (asprintf(&original_lib_path, "%s/runtime_build/%s/%s.%s",
                 info->runtime_compilation_build_path ? info->runtime_compilation_build_path
                                                      : AC_BINARY_PATH,
                 lib_path, lib_filename, lib_extension) == -1) {
        fprintf(stderr, "Fatal error while preparing path to library %s\n", lib_filename);
        exit(EXIT_FAILURE);
    }

    const char* new_lib_path = acLibraryVersion(original_lib_path, *counter, info->comm);
    ++counter;

    void* lib_handle = dlopen(new_lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib_handle) {
        fprintf(stderr, "Fatal error while loading library %s: %s\n", new_lib_path, dlerror());
        exit(EXIT_FAILURE);
    }

    return lib_handle;
}

AC_END_C_DECLARATIONS

#endif
