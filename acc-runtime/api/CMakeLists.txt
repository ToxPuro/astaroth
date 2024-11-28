## Options
option(BUILD_ACC_RUNTIME_LIBRARY "Builds the standalone acc runtime library" OFF)

## DSL module dir
if (NOT DEFINED DSL_MODULE_DIR)
    set(DSL_MODULE_DIR ${PROJECT_SOURCE_DIR}/samples/mhd_modular) # Default
else()
    get_filename_component(resolved_module_dir "${DSL_MODULE_DIR}" REALPATH
                           BASE_DIR ${CMAKE_BINARY_DIR})
    set(DSL_MODULE_DIR ${resolved_module_dir})
endif()

## Generate acc-runtime-headers
set(DSL_SOURCES "${DSL_MODULE_DIR}/*"
                "${PROJECT_SOURCE_DIR}/acc/stencilgen.c"
                "${PROJECT_SOURCE_DIR}/acc/stencil_accesses.cpp")
set(DSL_HEADERS "user_kernels.h"
                "user_defines.h"
                "user_declarations.h"
                "stencil_accesses.h"
                "stencilgen.h")
add_custom_command (
    COMMENT "Building ACC objects: ${DSL_MODULE_DIR}"
    COMMAND ${PROJECT_BINARY_DIR}/acc/acc "${DSL_MODULE_DIR}/*.ac"
    DEPENDS ${DSL_SOURCES} acc
    OUTPUT ${DSL_HEADERS}
)
add_library(acc-runtime-headers INTERFACE ${DSL_HEADERS})
target_include_directories(acc-runtime-headers INTERFACE . .. ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(acc-runtime-headers INTERFACE -DIMPLEMENTATION=${IMPLEMENTATION})
target_compile_options(acc-runtime-headers INTERFACE -DMAX_THREADS_PER_BLOCK=${MAX_THREADS_PER_BLOCK})

if (USE_HIP)
    target_link_libraries(acc-runtime-headers INTERFACE ${ROCTRACER})
    target_link_libraries(acc-runtime-headers INTERFACE ${HIPRAND})
endif()

message("project source dir ${PROJECT_SOURCE_DIR}")

## Build the runtime library
if (BUILD_ACC_RUNTIME_LIBRARY)

    if (USE_HIP)
        set_source_files_properties(acc_runtime.cu PROPERTIES LANGUAGE HIP)
    endif()
    
    add_library(acc-runtime STATIC acc_runtime.cu)
    target_include_directories(acc-runtime PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_link_libraries(acc-runtime acc-runtime-headers)

endif()
