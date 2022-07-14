## Options
option(BUILD_ACC_RUNTIME_LIBRARY "Builds the standalone acc runtime library" OFF)
option(USE_HIP     "Use HIP as the underlying GPGPU library instead of CUDA" OFF)

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
target_include_directories(acc-runtime-headers INTERFACE . ${CMAKE_CURRENT_BINARY_DIR})

if (USE_HIP)
    target_link_libraries(acc-runtime-headers PUBLIC roctracer64)
endif()

message("project source dir ${PROJECT_SOURCE_DIR}")

## Build the runtime library
if (BUILD_ACC_RUNTIME_LIBRARY)
    if (USE_HIP)
        add_definitions(-DAC_USE_HIP=1)
        list(APPEND CMAKE_PREFIX_PATH /opt/rocm-4.3.0/hip /opt/rocm-4.3.0)

        enable_language(CXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")
        set_source_files_properties(acc_runtime.cu PROPERTIES LANGUAGE CXX)
    else()
        enable_language(CUDA)
        string (REPLACE " " "," CUDA_COMMON_FLAGS "${COMMON_FLAGS}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow --compiler-options=${CUDA_COMMON_FLAGS}")

        # Set device code architecture
        if (NOT CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 70) # Default
        else ()
            set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}) # User-specified
        endif()
    endif()

    add_library(acc-runtime acc_runtime.cu)
    include_directories(${PROJECT_BINARY_DIR})
    target_link_libraries(acc-runtime acc-runtime-headers)

    target_compile_features(acc-runtime PRIVATE cxx_std_17)
endif()
