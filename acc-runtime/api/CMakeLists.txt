## ACC runtime library

if (NOT CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 60 70) # Default
else ()
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}) # User-specified
endif()

if (NOT DEFINED DSL_MODULE_DIR)
    set(DSL_MODULE_DIR ${PROJECT_SOURCE_DIR}/samples/mhd) # Default
else()
    get_filename_component(resolved_module_dir "${DSL_MODULE_DIR}" REALPATH
                           BASE_DIR ${CMAKE_BINARY_DIR})
    set(DSL_MODULE_DIR ${resolved_module_dir})
endif ()

set(DSL_SOURCES "${DSL_MODULE_DIR}/*.ac")
set(DSL_HEADERS "user_kernels.h"
                "user_defines.h"
                "user_declarations.h")
add_custom_command (
    COMMENT "Building ACC objects: ${DSL_MODULE_DIR}"
    COMMAND ${PROJECT_BINARY_DIR}/acc/acc ${DSL_SOURCES}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    DEPENDS ${DSL_SOURCES} acc
    OUTPUT ${DSL_HEADERS}
)
add_custom_target(acc-runtime-headers ALL DEPENDS ${DSL_HEADERS})

option(BUILD_ACC_RUNTIME_LIBRARY "Builds the standalone acc runtime library" OFF)
if (BUILD_ACC_RUNTIME_LIBRARY)
    add_library(acc-runtime acc_runtime.cu)
    include_directories(${PROJECT_BINARY_DIR})

    add_dependencies(acc-runtime acc)
    add_dependencies(acc-runtime acc-runtime-headers)

    target_compile_features(acc-runtime PRIVATE cxx_std_17)
endif()
