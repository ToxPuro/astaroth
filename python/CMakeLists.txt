## Project settings
project(pyastaroth
        LANGUAGES CXX)

## Dependencies
litgen_find_nanobind()

## Get a list of headers generated in by ACC Runtime.
get_property(ACC_DSL_HEADERS
             TARGET acc-runtime-headers
             PROPERTY SOURCES)
list(TRANSFORM ACC_DSL_HEADERS PREPEND "${CMAKE_BINARY_DIR}/acc-runtime/api/")
#foreach(header IN LISTS ACC_DSL_HEADERS)
#endforeach()

## Pre-compile headers to reduce the need for special-handling of macros.
set(litgen_generator "${CMAKE_CURRENT_SOURCE_DIR}/litgen_generate_bindings.py")
set(litgen_generator_depfile "${CMAKE_CURRENT_BINARY_DIR}/litgen_build_done.stamp")
set(pyastaroth_pydef "pydef/pyastaroth-impl.cc")
set(pyastaroth_stubs_pyi "stubs/pyastaroth/__init__.pyi")
set(pyastaroth_include_directories
     "${CMAKE_SOURCE_DIR}/include"
     "${CMAKE_BINARY_DIR}/acc-runtime/api"
     "${CMAKE_SOURCE_DIR}/acc-runtime/api"
)
# set(pyastaroth_header_files
#      # Fundamentals used almost everywhere (alphabetical unless order matters)
#      "${CMAKE_SOURCE_DIR}/include/func_define.h"
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/func_attributes.h"
#
#      ${ACC_DSL_HEADERS}
#
#      # No includes or only system-wide includes (alphabetical)
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/acreal.h"
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/common_kernels.h"
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/device_details.h"
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/device_headers.h"
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/hip.h"
# #    "${CMAKE_SOURCE_DIR}/acc-runtime/api/math_utils_base.h"
# #    "${CMAKE_SOURCE_DIR}/include/ac_mpi.h"
# #    "${CMAKE_SOURCE_DIR}/include/astaroth_analysis_helpers.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_debug.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_lib.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_logging.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_runtime_compilation.h"
#      "${CMAKE_SOURCE_DIR}/include/grid_detail.h"
# #    "${CMAKE_SOURCE_DIR}/include/timer_hires.h"
#
#      # Low-level APIs, dependent on ACC Runtime (alphabetical unless order matters)
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/datatypes.h"
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/astaroth_analysis.h"
#
#      # Low-level APIs (alphabetical unless order matters)
# #    "${CMAKE_SOURCE_DIR}/acc-runtime/api/static_analysis.h"
# #    "${CMAKE_SOURCE_DIR}/include/ac_fft.h"
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/host_datatypes.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_cuda_wrappers.h"
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/errchk.h"
#      "${CMAKE_SOURCE_DIR}/include/ac_helpers.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_legacy.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_random.h"
#
#      # Mid-level APIs, also dependent on ACC Runtime and previous (alphabetical unless order matters)
#      #     "${CMAKE_SOURCE_DIR}/acc-runtime/api/math_utils.h"
#
#      # High-level APIs dependent on ACC Runtime (alphabetical unless order matters)
#      "${CMAKE_SOURCE_DIR}/acc-runtime/api/acc_runtime.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_device.h"
#      #     "${CMAKE_SOURCE_DIR}/include/astaroth_helpers.h" (unused and should be deleted)
#
#      # High-level APIs (order matters)
#      "${CMAKE_SOURCE_DIR}/include/astaroth_base.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_grid.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_node.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth.h"
#      "${CMAKE_SOURCE_DIR}/include/astaroth_utils.h"
#      "${CMAKE_SOURCE_DIR}/include/device_detail.h"
# #    "${CMAKE_SOURCE_DIR}/include/astaroth_forcing.h"
# )

#add_library(pyastaroth_pch INTERFACE)
#target_precompile_headers(pyastaroth_pch INTERFACE ${pyastaroth_pch_header})
#target_include_directories(pyastaroth_pch INTERFACE
#                           ${AC_BASE_PATH}/include)
#target_include_directories(pyastaroth_pch INTERFACE
#                           ${AC_BASE_PATH}/acc-runtime/api)
#target_link_libraries(pyastaroth_pch INTERFACE
#                      astaroth_core astaroth_utils)

## Generate sources (i.e., headers) to generate bindings for. This is necessary
## because not all symbols are always available outside of build-time.
add_custom_target(pyastaroth_bindings
                  COMMAND ${litgen_generator} ${pyastaroth_pydef} ${pyastaroth_stubs_pyi} ${CMAKE_SOURCE_DIR} ${pyastaroth_include_directories}
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                  COMMENT "Generating C++ polyfil for Python bindings using litgen")
add_dependencies(pyastaroth_bindings acc-runtime-headers)

nanobind_add_module(pyastaroth_impl ${pyastaroth_pydef})

litgen_setup_module(astaroth_core
                    pyastaroth_impl
                    pyastaroth
                    ${CMAKE_CURRENT_SOURCE_DIR}/stubs)

add_dependencies(pyastaroth_impl pyastaroth_bindings)
target_include_directories(pyastaroth_impl PUBLIC ${AC_BASE_PATH}/include ${AC_BASE_PATH}/acc-runtime/api)
target_link_libraries(pyastaroth_impl PUBLIC acc-runtime-headers astaroth_utils)
