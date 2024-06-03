## Options
option(BUILD_ACC_RUNTIME_LIBRARY "Builds the standalone acc runtime library" OFF)

## DSL module dir

# NOTE: Manually defined DSL_MODULE_DIR must be set relative to the project root, not the actual
# build directory!
# NO! ../acc/mhd_solver
# YES! acc/mhd_solver
#
set(DSL_MODULE $ENV{DSL_MODULE})
if (DEFINED DSL_MODULE)
    set(DSL_MODULE_DIR ${PROJECT_SOURCE_DIR}/${DSL_MODULE})
endif ()

set(EXT_SOURCES $ENV{EXT_SOURCES})

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
                "${PROJECT_SOURCE_DIR}/acc/stencil_accesses.cpp"
		"${PROJECT_SOURCE_DIR}/${EXT_SOURCES}")
set(DSL_HEADERS "user_kernels.h"
                "user_defines.h"
                "user_declarations.h"
                "stencil_accesses.h"
                "user_taskgraphs.h"
                "user_loaders.h"
                "stencilgen.h")
if (NOT DEFINED DSL_MODULE_FILE)
  if (SINGLEPASS_INTEGRATION) 
  add_custom_command (
      COMMENT "Building ACC objects (singlepass): ${DSL_MODULE_DIR}"
      COMMAND ${PROJECT_BINARY_DIR}/acc/acc "${DSL_MODULE_DIR}/*single.ac"
      DEPENDS ${DSL_SOURCES} acc
      OUTPUT ${DSL_HEADERS}
  )
  else ()
  add_custom_command (
      COMMENT "Building ACC objects (twopass): ${DSL_MODULE_DIR}"
      COMMAND ${PROJECT_BINARY_DIR}/acc/acc "${DSL_MODULE_DIR}/*.ac"
      #COMMAND ${PROJECT_BINARY_DIR}/acc/acc "${DSL_MODULE_DIR}/*two.ac"
      DEPENDS ${DSL_SOURCES} acc
      OUTPUT ${DSL_HEADERS}
  )
  endif ()
else()
  if (SINGLEPASS_INTEGRATION) 
  add_custom_command (
      COMMENT "Building ACC objects (singlepass): ${DSL_MODULE_DIR}"
      COMMAND ${PROJECT_BINARY_DIR}/acc/acc "${DSL_MODULE_DIR}/*single.ac" "${DSL_MODULE_FILE}"
      DEPENDS ${DSL_SOURCES} acc
      OUTPUT ${DSL_HEADERS}
  )
  else ()
  add_custom_command (
      COMMENT "Building ACC objects (twopass): ${DSL_MODULE_DIR}"
      COMMAND ${PROJECT_BINARY_DIR}/acc/acc "${DSL_MODULE_DIR}/*.ac" "${DSL_MODULE_FILE}"
      #COMMAND ${PROJECT_BINARY_DIR}/acc/acc "${DSL_MODULE_DIR}/*two.ac"
      DEPENDS ${DSL_SOURCES} acc
      OUTPUT ${DSL_HEADERS}
  )
  endif()
endif()
add_library(acc-runtime-headers INTERFACE ${DSL_HEADERS})
target_include_directories(acc-runtime-headers INTERFACE . .. ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(acc-runtime-headers INTERFACE -DIMPLEMENTATION=${IMPLEMENTATION})
target_compile_options(acc-runtime-headers INTERFACE -DMAX_THREADS_PER_BLOCK=${MAX_THREADS_PER_BLOCK})

if (USE_HIP)
    target_link_libraries(acc-runtime-headers INTERFACE roctracer64)
    target_link_libraries(acc-runtime-headers INTERFACE hiprand)
endif()

message("project source dir ${PROJECT_SOURCE_DIR}")

## Build the runtime library
if (BUILD_ACC_RUNTIME_LIBRARY)
    if (USE_HIP)
        add_definitions(-DAC_USE_HIP=1)
	list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)             # Default
        list(APPEND CMAKE_PREFIX_PATH /opt/rocm-4.3.0/hip /opt/rocm-4.3.0) # Triton
        list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.2.3/hip /opt/rocm-5.2.3) # Lumi
	list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.3.3/hip /opt/rocm-5.3.3) # Dardel
	list(APPEND CMAKE_PREFIX_PATH /opt/rocm-default/hip /opt/rocm-default) # Frontier

        enable_language(CXX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")
        set_source_files_properties(acc_runtime.cu PROPERTIES LANGUAGE CXX)
    else()
        enable_language(CUDA)
        string (REPLACE " " "," CUDA_COMMON_FLAGS "${COMMON_FLAGS}")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wno-unused-variable -Wno-unused-parameter -Wno-unused-function -Wno-#warnings -Wno-#pragma-messages -Wno-extra-tokens -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow --compiler-options=${CUDA_COMMON_FLAGS}")
        # Set device code architecture
        if (NOT CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES 70) # Default
        else ()
            set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}) # User-specified
        endif()
    endif()

    include_directories(${PROJECT_BINARY_DIR})

    add_library(acc-runtime STATIC acc_runtime.cu)
    target_link_libraries(acc-runtime acc-runtime-headers)
    target_compile_features(acc-runtime PRIVATE cxx_std_17)

    add_library(acc-runtime-shared SHARED acc_runtime.cu)
    target_link_libraries(acc-runtime-shared acc-runtime-headers)
    target_compile_features(acc-runtime-shared PRIVATE cxx_std_17)
endif()

if (ADJACENT_VERTEX_BUFFERS)
    add_definitions(-DAC_ADJACENT_VERTEX_BUFFERS=1)
else ()
    add_definitions(-DAC_ADJACENT_VERTEX_BUFFERS=0)
endif ()
