## CMake settings
# V3.9 required for first-class CUDA support
# V3.17 required for the FindCUDAToolkit package
# V3.18 required for CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18)

if (USE_HIP)
    find_program(CMAKE_HIP_COMPILER hipcc REQUIRED)
    set(CMAKE_C_COMPILER ${CMAKE_HIP_COMPILER})
    set(CMAKE_CXX_COMPILER ${CMAKE_HIP_COMPILER})
endif()

## Project settings
project(astaroth C CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Project-wide compilation flags
# -Werror TODO readd
set(COMMON_FLAGS "-DOMPI_SKIP_MPICXX -Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow -Wno-unused-result") #-DOMPI_SKIP_MPICXX is to force OpenMPI to use the C interface
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

## Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release") # Default
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

## Options
option(DOUBLE_PRECISION   "Generates double precision code."                    OFF)
option(BUILD_SAMPLES      "Builds projects in samples subdirectory."            ON)
option(MPI_ENABLED        "Enables additional functions for MPI communciation." OFF)
option(USE_CUDA_AWARE_MPI "Uses GPUDirect RDMA for direct GPU-GPU communication instead of routing communication through host memory" ON)
option(MULTIGPU_ENABLED   "Enables multi-GPU on a single node. Uses peer-to-peer communication instead of MPI. Affects Legacy & Node layers only." ON)
option(VERBOSE            "Enables various status and warning messages"         OFF)
option(BUILD_UTILS        "Builds the utility library"                          ON)

## CUDA/HIP
if (USE_HIP)
    add_definitions(-DAC_USE_HIP=1)

    set(GPU_TARGETS "gfx908" CACHE STRING "GPU target")
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm) # Default path
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-4.3.0/hip /opt/rocm-4.3.0) # Triton

    find_package(hip)
else ()
    enable_language(CUDA)

    string (REPLACE " " "," CUDA_COMMON_FLAGS "${COMMON_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow --compiler-options=${CUDA_COMMON_FLAGS}")

    # Set device code architecture
    if (NOT CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70) # Default
    else ()
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}) # User-specified
    endif()
endif()

## Global flags
if (DOUBLE_PRECISION)
    add_definitions(-DAC_DOUBLE_PRECISION=1)
endif ()
if (VERBOSE)
    add_definitions(-DAC_VERBOSE=1)
endif ()
if (MPI_ENABLED)
    find_package(MPI REQUIRED COMPONENTS C)
    include_directories(${MPI_C_INCLUDE_DIRS})
    add_definitions(-DAC_MPI_ENABLED=1)
endif ()
if (USE_CUDA_AWARE_MPI)
    add_definitions(-DUSE_CUDA_AWARE_MPI=1)
endif()
if (MULTIGPU_ENABLED)
  add_definitions(-DAC_MULTIGPU_ENABLED=1)
endif()

add_definitions(-DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")

## Includes
include_directories(include)                                   # Library headers
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # CUDA headers

## Subdirectories
add_subdirectory(acc-runtime)
add_subdirectory(src/core/kernels)
add_subdirectory(src/core)

if (BUILD_UTILS)
    message("Note: utils depends on -DDSL_MODULE_DIR=<path-to-mhd-solver>. Otherwise causes undefined variable errors.")
    add_subdirectory(src/utils)
endif()

if (BUILD_SAMPLES)
    # add_subdirectory(samples/standalone)
    add_subdirectory(samples/standalone_mpi)
    # add_subdirectory(samples/blur)
    add_subdirectory(samples/ctest)
    add_subdirectory(samples/cpptest)
    add_subdirectory(samples/mpitest)
    add_subdirectory(samples/taskgraph_example)
    add_subdirectory(samples/taskgraph_test)
    add_subdirectory(samples/mpi_fullgriderror_test)
    add_subdirectory(samples/benchmark)
    add_subdirectory(samples/benchmark-device)
    add_subdirectory(samples/boundcond_test)
    #add_subdirectory(samples/genbenchmarkscripts)
    #add_subdirectory(samples/mpi_reduce_bench)
    #add_subdirectory(samples/fortrantest)

    #if (MPI_ENABLED)
    #  add_subdirectory(samples/bwtest)
    #endif()
endif()

# Additional program modules
if (PROGRAM_MODULE_DIR)
    add_subdirectory(${PROGRAM_MODULE_DIR})
endif()
