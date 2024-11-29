## CMake settings
# V3.9 required for first-class CUDA support
# V3.17 required for the FindCUDAToolkit package
# V3.18 required for CMAKE_CUDA_ARCHITECTURES
# V3.21 required for first-class HIP support
cmake_minimum_required(VERSION 3.21)

## Project settings
project(astaroth)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

## Build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

## Options
option(DOUBLE_PRECISION   "Generates double precision code."                    ON)
option(BUILD_STANDALONE   "Build the standalone_mpi executable to run standard simulation." ON)
option(BUILD_SAMPLES      "Builds projects in samples subdirectory."            ON)
option(BUILD_MHD_SAMPLES  "Builds MHD samples. Has no effect if BUILD_SAMPLES=OFF." ON)
option(BUILD_MLLIB_SAMPLES "Builds samples that use cuDNN or MIOpen" OFF)
option(MPI_ENABLED        "Enables additional functions for MPI communciation." OFF)
option(USE_POSIX_IO       "Uses POSIX I/O less error prone, but increased overheads." ON)
option(USE_CUDA_AWARE_MPI "Uses GPUDirect RDMA for direct GPU-GPU communication instead of routing communication through host memory" ON)
option(MULTIGPU_ENABLED   "Enables multi-GPU on a single node. Uses peer-to-peer communication instead of MPI. Affects Legacy & Node layers only." ON)
option(VERBOSE            "Enables various status and warning messages"         OFF)
option(BUILD_UTILS        "Builds the utility library"                          ON)
option(SINGLEPASS_INTEGRATION "Perform integration in a single pass. Improves performance by ~20% but may introduce slightly larger floating-point arithmetic error than the conventional approach" OFF)
option(USE_DISTRIBUTED_IO "Use distributed IO (one file per process) instead of collective" ON)
option(USE_HIP     "Use HIP as the underlying GPGPU library instead of CUDA" OFF)
option(BUILD_SHARED_LIBS "Build Astaroth as a collection of shared libraries instead of statically built modules" OFF)

option(USE_PERFSTUBS "Build with perfstubs" OFF)
option(USE_VENDORED_PERFSTUBS "Use the vendored perfstubs in 3rd_party/perfstubs" ON)
option(DEBUG_SYNC "Whether to sync after all tasks in default ops" OFF)

## Project-wide compilation flags
# -Werror TODO readd
set(COMMON_FLAGS "-DOMPI_SKIP_MPICXX -Wfatal-errors -Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow -Wno-unused-result") #-DOMPI_SKIP_MPICXX is to force OpenMPI to use the C interface
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")

## CUDA and HIP
find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)
if(NOT USE_HIP AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    enable_language(CUDA)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DCUDA_ENABLED)
    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # Silence warnings originating from CUDA headers

    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;80")
    message(STATUS "Enabled CUDA architectures (Astaroth): " ${CMAKE_CUDA_ARCHITECTURES})
elseif(USE_HIP OR HIP_FOUND)
    set(USE_HIP ON)
    message(STATUS "HIP found")
    enable_language(HIP)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DHIP_ENABLED)
    include_directories(SYSTEM ${HIP_INCLUDE_DIRS}) # Silence warnings originating from HIP headers

    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908")
    message(STATUS "Enabled HIP architectures (Astaroth): " ${CMAKE_HIP_ARCHITECTURES})
else()
    message(ERROR "Did not find CUDA or HIP")
endif()

include(FetchContent)

## Global flags
if (USE_HIP)
    add_definitions(-DAC_USE_HIP=1)
endif()
if (DOUBLE_PRECISION)
    add_definitions(-DAC_DOUBLE_PRECISION=1)
endif ()
if (VERBOSE)
    add_definitions(-DAC_VERBOSE=1)
endif ()
if (MPI_ENABLED)
    add_definitions(-DAC_MPI_ENABLED=1)
endif ()
if (USE_POSIX_IO)
    add_definitions(-DUSE_POSIX_IO=1)
endif()
if (USE_CUDA_AWARE_MPI)
    add_definitions(-DUSE_CUDA_AWARE_MPI=1)
endif()
if (MULTIGPU_ENABLED)
  add_definitions(-DAC_MULTIGPU_ENABLED=1)
endif()
if (SINGLEPASS_INTEGRATION)
    add_definitions(-DAC_SINGLEPASS_INTEGRATION)
endif()
if (USE_DISTRIBUTED_IO)
    add_definitions(-DUSE_DISTRIBUTED_IO=1)
endif()
if (USE_PERFSTUBS)
    add_definitions(-DUSE_PERFSTUBS=1)
    include(${PROJECT_SOURCE_DIR}/cmake/configure_perfstubs.cmake)
endif()
if (DEBUG_SYNC)
    add_definitions(-DDEBUG_SYNC)
endif()

add_definitions(-DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")

## Subdirectories

# 3rd party
add_subdirectory(3rdparty/inih)

# Astaroth modules
add_subdirectory(acc-runtime)
add_subdirectory(src/core/kernels)
add_subdirectory(src/core)
# if (NOT BUILD_SHARED_LIBS)
#     add_subdirectory(samples/ac-interpreter)
# endif()

if (BUILD_UTILS)
    add_subdirectory(src/utils)
endif()

# Other modules

## Have this as a separate option when no other "samples" aree needed.
## Useful e.g. when compilation time is long e.g. with shock viscosity.
if (BUILD_STANDALONE)
    add_subdirectory(samples/standalone_mpi)
endif()

if (BUILD_SAMPLES)
    add_subdirectory(samples/ctest)
    add_subdirectory(samples/cpptest)
    add_subdirectory(samples/benchmark)
    add_subdirectory(samples/benchmark-device)
    add_subdirectory(samples/benchmark-node)
    add_subdirectory(samples/mpi-io)
    add_subdirectory(samples/mpi-io-multithreaded)
    add_subdirectory(samples/pc-varfile-import)
    add_subdirectory(samples/bwtest)
    add_subdirectory(samples/microbenchmark)

    
    # add_subdirectory(samples/standalone)
    #add_subdirectory(samples/standalone_mpi)
    #add_subdirectory(samples/ctest)
    #add_subdirectory(samples/cpptest)
    #add_subdirectory(samples/mpitest)
    #add_subdirectory(samples/taskgraph_example)
    #add_subdirectory(samples/taskgraph_print)
    #add_subdirectory(samples/taskgraph_trace)
    #add_subdirectory(samples/taskgraph_test)
    #add_subdirectory(samples/mpi_fullgriderror_test)
    #add_subdirectory(samples/benchmark)
    #add_subdirectory(samples/benchmark-device)
    #add_subdirectory(samples/boundcond_test)
    
    #add_subdirectory(samples/mpi-io)
    #add_subdirectory(samples/genbenchmarkscripts)
    #add_subdirectory(samples/mpi_reduce_bench)
    #add_subdirectory(samples/fortrantest)
    
    #if (MPI_ENABLED)
    #  add_subdirectory(samples/bwtest)
    #endif()
endif()

if (BUILD_MLLIB_SAMPLES)
# Workaround for finding miopen on Triton
link_directories(/opt/rocm/miopen/lib/)
    add_subdirectory(samples/microbenchmark-nn)
endif ()

if (BUILD_MHD_SAMPLES)
    # add_subdirectory(samples/standalone)
    # add_subdirectory(samples/standalone_mpi)
    add_subdirectory(samples/mpitest)
    add_subdirectory(samples/devicetest)
    add_subdirectory(samples/stencil-loader)
endif()

# User-specified program modules
if (PROGRAM_MODULE_DIR)
    get_filename_component(resolved_program_dir "${PROGRAM_MODULE_DIR}" ABSOLUTE
                           BASE_DIR ${CMAKE_SOURCE_DIR})
    add_subdirectory(${resolved_program_dir})
    message("Resolved program module dir: " ${resolved_program_dir})
endif()
