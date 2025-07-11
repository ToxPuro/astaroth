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
set(CMAKE_HIP_STANDARD 17)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

## Build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

## Options
option(DOUBLE_PRECISION   "Generates double precision code."                    ON)
option(INCLUDE_3RD_PARTIES "Whether to include 3rd party dependencies"          OFF)
option(BUILD_STANDALONE   "Build the standalone_mpi executable to run standard simulation." ON)
option(BUILD_SAMPLES      "Builds projects in samples subdirectory."            ON)
option(BUILD_ADVECTION_EXAMPLE "Builds hello world example"            OFF)
option(BUILD_TESTS        "Builds tests in test subdirectory."            OFF)
option(BUILD_MHD_SAMPLES  "Builds MHD samples. Has no effect if BUILD_SAMPLES=OFF." ON)
option(BUILD_MLLIB_SAMPLES "Builds samples that use cuDNN or MIOpen" OFF)
option(MPI_ENABLED        "Enables additional functions for MPI communciation." OFF)
option(BUILD_ACM          "Whether to build to the Astaroth Communication Module (ACM)." ON)
option(USE_POSIX_IO       "Uses POSIX I/O less error prone, but increased overheads." ON)
option(MULTIGPU_ENABLED   "Enables multi-GPU on a single node. Uses peer-to-peer communication instead of MPI. Affects Legacy & Node layers only." ON)
option(VERBOSE            "Enables various status and warning messages"         OFF)
option(BUILD_UTILS        "Builds the utility library"                          ON)
option(SINGLEPASS_INTEGRATION "Perform integration in a single pass. Improves performance by ~20% but may introduce slightly larger floating-point arithmetic error than the conventional approach" OFF)
option(USE_DISTRIBUTED_IO "Use distributed IO (one file per process) instead of collective" ON)
option(USE_HIP     "Use HIP as the underlying GPGPU library instead of CUDA" OFF)
option(BUILD_SHARED_LIBS "Build Astaroth as a collection of shared libraries instead of statically built modules" OFF)
option(PACKED_DATA_TRANSFERS    "Enables kernel for packed data transfers"          OFF)
option(LAGRANGIAN_GRID "Use Lagrangian grid instead of Eulerian" OFF)
option(2D  "Compile Astaroth for 2D simulation" OFF)
option(USE_PERFSTUBS "Build with perfstubs" OFF)
option(USE_VENDORED_PERFSTUBS "Use the vendored perfstubs in 3rd_party/perfstubs" ON)
option(DEBUG_SYNC "Whether to sync after all tasks in default ops" OFF)
option(RUNTIME_COMPILATION "Compile version of Astaroth compiled at runtime" OFF)
option(SUPPRESS_COMPILER_WARNINGS "Suppress default compiler warnings of Astaroth" OFF)
option(PROFILING_ENABLED "Whether to enable profiling and link to the vendor profiling libraries" ON)
option(FFT_ENABLED "Whether to enable FFT" OFF)

## Project-wide compilation flags
# -Werror TODO read
if(NOT SUPPRESS_COMPILER_WARNINGS)
set(COMMON_FLAGS "-DOMPI_SKIP_MPICXX -Wfatal-errors -Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow -Wno-unused-result -Wno-cast-function-type") #-DOMPI_SKIP_MPICXX is to force OpenMPI to use the C interface
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")
endif()

set(AC_BASE_PATH "${CMAKE_SOURCE_DIR}")
add_definitions(-DAC_BASE_PATH="${AC_BASE_PATH}")

set(AC_BINARY_PATH "${CMAKE_BINARY_DIR}")
add_definitions(-DAC_BINARY_PATH="${AC_BINARY_PATH}")

if (RUNTIME_COMPILATION)
	add_definitions(-DAC_RUNTIME_COMPILATION=1)
else()
	add_definitions(-DAC_RUNTIME_COMPILATION=0)
endif()

if (LAGRANGIAN_GRID)
	add_definitions(-DAC_LAGRANGIAN_GRID=1)
else()
	add_definitions(-DAC_LAGRANGIAN_GRID=0)
endif()
if(PROFILING_ENABLED)
	add_definitions(-DPROFILING_ENABLED=1)
else()
	add_definitions(-DPROFILING_ENABLED=0)
endif()

if(NOT SUPPRESS_COMPILER_WARNINGS)
add_compile_options(# C/CXX/CUDA/HIP
                    # $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Werror>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wfatal-errors>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wall>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wextra>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wdouble-promotion>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wfloat-conversion>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wno-parentheses-equality>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wshadow>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wno-unused-result>
                    # Workaround: MPI_CXX_SKIP_MPICXX not applied properly for some reason
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-DOMPI_SKIP_MPICXX>
)
endif()

## CUDA and HIP
find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)
if (NOT USE_HIP AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    enable_language(CUDA)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DCUDA_ENABLED)
    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # Silence warnings originating from CUDA headers

    ##TP: do not set GPU architectures to speedup debug compilations
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    else()
    	set(CMAKE_CUDA_ARCHITECTURES "60;61;70;80")
    endif()
    message(STATUS "Enabled CUDA architectures (Astaroth): " ${CMAKE_CUDA_ARCHITECTURES})
elseif (USE_HIP OR HIP_FOUND)
    set(USE_HIP ON)
    message(STATUS "HIP found")
    enable_language(HIP)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DHIP_ENABLED)
    include_directories(SYSTEM ${HIP_INCLUDE_DIRS}) # Silence warnings originating from HIP headers

    ##TP: do not set GPU architectures to speedup debug compilations
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    else()
    	set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908")
    endif()
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
else()
    add_definitions(-DAC_DOUBLE_PRECISION=0)
endif()
if (VERBOSE)
    add_definitions(-DAC_VERBOSE=1)
endif ()
if (MPI_ENABLED)
    add_definitions(-DAC_MPI_ENABLED=1)
else()
    add_definitions(-DAC_MPI_ENABLED=0)
endif ()
if (FFT_ENABLED)
	add_definitions(-DAC_FFT_ENABLED=1)
else()
	add_definitions(-DAC_FFT_ENABLED=0)
endif()
if (USE_POSIX_IO)
    add_definitions(-DUSE_POSIX_IO=1)
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

if (2D)
     add_definitions(-DTWO_D=1)
else () 
     add_definitions(-DTWO_D=0)
endif ()

add_definitions(-DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")

## Subdirectories

# 3rd party
if (INCLUDE_3RD_PARTIES)
	add_subdirectory(3rdparty/inih)
endif()

# Astaroth modules
add_subdirectory(src/core/helpers)
add_subdirectory(src/core/cuda_wrappers)
add_subdirectory(acc-runtime)
add_subdirectory(src/core/kernels)
add_subdirectory(src/core)

if (BUILD_TESTS)
	add_subdirectory(test)
	add_definitions(-DINCLUDE_MODEL_SOLVER=0)
else()
	add_definitions(-DINCLUDE_MODEL_SOLVER=1)
endif()

if (BUILD_UTILS)
    add_subdirectory(src/utils)
endif()

if (RUNTIME_COMPILATION)
	add_subdirectory(runtime_compilation)
endif()

## Have this as a separate option when no other "samples" aree needed.
## Useful e.g. when compilation time is long e.g. with shock viscosity. 
if(FFT_ENABLED AND NOT USE_HIP)
	add_subdirectory(test/fft-shift-test)
endif()
if (BUILD_STANDALONE AND NOT BUILD_TESTS)
    add_subdirectory(samples/standalone_mpi)
endif()

if(BUILD_ADVECTION_EXAMPLE)
    add_subdirectory(samples/advection-example)
endif()
if (BUILD_SAMPLES AND NOT BUILD_TESTS)
    add_subdirectory(samples/ctest)
    add_subdirectory(samples/cpptest)
    #TP: we should make a wrapper functions for the CUDA calls
    add_subdirectory(samples/benchmark-device)
    if  (NOT RUNTIME_COMPILATION)
    add_subdirectory(samples/benchmark)
    add_subdirectory(samples/benchmark-node)
    endif()
    add_subdirectory(samples/mpi-io)
    add_subdirectory(samples/mpi-io-multithreaded)
    add_subdirectory(samples/pc-varfile-import)
    add_subdirectory(samples/microbenchmark)
    #add_subdirectory(samples/stress)

    
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

if (BUILD_SAMPLES AND BUILD_MLLIB_SAMPLES)
# Workaround for finding miopen on Triton
link_directories(/opt/rocm/miopen/lib/)
    add_subdirectory(samples/microbenchmark-nn)
endif ()

if (BUILD_SAMPLES AND BUILD_MHD_SAMPLES)
    # add_subdirectory(samples/standalone)
    # add_subdirectory(samples/standalone_mpi)
    add_subdirectory(samples/mpitest)
    add_subdirectory(samples/devicetest)
    add_subdirectory(samples/stencil-loader)
endif()

if (MPI_ENABLED AND BUILD_ACM)
    add_subdirectory(acc-comm)
endif()

# User-specified program modules
if (PROGRAM_MODULE_DIR)
    get_filename_component(resolved_program_dir "${PROGRAM_MODULE_DIR}" ABSOLUTE
                           BASE_DIR ${CMAKE_SOURCE_DIR})
    add_subdirectory(${resolved_program_dir})
    message("Resolved program module dir: " ${resolved_program_dir})
endif()

if(EXTRA_SOURCES_DIR)
	add_subdirectory(${EXTRA_SOURCES_DIR} ${EXTRA_SOURCES_DIR}/cmake)
endif()
