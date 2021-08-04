## CMake settings
# V3.9 required for first-class CUDA support
# V3.17 required for the FindCUDAToolkit package
# V3.18 required for CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.18)
find_program(CMAKE_C_COMPILER NAMES $ENV{CC} gcc PATHS ENV PATH NO_DEFAULT_PATH)

if(USE_HIP)
    # NOTE!! WARNING!! HIP-SUPPORT MOSTLY HACKED TOGETHER, TODO CLEANUP	
    set(COMMON_FLAGS "-DOMPI_SKIP_MPICXX -Wall -Wextra -Wshadow") # -DOMPI_SKIP_MPICXX is to force OpenMPI to use the C interface

    add_definitions(-DAC_USE_HIP=1)
    add_definitions(-D__HIP_PLATFORM_HCC__) # or __HIP_PLATFORM_NVCC__ and -fopenmp?

    find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} hipcc PATHS ENV PATH NO_DEFAULT_PATH)
    include_directories("/opt/rocm-4.1.1/include") # Triton (TODO HACK until CMake/HIP support)
    # include_directories("/appl/opt/rocm/rocm-4.0.0/hsa/include/") # For missing hsa/hsa.h
    
    # Conversion from CUDA to HIP
    add_compile_definitions(CUresult=hipError_t)
    add_compile_definitions(cudaError_t=hipError_t)
    add_compile_definitions(cudaSuccess=hipSuccess)
    add_compile_definitions(cudaStream_t=hipStream_t)
    add_compile_definitions(cudaGetErrorString=hipGetErrorString)
    add_compile_definitions(cuFloatComplex=hipFloatComplex)
    add_compile_definitions(make_cuFloatComplex=make_hipFloatComplex)
    add_compile_definitions(cuDoubleComplex=hipDoubleComplex)
    add_compile_definitions(make_cuDoubleComplex=make_hipDoubleComplex)
    add_compile_definitions(cudaPeekAtLastError=hipPeekAtLastError)
    add_compile_definitions(cudaGetLastError=hipGetLastError)
    add_compile_definitions(cudaDeviceSynchronize=hipDeviceSynchronize)

    add_compile_definitions(cudaEvent_t=hipEvent_t)
    add_compile_definitions(cudaEventCreate=hipEventCreate)
    add_compile_definitions(cudaEventRecord=hipEventRecord)
    add_compile_definitions(cudaEventSynchronize=hipEventSynchronize)
    add_compile_definitions(cudaEventElapsedTime=hipEventElapsedTime)
    add_compile_definitions(cudaStreamSynchronize=hipStreamSynchronize)
    add_compile_definitions(cudaStreamCreate=hipStreamCreate)
    add_compile_definitions(cudaStreamCreateWithPriority=hipStreamCreateWithPriority)
    add_compile_definitions(cudaStreamDestroy=hipStreamDestroy)
    add_compile_definitions(cudaMemcpyDeviceToHost=hipMemcpyDeviceToHost)
    add_compile_definitions(cudaMemcpyHostToDevice=hipMemcpyHostToDevice)
    add_compile_definitions(cudaSetDevice=hipSetDevice)

    add_compile_definitions(cudaMemcpy=hipMemcpy)
    add_compile_definitions(cudaMemcpyAsync=hipMemcpyAsync)
    add_compile_definitions(cudaMemcpyToSymbolAsync=hipMemcpyToSymbolAsync)
    
    add_compile_definitions(cudaMalloc=hipMalloc)
    add_compile_definitions(cudaFree=hipFree)


	add_compile_definitions(cudaDeviceProp=hipDeviceProp_t)
	add_compile_definitions(cudaGetDeviceCount=hipGetDeviceCount)
	add_compile_definitions(cudaDeviceGetPCIBusId=hipDeviceGetPCIBusId)
	add_compile_definitions(cudaMemGetInfo=hipMemGetInfo)
	add_compile_definitions(cudaStreamNonBlocking=hipStreamNonBlocking)
	add_compile_definitions(cudaMemcpyPeerAsync=hipMemcpyPeerAsync)
	add_compile_definitions(cudaGetDeviceProperties=hipGetDeviceProperties)
else()	
    find_program(CMAKE_CXX_COMPILER NAMES $ENV{CXX} g++ PATHS ENV PATH NO_DEFAULT_PATH)

    set(COMMON_FLAGS "-DOMPI_SKIP_MPICXX -Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow") # -DOMPI_SKIP_MPICXX is to force OpenMPI to use the C interface

    include(CheckLanguage)
    ## CUDA
    # GPU, compute capability
    # K40, 3.5
    # K80, 3.7
    # P100, 6.0
    # V100, 7.0
    if (NOT CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 70) # Default
    else ()
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}) # User-specified
    endif()
    string (REPLACE " " "," CUDA_COMMON_FLAGS "${COMMON_FLAGS}")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options=${CUDA_COMMON_FLAGS}")

    # todo check if we dont hip here and find cuda
    check_language(CUDA REQUIRED)
    enable_language(CUDA)
endif()

## Project settings
project(astaroth C CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Project-wide compilation flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")

# These do not seem to have any effect
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=gfx908")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-sroa")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-load-store-vectorizer")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-scalarize-global-loads")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-internalize-symbols")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-early-inline-all")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-mode-register")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-dpp-combine")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -amdgpu-aa")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)


## Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release") # Default
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

## Options
option(DOUBLE_PRECISION   "Generates double precision code."                    OFF)
option(BUILD_SAMPLES      "Builds projects in samples subdirectory."            OFF)
option(MPI_ENABLED        "Enables additional functions for MPI communciation." OFF)
option(USE_CUDA_AWARE_MPI "Uses GPUDirect RDMA for direct GPU-GPU communication instead of routing communication through host memory" ON)
option(MULTIGPU_ENABLED   "Enables multi-GPU on a single node. Uses peer-to-peer communication instead of MPI. Affects Legacy & Node layers only." ON)
option(VERBOSE            "Enables various status and warning messages"         OFF)
option(BUILD_UTILS        "Builds the utility library"                          ON)
option(USE_HIP            "Uses HIP instead of CUDA for GPU part"               OFF)


## Compile ACC
add_subdirectory(acc)

## Compile AC headers
if (NOT DEFINED DSL_MODULE_DIR)
    set(DSL_MODULE_DIR ${CMAKE_SOURCE_DIR}/acc/mhd_solver) # Default
endif ()
get_filename_component(DSL_MODULE_DIR ${DSL_MODULE_DIR} REALPATH)
message(STATUS "AC module dir: ${DSL_MODULE_DIR}")

file(GLOB DSL_SOURCES ${DSL_MODULE_DIR}/*
                      ${CMAKE_SOURCE_DIR}/acc/stdlib/*)
set(DSL_HEADERS "${PROJECT_BINARY_DIR}/user_kernels.h"
                "${PROJECT_BINARY_DIR}/user_defines.h"
                "${PROJECT_BINARY_DIR}/astaroth.f90")

add_custom_command (
    COMMENT "Building ACC objects ${DSL_MODULE_DIR}"
    COMMAND ${CMAKE_SOURCE_DIR}/acc/compile_acc_module.sh ${DSL_MODULE_DIR}
    DEPENDS ${DSL_SOURCES} acc
    OUTPUT ${DSL_HEADERS}
)
add_custom_target(dsl_headers ALL DEPENDS ${DSL_HEADERS})

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
include_directories(src/common)                                # Common headers
include_directories(${CMAKE_BINARY_DIR})                       # DSL headers
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # CUDA headers

## Subdirectories
add_subdirectory(src/core/kernels)
add_subdirectory(src/core)
if (BUILD_UTILS)
    message("Note: utils depends on -DDSL_MODULE_DIR=acc/mhd_solver. Otherwise causes undefined variable errors.")
    add_subdirectory(src/utils)
endif()

add_subdirectory(samples/benchmark-device)
if (BUILD_SAMPLES)
    add_subdirectory(samples/standalone)
    add_subdirectory(samples/standalone_mpi)
    add_subdirectory(samples/ctest)
    add_subdirectory(samples/cpptest)
    add_subdirectory(samples/mpitest)
    add_subdirectory(samples/mpi_fullgriderror_test)
    add_subdirectory(samples/benchmark)
    add_subdirectory(samples/benchmark-device)
    #add_subdirectory(samples/genbenchmarkscripts)
    #add_subdirectory(samples/mpi_reduce_bench)
    add_subdirectory(samples/fortrantest)

    #if (MPI_ENABLED)
    #  add_subdirectory(samples/bwtest)
    #endif()
endif()

# Additional program modules
if (PROGRAM_MODULE_DIR)
    add_subdirectory(${PROGRAM_MODULE_DIR})
endif()
