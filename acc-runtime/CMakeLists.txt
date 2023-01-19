## CMake settings
cmake_minimum_required(VERSION 3.19)

## Project options
option(USE_HIP     "Use HIP as the underlying GPGPU library instead of CUDA" ON)
if (USE_HIP)
    find_program(CMAKE_HIP_COMPILER hipcc REQUIRED)
    set(CMAKE_C_COMPILER ${CMAKE_HIP_COMPILER})
    set(CMAKE_CXX_COMPILER ${CMAKE_HIP_COMPILER})
    
    # Workarounds until we have first-class support for hip/rocm in CMake
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm) # Default path
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.0.0/hip /opt/rocm-5.0.0) # Triton
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.0.0/hip/include /opt/rocm-5.0.0) # Triton
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm/bin)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.0.0/roctracer)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.0.0/roctracer/include)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip/include)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/roctracer/include)

    ## Triton
	include_directories(/opt/rocm-5.0.0/roctracer/include)
	link_directories(/opt/rocm-5.0.0/roctracer/lib)

    ## LUMI
    include_directories(/appl/lumi/SW/LUMI-21.12/common/EB/rocm/4.5.2/include/roctracer)
    link_directories(/appl/lumi/SW/LUMI-21.12/common/EB/rocm/4.5.2/lib)
endif()

## Project settings
project(acc-runtime)

## Project-wide compilation flags
set(COMMON_FLAGS "-Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow")

## Include directories
include_directories(.)

## Compile options
if (NOT DEFINED IMPLEMENTATION)
    set(IMPLEMENTATION 1)
endif()
if (NOT DEFINED MAX_THREADS_PER_BLOCK)
    set(MAX_THREADS_PER_BLOCK 0)
endif()
add_compile_options(-DIMPLEMENTATION=${IMPLEMENTATION})
add_compile_options(-DMAX_THREADS_PER_BLOCK=${MAX_THREADS_PER_BLOCK})
if (USE_HIP)
add_compile_options(-DAC_USE_HIP=1)
# Includes for building stencil_accesses.cpp
list(APPEND CMAKE_PREFIX_PATH /opt/rocm-4.3.0/hip /opt/rocm-4.3.0)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.0.0/hip/include /opt/rocm-5.0.0)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.0.0/roctracer)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.0.0/roctracer/include)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip/include)
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/roctracer/include)

string(REPLACE ";" " -I " GPU_API_INCLUDES "${CMAKE_PREFIX_PATH}")
add_compile_definitions(GPU_API_INCLUDES="${GPU_API_INCLUDES}")
else()
add_compile_definitions(GPU_API_INCLUDES="${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

## Subdirectories
add_subdirectory(acc)
add_subdirectory(api)

