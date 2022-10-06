## CMake settings
cmake_minimum_required(VERSION 3.19)

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
string(REPLACE ";" " -I " GPU_API_INCLUDES "${CMAKE_PREFIX_PATH}")
add_compile_definitions(GPU_API_INCLUDES="${GPU_API_INCLUDES}")
endif()

## Subdirectories
add_subdirectory(acc)
add_subdirectory(api)

