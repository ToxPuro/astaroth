## CMake settings
cmake_minimum_required(VERSION 3.21)

## Project settings
project(acc-runtime)

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
option(USE_HIP "Enables compilation without CUDA/HIP toolkits" OFF)

## Linting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Project-wide compilation flags
set(COMMON_FLAGS "-Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow")

## CUDA and HIP
find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)
if(NOT USE_HIP AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    enable_language(CUDA)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DCUDA_ENABLED)
    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;80")
    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # Silence warnings originating from CUDA headers
elseif(USE_HIP OR HIP_FOUND)
    set(USE_HIP ON)
    message(STATUS "HIP found")
    enable_language(HIP)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DHIP_ENABLED)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908")
    include_directories(SYSTEM ${HIP_INCLUDE_DIRS}) # Silence warnings originating from HIP headers
else()
    message(ERROR "Did not find CUDA or HIP")
endif()

## Include directories
# include_directories(.)

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
add_compile_definitions(GPU_API_INCLUDES="${ROCM_PATH}/include -I ${ROCM_PATH}/roctracer/include ")
else()
add_compile_definitions(GPU_API_INCLUDES="${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
endif()

## Subdirectories
add_subdirectory(acc)
add_subdirectory(api)

