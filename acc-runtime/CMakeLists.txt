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

## Project options
option(USE_HIP     "Use HIP as the underlying GPGPU library instead of CUDA" OFF)

## CUDA and HIP
find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)
if(NOT USE_HIP AND CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    enable_language(CUDA)
    add_compile_definitions(DEVICE_ENABLED)
    add_compile_definitions(CUDA_ENABLED)
    include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # Silence warnings originating from CUDA headers

    set(CMAKE_CUDA_ARCHITECTURES "60;61;70;80")
    message(STATUS "Enabled CUDA architectures (ACC): " ${CMAKE_CUDA_ARCHITECTURES})
elseif(USE_HIP OR HIP_FOUND)
    set(USE_HIP ON)
    message(STATUS "HIP found")
    enable_language(HIP)
    add_compile_definitions(DEVICE_ENABLED)
    add_compile_definitions(HIP_ENABLED)
    include_directories(SYSTEM ${HIP_INCLUDE_DIRS}) # Silence warnings originating from HIP headers

    find_library(ROCTRACER NAMES roctracer64 PATHS ${ROCM_PATH}/lib REQUIRED)
    find_library(HIPRAND NAMES hiprand PATHS ${ROCM_PATH}/lib REQUIRED)

    set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx908")
    message(STATUS "Enabled HIP architectures (ACC): " ${CMAKE_HIP_ARCHITECTURES})
else()
    message(ERROR "Did not find CUDA or HIP")
endif()

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

