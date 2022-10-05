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

## Subdirectories
add_subdirectory(acc)
add_subdirectory(api)

