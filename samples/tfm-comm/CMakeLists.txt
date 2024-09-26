## CMake settings
cmake_minimum_required(VERSION 3.5)
set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

## Project settings
project(tfm-comm)

## Project-wide compilation flags
add_compile_options(-Wfatal-errors 
                    -Wall
                    -Wextra
                    -Wdouble-promotion
                    -Wfloat-conversion
                    -Wshadow
                    -Wno-unused-result
                    -Wimplicit-function-declaration
                    -Wconversion
                    -Wsign-conversion
                    -Wimplicit-int
                    -Werror=implicit-fallthrough
                    -pedantic
                    -pedantic-errors)

## Packages
find_package(MPI REQUIRED COMPONENTS C)

## Libraries
# add_library(tfm-print SHARED print.c)
# add_library(tfm-type-conversion SHARED type_conversion.c)
# add_library(tfm-math-utils SHARED math_utils.c)

add_library(tfm-utils SHARED print.c math_utils.c type_conversion.c ndarray.c buffer.c halo_segment.c partition.c dynamic_array.c matrix.c array.c)
#target_link_libraries(tfm-utils INTERFACE tfm-print tfm-math-utils)

add_library(tfm-comm SHARED comm.c halo_segment_batch.c mpi_utils.c)
target_link_libraries(tfm-comm PRIVATE MPI::MPI_C tfm-utils)

#add_library(tfm-math-utils SHARED math_utils.c)

## Program
add_executable(run main.c)
target_link_libraries(run PRIVATE tfm-comm tfm-utils)

add_executable(test test.c)
target_link_libraries(test PRIVATE tfm-utils tfm-comm)
