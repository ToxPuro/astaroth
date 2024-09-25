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
add_library(tfm-print SHARED print.c)
add_library(tfm-type-conversion SHARED type_conversion.c)
add_library(tfm-math-utils SHARED math_utils.c)

add_library(tfm-partition SHARED partition.c)
target_link_libraries(tfm-partition PRIVATE tfm-math-utils tfm-print tfm-dynamic-array tfm-ndarray)

add_library(tfm-ndarray SHARED ndarray.c)
target_link_libraries(tfm-ndarray PRIVATE tfm-math-utils tfm-print)

add_library(tfm-dynamic-array SHARED dynamic_array.c)
target_link_libraries(tfm-dynamic-array PRIVATE tfm-ndarray tfm-math-utils)

add_library(tfm-decomp SHARED decomp.c)
target_link_libraries(tfm-decomp PRIVATE tfm-math-utils tfm-print tfm-type-conversion)

add_library(tfm-comm SHARED comm.c comm_cart.c)
target_link_libraries(tfm-comm PRIVATE MPI::MPI_C tfm-math-utils tfm-type-conversion tfm-print tfm-decomp tfm-dynamic-array tfm-ndarray tfm-halo-segment-batch m)

### Communication abstractions
add_library(tfm-buffer SHARED buffer.c)
target_link_libraries(tfm-buffer PRIVATE tfm-math-utils tfm-print)

add_library(tfm-halo-segment SHARED halo_segment.c)
target_link_libraries(tfm-halo-segment PRIVATE tfm-buffer tfm-math-utils tfm-print)

add_library(tfm-halo-segment-batch SHARED halo_segment_batch.c)
target_link_libraries(tfm-halo-segment-batch PRIVATE tfm-halo-segment tfm-math-utils tfm-print tfm-dynamic-array tfm-ndarray)

## Program
add_executable(run main.c)
target_link_libraries(run tfm-comm tfm-math-utils tfm-ndarray tfm-partition)


add_library(tfm-array SHARED array.c)
