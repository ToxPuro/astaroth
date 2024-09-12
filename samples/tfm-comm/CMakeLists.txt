## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(tfm-comm C)

## Packages
find_package(MPI REQUIRED COMPONENTS C)

## Libraries
add_library(tfm-print SHARED print.c)

add_library(tfm-math-utils SHARED math_utils.c)
target_link_libraries(tfm-math-utils)

add_library(tfm-type-conversion SHARED type_conversion.c)
target_link_libraries(tfm-type-conversion)

add_library(tfm-decomp SHARED decomp.c)
target_link_libraries(tfm-decomp tfm-math-utils tfm-print tfm-type-conversion)

add_library(tfm-comm SHARED comm.c)
target_link_libraries(tfm-comm PRIVATE MPI::MPI_C)


## Program
add_executable(run main.c)
target_link_libraries(run tfm-comm tfm-decomp tfm-print)