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

add_library(tfm-math-utils SHARED math_utils.c)
target_link_libraries(tfm-math-utils)

add_library(tfm-type-conversion SHARED type_conversion.c)
target_link_libraries(tfm-type-conversion)

add_library(tfm-decomp SHARED decomp.c)
target_link_libraries(tfm-decomp tfm-math-utils tfm-print tfm-type-conversion)

add_library(tfm-pack SHARED pack.cc)

add_library(tfm-array SHARED array.cc)

add_library(tfm-comm SHARED comm.c)
target_link_libraries(tfm-comm PRIVATE tfm-decomp MPI::MPI_C tfm-pack tfm-array)

## Program
add_executable(run main.c)
target_link_libraries(run tfm-comm)

add_executable(test test.c)
target_link_libraries(test PRIVATE tfm-math-utils tfm-print)
