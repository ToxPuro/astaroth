## CMake settings
cmake_minimum_required(VERSION 3.5)
set(CMAKE_C_STANDARD 11)

## Project settings
project(tfm-comm C)

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

add_library(tfm-comm SHARED comm.c)
target_link_libraries(tfm-comm PRIVATE MPI::MPI_C)


## Program
add_executable(run main.c)
target_link_libraries(run tfm-comm tfm-decomp tfm-print)
