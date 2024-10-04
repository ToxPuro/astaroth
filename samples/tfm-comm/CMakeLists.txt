## CMake settings
cmake_minimum_required(VERSION 3.5)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

## Project settings
project(tfm-comm)

## Project-wide compilation flags
add_compile_options(#-Werror
                    -Wfatal-errors 
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
                    -Wpedantic
                    -pedantic-errors
                    -Wvla
                    -Wbad-function-cast
                    -Wbad-function-cast
                    -Wcast-align
                    -Wcast-qual
                    -Wfloat-equal
                    -Wformat=2
                    -Wmissing-declarations
                    -Wmissing-include-dirs
                    -Wmissing-prototypes
                    -Wnested-externs
                    -Wpointer-arith
                    -Wredundant-decls
                    -Wsequence-point
                    -Wswitch
                    -Wundef
                    -Wunreachable-code
                    -Wunused-but-set-parameter
                    -Wwrite-strings
                    )

## Packages
add_library(tfm-utils SHARED mpi_utils.c type_conversion.c math_utils.c print.c partition.c dynarr.c segment.c buf.c)

#add_library(tfm-pack SHARED pack.cc)
#target_link_libraries(tfm-pack PRIVATE tfm-utils)

find_package(MPI REQUIRED COMPONENTS C)
add_library(tfm-comm SHARED comm.c)
target_link_libraries(tfm-comm PRIVATE MPI::MPI_C tfm-utils)

add_executable(test test.c)
target_link_libraries(test PRIVATE tfm-utils tfm-comm)
