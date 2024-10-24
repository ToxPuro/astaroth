## CMake settings
cmake_minimum_required(VERSION 3.5)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

## Project settings
project(tfm-comm)

## Project-wide compilation flags
add_compile_options(# C/CXX/CUDA/HIP
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wfatal-errors>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wall>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wextra>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wdouble-promotion>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wfloat-conversion>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wshadow>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wno-unused-result>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wconversion>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wsign-conversion>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wvla>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wcast-align>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wcast-qual>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wfloat-equal>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wformat=2>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wmissing-declarations>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wmissing-include-dirs>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wpointer-arith>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wredundant-decls>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wsequence-point>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wswitch>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wunreachable-code>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wunused-but-set-parameter>
                    $<$<COMPILE_LANGUAGE:C,CXX,CUDA,HIP>:-Wwrite-strings>
                    # C/CXX
                    $<$<COMPILE_LANGUAGE:C,CXX>:-Wpedantic>
                    $<$<COMPILE_LANGUAGE:C,CXX>:-pedantic-errors>
                    $<$<COMPILE_LANGUAGE:C,CXX>:-Werror=implicit-fallthrough>
                    $<$<COMPILE_LANGUAGE:C,CXX>:-Wundef>
                    # C
                    $<$<COMPILE_LANGUAGE:C>:-Wimplicit-function-declaration>
                    $<$<COMPILE_LANGUAGE:C>:-Wimplicit-int>
                    $<$<COMPILE_LANGUAGE:C>:-Wbad-function-cast>
                    $<$<COMPILE_LANGUAGE:C>:-Wmissing-prototypes>
                    $<$<COMPILE_LANGUAGE:C>:-Wnested-externs>
                    # CUDA/HIP
                    $<$<COMPILE_LANGUAGE:CUDA,HIP>:-Wno-redundant-decls>
)

## Packages
add_library(tfm-utils STATIC decomp.cc partition.cc segment.cc ndarray.cc static_array.cc type_conversion.cc errchk_print.c)

find_package(MPI REQUIRED COMPONENTS C)
add_executable(run main.cc packet.cc)
target_link_libraries(run PRIVATE MPI::MPI_C tfm-utils)

add_executable(test test.cc)
target_link_libraries(test PRIVATE tfm-utils)
