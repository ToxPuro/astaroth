## CMake settings
cmake_minimum_required(VERSION 3.5)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

## Project settings
project(tfm-comm)

## Find packages
find_package(MPI REQUIRED COMPONENTS C)

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
add_library(tfm-utils SHARED mpi_utils.c type_conversion.c math_utils.c print.c partition.c dynarr.c segment.c buf.c)

add_library(tfm-pack SHARED pack.cc pack.c)
target_link_libraries(tfm-pack PRIVATE tfm-utils)

add_library(tfm-comm SHARED comm.c)
target_link_libraries(tfm-comm PRIVATE MPI::MPI_C tfm-utils tfm-pack)

add_executable(test test.c)
target_link_libraries(test PRIVATE tfm-utils tfm-comm tfm-pack)

add_executable(run main.c)
target_link_libraries(run PRIVATE tfm-utils tfm-comm tfm-pack)

# Test that the MPI installation works
add_executable(mpitest mpitest.c)
target_link_libraries(mpitest PRIVATE MPI::MPI_C)

# Test CUDA-aware MPI
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
add_library(buffer SHARED buffer.cu)
set_target_properties(buffer PROPERTIES CUDA_ARCHITECTURES "80")

add_executable(ca-mpi cuda-aware-mpi-test.c)
target_link_libraries(ca-mpi PRIVATE MPI::MPI_C buffer)

# find_package(cuda REQUIRED)
# enable_language(CUDA)
# add_executable(tfm-gpu gpu.cu)
# target_link_libraries(tfm-gpu PRIVATE hip::host hip::device)
# set_source_files_properties(gpu.cu PROPERTIES LANGUAGE HIP)