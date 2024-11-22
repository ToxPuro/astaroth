## CMake settings
cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

## Build type
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

## Project settings
project(astaroth-tfm)

## Linting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Subdirectories
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../acc-comm acc-comm)

set(BUILD_ACC_RUNTIME_LIBRARY ON)
set(OPTIMIZE_MEM_ACCESSES ON)
set(DSL_MODULE_DIR $ENV{ASTAROTH}/samples/tfm/mhd)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../acc-runtime acc-runtime)

## Module compilation flags
add_compile_options(# C/CXX/CUDA/HIP
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wfatal-errors>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wall>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wextra>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wdouble-promotion>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wfloat-conversion>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wshadow>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wno-unused-result>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wconversion>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wsign-conversion>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wvla>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wcast-align>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wcast-qual>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wfloat-equal>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wformat=2>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wmissing-declarations>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wmissing-include-dirs>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wpointer-arith>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wredundant-decls>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wsequence-point>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wswitch>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wunreachable-code>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wunused-but-set-parameter>
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Wwrite-strings>
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
                    # HIP
                    #$<$<COMPILE_LANGUAGE:HIP>:-Wno-redundant-decls>
                    #$<$<COMPILE_LANGUAGE:HIP>:-Wno-implicit-int-float-conversion>
                    # CUDA
                    #$<$<COMPILE_LANGUAGE:CUDA>:--restrict>
)

# Module
add_executable(tfm tfm.cc)
target_link_libraries(tfm PRIVATE acc-comm acc-runtime)
target_include_directories(tfm SYSTEM PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../acc-runtime ${CMAKE_CURRENT_SOURCE_DIR}/../../acc-comm)
