## CMake settings
cmake_minimum_required(VERSION 3.21)

## Project settings
project(acc-comm)

## Standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

## Build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

## Options
option(HOST_ONLY_MODE "Enables compilation without CUDA/HIP toolkits" OFF)

## Linting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Project-wide compilation flags
add_compile_options(# C/CXX/CUDA/HIP
                    # $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-Werror>
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
                    # MPI
                    # Workaround: MPI_CXX_SKIP_MPICXX not applied properly for some reason
                    $<$<COMPILE_LANGUAGE:C,CXX,HIP>:-DOMPI_SKIP_MPICXX>
)

## Testing
enable_testing()

## Subdirectories
add_subdirectory(acm/detail)
add_subdirectory(tests)
add_subdirectory(benchmarks)

