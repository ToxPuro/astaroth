## CMake settings
cmake_minimum_required(VERSION 3.10)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

## Project settings
project(tfm-comm)

## Linting
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

## Project-wide compilation flags
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
                    $<$<COMPILE_LANGUAGE:HIP>:-Wno-redundant-decls>
                    # CUDA
                    $<$<COMPILE_LANGUAGE:CUDA,HIP>:--restrict>
)

## Check whether device code is available on the system
add_library(device-libraries INTERFACE)
find_package(CUDAToolkit QUIET)
find_package(HIP QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA found")
    enable_language(CUDA)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DCUDA_ENABLED)
    set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "60")
    target_link_libraries(device-libraries INTERFACE CUDA::cudart)

    add_library(device-module STATIC pack.cu pack.cc)
elseif(HIP_FOUND)
    message(STATUS "HIP found")
    enable_language(HIP)
    add_compile_options(-DDEVICE_ENABLED)
    add_compile_options(-DHIP_ENABLED)
    set_property(GLOBAL PROPERTY HIP_ARCHITECTURES "gfx90a")
    target_link_libraries(device-libraries INTERFACE hip::host hip::device)

    add_library(device-module STATIC pack.cu pack.cc)
else()
    message(WARNING "Did not find CUDA or HIP, compiling in host-only mode")
    add_library(device-module STATIC pack.cc)
endif()

## Packages
add_library(tfm-utils STATIC buffer_exchange.cc decomp.cc partition.cc segment.cc buffer.cc ndarray.cc math_utils.cc static_array.cc type_conversion.cc errchk_print.c)
target_link_libraries(tfm-utils PRIVATE device-module device-libraries)

find_package(MPI REQUIRED COMPONENTS CXX)
add_executable(run main.cc packet.cc mpi_utils.cc)
target_link_libraries(run PRIVATE MPI::MPI_CXX tfm-utils)

## Automated tests
include(CTest)
add_executable(unit_test unit_test.cc)
target_link_libraries(unit_test PRIVATE tfm-utils)

add_test(NAME UnitTest COMMAND unit_test)


## Other
# find_package(CUDAToolkit QUIET)
# find_package(HIP QUIET)
# if (CUDAToolkit_FOUND)
#     message(STATUS "CUDA found")
#     enable_language(CUDA)
#     add_executable(gpu gpu.cu errchk_print.c)
#     # set_target_properties(gpu PROPERTIES CUDA_ARCHITECTURES "60")
# elseif (HIP_FOUND)
#     message(STATUS "HIP found")
#     enable_language(HIP)
#     add_executable(gpu gpu.cu errchk_print.c)
#     target_link_libraries(gpu PRIVATE hip::host hip::device)
#     set_source_files_properties(gpu.cu PROPERTIES LANGUAGE HIP)
#     set_target_properties(gpu PROPERTIES HIP_ARCHITECTURES "gfx90a")
# else ()
#     message(WARNING "Did not find CUDA or HIP")
# endif ()

