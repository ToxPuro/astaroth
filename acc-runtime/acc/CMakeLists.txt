## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(acc C)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Options
option(OPTIMIZE_MEM_ACCESSES "Optimizes memory accesses by computing only the bare minimum number of stencils but can introduce errors in some use cases, f.ex. if a stencil is accessed conditionally based on a value not known at compile time." OFF)

## Find dependencies
find_package(BISON)
find_package(FLEX)

bison_target(parser ac.y ${PROJECT_BINARY_DIR}/tab.c VERBOSE) # COMPILE_FLAGS -Wcounterexamples) # Counterexamples need relatively new Bison (at least 3.7 works)
flex_target(scanner ac.l ${PROJECT_BINARY_DIR}/yy.c)
add_flex_bison_dependency(scanner parser)

## Include directories
include_directories(.)
include_directories(${PROJECT_BINARY_DIR})

## Executable
add_executable(acc codegen.c ${BISON_parser_OUTPUTS} ${FLEX_scanner_OUTPUTS})
target_compile_definitions(acc PRIVATE -DACC_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
if (OPTIMIZE_MEM_ACCESSES)
    target_compile_definitions(acc PRIVATE -DOPTIMIZE_MEM_ACCESSES=1)
else()
    target_compile_definitions(acc PRIVATE -DOPTIMIZE_MEM_ACCESSES=0)
endif()

## Terrible hack, copied also in acc/CMakeLists and codegen.c when building stencilgen (look for gcc build call)
if(NOT DEFINED IMPLEMENTATION)
    set(IMPLEMENTATION 3)
endif()
if (NOT DEFINED MAX_THREADS_PER_BLOCK)
    set(MAX_THREADS_PER_BLOCK 192)
endif()

## Options
# cmake -DIMPLEMENTATION=<1-NUM_IMPLEMENTATIONS> .. to switch implementation
target_compile_options(acc PUBLIC -DIMPLEMENTATION=${IMPLEMENTATION})
# cmake -DMAX_THREADS_PER_BLOCK=192 .. to set launch bounds in bwtest-benchmark
target_compile_options(acc PUBLIC -DMAX_THREADS_PER_BLOCK=${MAX_THREADS_PER_BLOCK})
# -Xptxas -dlcm=ca