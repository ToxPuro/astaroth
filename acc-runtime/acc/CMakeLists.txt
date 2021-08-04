## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(acc C)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

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