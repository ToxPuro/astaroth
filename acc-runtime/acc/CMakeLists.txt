## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(acc C)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Compilation flags
add_compile_options(-mavx -Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow)

## Find dependencies
find_package(BISON)
find_package(FLEX)

bison_target(parser ms.y ${PROJECT_BINARY_DIR}/tab.c VERBOSE COMPILE_FLAGS -Wcounterexamples)
flex_target(scanner ms.l ${PROJECT_BINARY_DIR}/yy.c)
add_flex_bison_dependency(scanner parser)

## Include directories
include_directories(.)
include_directories(${PROJECT_BINARY_DIR})

## Executable
add_executable(acc codegen.c ${BISON_parser_OUTPUTS} ${FLEX_scanner_OUTPUTS})
