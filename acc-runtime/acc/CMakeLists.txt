## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(acc C)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Options
option(OPTIMIZE_MEM_ACCESSES "Optimizes memory accesses by computing only the bare minimum number of stencils but can introduce errors in some use cases, f.ex. if a stencil is accessed conditionally based on a value not known at compile time." OFF)
option(OPTIMIZE_CONDITIONALS "Optimizes conditionals based on kernel input parameters by making copies of kernels for each combination of input params, works well with OPTIMIZE_MEM_ACCESSES since reduces the number of conditional accesses to kernels" OFF)

add_compile_options(-pg)
## Find dependencies
find_package(BISON)
find_package(FLEX)
add_library(tinyexpr STATIC tinyexpr.c)
target_link_libraries(tinyexpr PUBLIC m)

bison_target(parser ac.y ${PROJECT_BINARY_DIR}/tab.c VERBOSE COMPILE_FLAGS -Wcounterexamples) # Counterexamples need relatively new Bison (at least 3.7 works)
flex_target(scanner ac.l ${PROJECT_BINARY_DIR}/yy.c)
add_flex_bison_dependency(scanner parser)

## Include directories
include_directories(.)
include_directories(${PROJECT_BINARY_DIR})

## Executable
add_executable(acc codegen.c ${BISON_parser_OUTPUTS} ${FLEX_scanner_OUTPUTS})

target_compile_definitions(acc PRIVATE -DACC_BUILTIN_PATH="${AC_BASE_PATH}/acc-runtime/built-in")
target_compile_definitions(acc PRIVATE -DACC_BUILTIN_FUNCS="${AC_BASE_PATH}/acc-runtime/built-in/functions.h")
target_compile_definitions(acc PRIVATE -DACC_BUILTIN_TYPEDEFS="${AC_BASE_PATH}/acc-runtime/built-in/typedefs.h")
target_compile_definitions(acc PRIVATE -DACC_BUILTIN_KERNELS="${AC_BASE_PATH}/acc-runtime/built-in/kernels.h")
target_compile_definitions(acc PRIVATE -DACC_BUILTIN_DEFAULT_VALUES="${AC_BASE_PATH}/acc-runtime/built-in/default_values.h")
target_compile_definitions(acc PRIVATE -DACC_BUILTIN_VARIABLES="${AC_BASE_PATH}/acc-runtime/built-in/variables.h")
target_compile_definitions(acc PRIVATE -DACC_OVERRIDES_PATH="${AC_BASE_PATH}/acc-runtime/overrides.h")
target_compile_definitions(acc PRIVATE -DACC_GEN_PATH="${AC_BASE_PATH}/acc-runtime/dynamic")
target_compile_definitions(acc PRIVATE -DACC_MATH_DIR="${AC_BASE_PATH}/acc-runtime/stdlib/math")
target_compile_definitions(acc PRIVATE -DACC_UTILS_INTRINSICS="${AC_BASE_PATH}/acc-runtime/stdlib/utils/intrinsics.h")
target_compile_definitions(acc PRIVATE -DACC_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

#target_compile_definitions(acc PRIVATE -DACC_DIR="${CMAKE_CURRENT_SOURCE_DIR}" -DACC_MATH_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../stdlib/math" -DACC_UTILS_INTRINSICS="${CMAKE_CURRENT_SOURCE_DIR}/../stdlib/utils/intrinsics.h")
target_link_libraries(acc PRIVATE tinyexpr)

if (OPTIMIZE_MEM_ACCESSES)
	if(OPTIMIZE_CONDITIONALS)
		target_compile_definitions(acc PRIVATE -DOPTIMIZE_MEM_ACCESSES=1 -DOPTIMIZE_CONDITIONALS=1)
	else()
		target_compile_definitions(acc PRIVATE -DOPTIMIZE_MEM_ACCESSES=1 -DOPTIMIZE_CONDITIONALS=0)
	endif()
else()
	if(OPTIMIZE_CONDITIONALS)
		target_compile_definitions(acc PRIVATE -DOPTIMIZE_MEM_ACCESSES=0 -DOPTIMIZE_CONDITIONALS=1)
	else()
		target_compile_definitions(acc PRIVATE -DOPTIMIZE_MEM_ACCESSES=0 -DOPTIMIZE_CONDITIONALS=0)
	endif()
endif()
#if (OPTIMIZE_CONDITIONALS)
#    target_compile_definitions(acc PRIVATE -DOPTIMIZE_CONDITIONALS=1)
#else()
#    target_compile_definitions(acc PRIVATE -DOPTIMIZE_CONDITIIONALS=0)
#endif()
