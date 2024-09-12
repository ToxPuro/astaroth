## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(acc C)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Options
option(OPTIMIZE_MEM_ACCESSES "Optimizes memory accesses by computing only the bare minimum number of stencils but can introduce errors in some use cases, f.ex. if a stencil is accessed conditionally based on a value not known at compile time, runtime compilation lessens this" OFF)
option(OPTIMIZE_CONDITIONALS "Optimizes conditionals based on kernel input parameters by making copies of kernels for each combination of input params, works well with OPTIMIZE_MEM_ACCESSES since reduces the number of conditional accesses to kernels" OFF)
option(OPTIMIZE_FIELDS       "Optimizes fields to be auxiliary,(non-)communicated and dead based on writes, reads and stencil accesses in kernels. Pitfalls are conditional accesses and writes, which runtime compilation can lessen" OFF)

set(ACC_COMPILER_PATH $ENV{ACC_COMPILER_PATH})
#TP: if defined the user already has the executable and has exported a path to it
if(DEFINED ACC_COMPILER_PATH)
	add_executable(acc empty.c)
else()
	## Find dependencies
	find_package(BISON)
	find_package(FLEX)
	add_library(tinyexpr STATIC tinyexpr.c)
	target_link_libraries(tinyexpr PUBLIC m)
	
	bison_target(parser ac.y ${PROJECT_BINARY_DIR}/tab.c VERBOSE)# COMPILE_FLAGS -Wcounterexamples) # Counterexamples need relatively new Bison (at least 3.7 works)
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
	target_compile_definitions(acc PRIVATE -DACC_OUTPUT_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
	target_compile_definitions(acc PRIVATE -DACC_COMPILER_PATH="${PROJECT_BINARY_DIR}/acc")
	
	#target_compile_definitions(acc PRIVATE -DACC_DIR="${CMAKE_CURRENT_SOURCE_DIR}" -DACC_MATH_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../stdlib/math" -DACC_UTILS_INTRINSICS="${CMAKE_CURRENT_SOURCE_DIR}/../stdlib/utils/intrinsics.h")
	target_link_libraries(acc PRIVATE tinyexpr)
	
	if (OPTIMIZE_MEM_ACCESSES)
		set(OPTIMIZE_MEM_ACCESSES_VAL 1)
	else()
		set(OPTIMIZE_MEM_ACCESSES_VAL 0)
	endif()
	
	if (OPTIMIZE_CONDITIONALS)
		set(OPTIMIZE_CONDITIONALS_VAL 1)
	else()
		set(OPTIMIZE_CONDITIONALS_VAL 0)
	endif()

	if (OPTIMIZE_FIELDS)
		set(OPTIMIZE_FIELDS_VAL 1)
	else()
		set(OPTIMIZE_FIELDS_VAL 0)
	endif()
	
	
	target_compile_definitions(acc PRIVATE -DOPTIMIZE_MEM_ACCESSES=${OPTIMIZE_MEM_ACCESSES_VAL} -DOPTIMIZE_CONDITIONALS=${OPTIMIZE_CONDITIONALS_VAL} -DOPTIMIZE_FIELDS=${OPTIMIZE_FIELDS_VAL})
endif()
