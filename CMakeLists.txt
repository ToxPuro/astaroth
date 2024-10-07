## CMake settings
# V3.9 required for first-class CUDA support
# V3.17 required for the FindCUDAToolkit package
# V3.18 required for CMAKE_CUDA_ARCHITECTURES
cmake_minimum_required(VERSION 3.19)

## Options
option(DOUBLE_PRECISION   "Generates double precision code."                    ON)
option(BUILD_STANDALONE   "Build the standalone_mpi executable to run standard simulation." ON)
option(BUILD_SAMPLES      "Builds projects in samples subdirectory."            ON)
option(BUILD_TESTS        "Builds tests in test subdirectory."            OFF)
option(BUILD_MHD_SAMPLES  "Builds MHD samples. Has no effect if BUILD_SAMPLES=OFF." ON)
option(BUILD_MLLIB_SAMPLES "Builds samples that use cuDNN or MIOpen" OFF)
option(MPI_ENABLED        "Enables additional functions for MPI communciation." OFF)
option(USE_POSIX_IO       "Uses POSIX I/O less error prone, but increased overheads." OFF)
option(USE_CUDA_AWARE_MPI "Uses GPUDirect RDMA for direct GPU-GPU communication instead of routing communication through host memory" ON)
option(MULTIGPU_ENABLED   "Enables multi-GPU on a single node. Uses peer-to-peer communication instead of MPI. Affects Legacy & Node layers only." ON)
option(VERBOSE            "Enables various status and warning messages"         OFF)
option(BUILD_UTILS        "Builds the utility library"                          ON)
option(SINGLEPASS_INTEGRATION "Perform integration in a single pass. Improves performance by ~20% but may introduce slightly larger floating-point arithmetic error than the conventional approach" OFF)
option(USE_DISTRIBUTED_IO "Use distributed IO (one file per process) instead of collective" ON)
option(USE_HIP     "Use HIP as the underlying GPGPU library instead of CUDA" OFF)
option(BUILD_SHARED_LIBS "Build Astaroth as a collection of shared libraries instead of statically built modules" OFF)
option(PACKED_DATA_TRANSFERS    "Enables kernel for packed data transfers"          OFF)
option(ADJACENT_VERTEX_BUFFERS "Allocate vertex buffers such that they form a single massive" OFF)
option(LAGRANGIAN_GRID "Use Lagrangian grid instead of Eulerian" OFF)
option(2D  "Compile Astaroth for 2D simulation" OFF)
option(USE_PERFSTUBS "Build with perfstubs" OFF)
option(USE_VENDORED_PERFSTUBS "Use the vendored perfstubs in 3rd_party/perfstubs" ON)
option(DEBUG_SYNC "Whether to sync after all tasks in default ops" OFF)
option(RUNTIME_COMPILATION "Compile version of Astaroth compiled at runtime" OFF)
option(ROW_MAJOR_ORDER"Uses row-major order instead of the default column-major-order" OFF)

if (USE_HIP)
    find_program(CMAKE_HIP_COMPILER hipcc REQUIRED)
    set(CMAKE_C_COMPILER ${CMAKE_HIP_COMPILER})
    set(CMAKE_CXX_COMPILER ${CMAKE_HIP_COMPILER})
endif()



## Project settings
project(astaroth C CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

## Project-wide compilation flags
# -Werror TODO read
set(COMMON_FLAGS "-DOMPI_SKIP_MPICXX -Wfatal-errors -Wall -Werror -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow -Wno-unused-result -Wno-cast-function-type") #-DOMPI_SKIP_MPICXX is to force OpenMPI to use the C interface
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # For IDE highlighting (clangd)
add_definitions(-DAC_OVERRIDES_PATH="${CMAKE_SOURCE_DIR}/acc-runtime/overrides.h")
add_definitions(-DAC_BASE_PATH="${CMAKE_SOURCE_DIR}")
set(AC_BASE_PATH "${CMAKE_SOURCE_DIR}")
## Build type
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release") # Default
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})


if(RUNTIME_COMPILATION)
	add_definitions(-DAC_RUNTIME_COMPILATION=1)
else()
	add_definitions(-DAC_RUNTIME_COMPILATION=0)
endif()

if(LAGRANGIAN_GRID)
	add_definitions(-DAC_LAGRANGIAN_GRID=1)
else()
	add_definitions(-DAC_LAGRANGIAN_GRID=0)
endif()
if(ROW_MAJOR_ORDER)
	add_definitions(-DAC_ROW_MAJOR_ORDER=1)
else()
	add_definitions(-DAC_ROW_MAJOR_ORDER=0)
endif()

## CUDA/HIP
if (USE_HIP)
    add_definitions(-DAC_USE_HIP=1)

    set(GPU_TARGETS "gfx908" CACHE STRING "GPU target")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --offload-arch=gfx908 --offload-arch=gfx90a")

    # Workaround for finding HIP on Triton
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    # Workaround for finding HIP on Lumi
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.2.3/hip /opt/rocm-5.2.3)
    # Workaround for finding HIP on Dardel
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-5.3.3/hip /opt/rocm-5.3.3)
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm)
    # Workaround for finding roctracer on Frontier
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm-default/roctracer /opt/rocm-default)
    include_directories(/opt/rocm/roctracer/include)
    link_directories(/opt/rocm/roctracer/lib)

    # Rocm
    #include_directories(/opt/rocm/include)
    #link_directories(/opt/rocm/lib)

    # Roctracer
    #Deprecated (roctracer inlude will warn about deprecated header, better to include non deprecated header)
    #include_directories(/opt/rocm/roctracer/include)
    ##link_directories(/opt/rocm/roctracer/lib)
    #LUMI & Dardel
    #set(ROC_TRACER_PATH "/opt/rocm-5.2.3/include/roctracer")
    set(ROC_TRACER_PATH "/opt/rocm/include/roctracer")
    include_directories(${ROC_TRACER_PATH})
    link_directories(${ROC_TRACER_PATH})
    list(APPEND CMAKE_PREFIX_PATH ${ROC_TRACER_PATH})

    # Rocrand
    include_directories(/opt/rocm-5.2.3/hiprand/include)
    include_directories(/opt/rocm-5.2.3/rocrand/include)
    link_directories(/opt/rocm-5.2.3/hiprand/lib)
    link_directories(/opt/rocm-5.2.3/rocrand/lib)

    find_package(hip)
else ()
    enable_language(CUDA)

    # Commented out CUDA host compiler flags: seem to be ignored anyways
    #string (REPLACE " " "," CUDA_COMMON_FLAGS "${COMMON_FLAGS}")
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -forward-unknown-to-host-compiler -Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow --compiler-options=${CUDA_COMMON_FLAGS}")

    # To inspect ptx, run
    # cmake .. && make -j && nvcc -DAC_DOUBLE_PRECISION=1 --generate-code=arch=compute_60,code=[compute_60,sm_60] -c ../src/core/kernels/kernels.cu -I acc-runtime/api/ -I ../include/ -I ../acc-runtime/api/ -I ../acc-runtime -ptx -Xptxas=-Werror,--warn-on-double-precision-use
    # less kernels.ptx
    # Additional compilation info: --resource-usage

    # Not supported with older CUDA
    # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror all-warnings")
    if (NOT DOUBLE_PRECISION)
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas=--warn-on-double-precision-use") # -Werror,
    endif()
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=ca") # Cache global loads in L1
    #set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -dlcm=cg") # Cache global loads in L2 (default on >= NVIDIA Fermi)

    # Set device code architecture
    if (NOT CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 60 61 70 80) # Default
    else ()
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES}) # User-specified
    endif()

    # CUB and Thrust
    include_directories(${CUB_PATH})
    include_directories(${THRUST_PATH})
endif()

include(FetchContent)

## Global flags
if (DOUBLE_PRECISION)
    add_definitions(-DAC_DOUBLE_PRECISION=1)
endif ()
if (VERBOSE)
    add_definitions(-DAC_VERBOSE=1)
endif ()
if (MPI_ENABLED)
    if (USE_DARDEL)
      set(MPI_INCLUDE /opt/cray/pe/mpich/8.1.28/ofi/crayclang/17.0/include)
      set(MPI_LIB /opt/cray/pe/mpich/8.1.28/ofi/crayclang/17.0/lib)
      include_directories(${MPI_INCLUDE})
      link_directories(${MPI_LIB})
    else()
      find_package(MPI REQUIRED COMPONENTS C)
      include_directories(${MPI_C_INCLUDE_DIRS})
    endif()
    add_definitions(-DAC_MPI_ENABLED=1)
endif ()
if (USE_POSIX_IO)
    add_definitions(-DUSE_POSIX_IO=1)
endif()
if (USE_CUDA_AWARE_MPI)
    add_definitions(-DUSE_CUDA_AWARE_MPI=1)
endif()
if (MULTIGPU_ENABLED)
  add_definitions(-DAC_MULTIGPU_ENABLED=1)
endif()
if (SINGLEPASS_INTEGRATION)
    add_definitions(-DAC_SINGLEPASS_INTEGRATION)
endif()
if (USE_DISTRIBUTED_IO)
    add_definitions(-DUSE_DISTRIBUTED_IO=1)
endif()
if (USE_PERFSTUBS)
    add_definitions(-DUSE_PERFSTUBS=1)
    include(${PROJECT_SOURCE_DIR}/cmake/configure_perfstubs.cmake)
endif()
if (DEBUG_SYNC)
    add_definitions(-DDEBUG_SYNC)
endif()

if (ADJACENT_VERTEX_BUFFERS)
    add_definitions(-DAC_ADJACENT_VERTEX_BUFFERS=1)
else ()
    add_definitions(-DAC_ADJACENT_VERTEX_BUFFERS=0)
endif ()

if (2D)
     add_definitions(-DTWO_D=1)
else () 
     add_definitions(-DTWO_D=0)
endif ()

add_definitions(-DAC_DEFAULT_CONFIG="${CMAKE_SOURCE_DIR}/config/astaroth.conf")

## Includes
include_directories(include)                                   # Library headers
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # CUDA headers

# Additional program modules
if (PROGRAM_MODULE_DIR)
    get_filename_component(resolved_program_dir "${PROGRAM_MODULE_DIR}" ABSOLUTE
                           BASE_DIR ${CMAKE_SOURCE_DIR})
    add_subdirectory(${resolved_program_dir})
    message("Resolved program module dir: " ${resolved_program_dir})
endif()

## Subdirectories
add_subdirectory(acc-runtime)
add_subdirectory(src/core/kernels)
add_subdirectory(src/core)
##if (NOT BUILD_SHARED_LIBS)
##    if(NOT 2D)
##    	add_subdirectory(samples/ac-interpreter)
##    endif()
##endif()

if(BUILD_TESTS)
	add_subdirectory(test)
	add_definitions(-DINCLUDE_MODEL_SOLVER=0)
else()
	add_definitions(-DINCLUDE_MODEL_SOLVER=1)
endif()

if (BUILD_UTILS)
    add_subdirectory(src/utils)
endif()

if (RUNTIME_COMPILATION)
	add_subdirectory(runtime_compilation)
endif()

## Have this as a separate option when no other "samples" aree needed.
## Useful e.g. when compilation time is long e.g. with shock viscosity. 
if (BUILD_STANDALONE AND NOT BUILD_TESTS)
    add_subdirectory(samples/standalone_mpi)
endif()

if (BUILD_SAMPLES AND NOT BUILD_TESTS)
    add_subdirectory(samples/ctest)
    add_subdirectory(samples/cpptest)
    #TP: we should make a wrapper functions for the CUDA calls
    if  (NOT RUNTIME_COMPILATION)
    add_subdirectory(samples/benchmark)
    add_subdirectory(samples/benchmark-device)
    add_subdirectory(samples/benchmark-node)
    endif()
    add_subdirectory(samples/mpi-io)
    add_subdirectory(samples/mpi-io-multithreaded)
    add_subdirectory(samples/pc-varfile-import)
    add_subdirectory(samples/microbenchmark)

    if (BUILD_MLLIB_SAMPLES)
	# Workaround for finding miopen on Triton
	link_directories(/opt/rocm/miopen/lib/)
        add_subdirectory(samples/microbenchmark-nn)
    endif ()

    if (BUILD_MHD_SAMPLES)
        # add_subdirectory(samples/standalone)
        # add_subdirectory(samples/standalone_mpi)
        add_subdirectory(samples/mpitest)
        add_subdirectory(samples/devicetest)
	if(TWO_D)
	else()
        	add_subdirectory(samples/stencil-loader)
	endif()
    endif()
    #add_subdirectory(samples/stress)

  # add_subdirectory(samples/standalone)
  #add_subdirectory(samples/standalone_mpi)
  #add_subdirectory(samples/ctest)
  #add_subdirectory(samples/cpptest)
  #add_subdirectory(samples/mpitest)
  #add_subdirectory(samples/taskgraph_example)
  #add_subdirectory(samples/taskgraph_print)
  #add_subdirectory(samples/taskgraph_trace)
  #add_subdirectory(samples/taskgraph_test)
  #add_subdirectory(samples/mpi_fullgriderror_test)
  #add_subdirectory(samples/benchmark)
  #add_subdirectory(samples/benchmark-device)
  #add_subdirectory(samples/boundcond_test)

  #add_subdirectory(samples/mpi-io)
  #add_subdirectory(samples/genbenchmarkscripts)
  #add_subdirectory(samples/mpi_reduce_bench)
  #add_subdirectory(samples/fortrantest)

  #if (MPI_ENABLED)
  #  add_subdirectory(samples/bwtest)
  #endif()
endif()
