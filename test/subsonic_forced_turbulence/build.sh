#!/bin/bash

# This is a sample script. Please copy it to the directory you want to run the
# code in and customize occordingly. 

# The following write the commit indentifier corresponding to the simulation
# run  into a file. This is to help keep track what version of the code was
# used to perform the simulation.

git rev-parse HEAD > COMMIT_CODE.log

# Run cmake to construct makefiles
# In the case you compile in astaroth/build/ directory. Otherwise change ".." to
# the correct path to astaroth/CMakeLists.txt

# Default cmake configuration
CMAKE_FLAGS="-DOPTIMIZE_MEM_ACCESSES=ON -DDOUBLE_PRECISION=ON -DMPI_ENABLED=ON -DOPTIMIZE_INPUT_PARAMS=ON -DUSE_CUDA_AWARE_MPI=OFF"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --old_tiara)
            CMAKE_FLAGS="$CMAKE_FLAGS -DBUILD_SAMPLES=OFF -DUSE_HIP=OFF -DUSE_CUDA_AWARE_MPI=OFF -DDSL_MODULE_DIR=../../../acc-runtime/samples/mhd_modular/ -DCMAKE_CXX_COMPILER=/software/opt/gcc/9.1.0/bin/gcc -DCMAKE_C_COMPILER=/software/opt/gcc/9.1.0/bin/gcc"
            ;;
        --fedora44)
            echo "Setting up for Fedora 44"
            CMAKE_FLAGS="$CMAKE_FLAGS -DBUILD_SAMPLES=OFF -DUSE_HIP=OFF -DUSE_CUDA_AWARE_MPI=OFF -DDSL_MODULE_DIR=../../../acc-runtime/samples/mhd_modular/"
            CMAKE_FLAGS="$CMAKE_FLAGS -DCMAKE_CXX_COMPILER=$HOME/gcc15/bin/g++ -DCMAKE_C_COMPILER=$HOME/gcc15/bin/gcc -DCMAKE_CUDA_HOST_COMPILER=$HOME/gcc15/bin/g++"
            CMAKE_FLAGS="$CMAKE_FLAGS -DMPI_C_COMPILER=$HOME/openmpi/v5/bin/mpicc"
            CMAKE_FLAGS="$CMAKE_FLAGS -DMPI_CXX_COMPILER=$HOME/openmpi/v5/bin/mpicxx"
            CMAKE_FLAGS="$CMAKE_FLAGS -DMPI_Fortran_COMPILER=$HOME/openmpi/v5/bin/mpifort"
            CMAKE_FLAGS="$CMAKE_FLAGS -DMPIEXEC=$HOME/openmpi/v5/bin/mpiexec"
            CMAKE_FLAGS="$CMAKE_FLAGS -DMPI_C_INCLUDE_PATH=$HOME/openmpi/v5/include"
            CMAKE_FLAGS="$CMAKE_FLAGS -DMPI_C_LIBRARIES=$HOME/openmpi/v5/lib/libmpi.so"
            CMAKE_FLAGS="$CMAKE_FLAGS -DMPI_CXX_INCLUDE_PATH=$HOME/openmpi/v5/include"
            CMAKE_FLAGS="$CMAKE_FLAGS -DMPI_CXX_LIBRARIES=$HOME/openmpi/v5/lib/libmpi.so"
            CMAKE_FLAGS="$CMAKE_FLAGS -DGPU_ARCH=100"
            ;;
        *)
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
    shift
done

echo "Building CMake..."

# Run cmake with the selected flags
echo $CMAKE_FLAGS
cmake -B build -S $AC_HOME -DRUNTIME_CMAKE_OPTIONS="${CMAKE_FLAGS}" -DRUNTIME_COMPILATION=ON $CMAKE_FLAGS ../../..

echo "Compiling..."

# Standard compilation

cmake --build build -j

