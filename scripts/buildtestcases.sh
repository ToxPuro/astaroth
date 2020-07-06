#!/bin/bash

SRUN="srun --account=project_2000403 --mem=32000 -t 00:14:59 -p gpu"
#SRUN="srun --account=project_2000403 --mem=32000 -t 00:14:59 -p gpu --exclusive"
# SLURM bug #1: Exclusive flag invalidates -ntasks-per-socket
# SLURM bug #2: ntasks-per-socket does not work w/ sbath, only interactive srun

SRUN_1="$SRUN --gres=gpu:v100:1 --ntasks-per-socket=1 -n 1 -N 1"
SRUN_2="$SRUN --gres=gpu:v100:2 --ntasks-per-socket=1 -n 2 -N 1"
SRUN_4="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 4 -N 1"
SRUN_8="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 8 -N 2"
SRUN_16="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 16 -N 4"
SRUN_32="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 32 -N 8"
SRUN_64="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 64 -N 16"

load_default_case() {
  # Mesh size
  sed -i 's/const int nx = [0-9]*;/const int nx = 256;/' samples/genbenchmarkscripts/main.c

  # Stencil order
  sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (6)/' acc/stdlib/stdderiv.h
  sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (6)/' include/astaroth.h

  # Timings
  sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (1)/' src/core/device.cc
  sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
  sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc

  # Decomposition
  sed -i 's/MPI_DECOMPOSITION_AXES (.)/MPI_DECOMPOSITION_AXES (3)/' src/core/device.cc

  # Strong/Weak
  sed -i 's/const TestType test = .*;/const TestType test = TEST_STRONG_SCALING;/' samples/benchmark/main.cc

  # Num iters
  sed -i 's/const size_t num_iters      = .*;/const size_t num_iters      = 1000;/' samples/benchmark/main.cc
}

srun_all() {
    sbatch benchmark_1.sh
    sbatch benchmark_2.sh
    sbatch benchmark_4.sh
    sbatch benchmark_8.sh
    sbatch benchmark_16.sh
    sbatch benchmark_32.sh
    sbatch benchmark_64.sh
}

# $1 grid size
srun_all2() {
  $SRUN_1 ./benchmark $1 $1 $1 &
  $SRUN_2 ./benchmark $1 $1 $1 &
  $SRUN_4 ./benchmark $1 $1 $1 &
  $SRUN_8 ./benchmark $1 $1 $1 &
  $SRUN_16 ./benchmark $1 $1 $1 &
  $SRUN_32 ./benchmark $1 $1 $1 &
  $SRUN_64 ./benchmark $1 $1 $1 &
}

# $1 test name
# $2 grid size
create_case() {
  DIR="prefix_$1"
  mkdir -p $DIR
  cd $DIR
  /users/pekkila/cmake/build/bin/cmake .. && make -j
  srun_all $2
  cd ..
}

# Mesh size
load_default_case
create_case "meshsize_256" 256

sed -i 's/const size_t num_iters      = .*;/const size_t num_iters      = 100;/' samples/benchmark/main.cc
create_case "meshsize_512" 512
create_case "meshsize_1024" 1024
create_case "meshsize_1792" 1792


# Decomposition
load_default_case
sed -i 's/MPI_DECOMPOSITION_AXES (.)/MPI_DECOMPOSITION_AXES (1)/' src/core/device.cc
create_case "decomp_1D" 256
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "decomp_1D_comm" 256

load_default_case
sed -i 's/MPI_DECOMPOSITION_AXES (.)/MPI_DECOMPOSITION_AXES (2)/' src/core/device.cc
create_case "decomp_2D" 256
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "decomp_2D_comm" 256

load_default_case
sed -i 's/MPI_DECOMPOSITION_AXES (.)/MPI_DECOMPOSITION_AXES (3)/' src/core/device.cc
create_case "decomp_3D" 256
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "decomp_3D_comm" 256

# Stencil order
load_default_case
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (2)/' acc/stdlib/stdderiv.h
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (2)/' include/astaroth.h
create_case "stencilord_2" 256

sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (4)/' acc/stdlib/stdderiv.h
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (4)/' include/astaroth.h
create_case "stencilord_4" 256

sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (6)/' acc/stdlib/stdderiv.h
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (6)/' include/astaroth.h
create_case "stencilord_6" 256

sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (8)/' acc/stdlib/stdderiv.h
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (8)/' include/astaroth.h
create_case "stencilord_8" 256

# Timings
load_default_case
create_case "timings_default" 256

sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (1)/' src/core/device.cc
create_case "timings_corners" 256

load_default_case
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "timings_control" 256

load_default_case
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "timings_comm" 256

load_default_case
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "timings_comp" 256

# Weak scaling
load_default_case
sed -i 's/const TestType test = .*;/const TestType test = TEST_WEAK_SCALING;/' samples/benchmark/main.cc
create_case "weak_128" 128
create_case "weak_256" 256
create_case "weak_448" 448

load_default_case
