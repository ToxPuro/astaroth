#!/bin/bash

# Modules (!!!)
#module load gcc/8.3.0 cuda/10.1.168 cmake openmpi/4.0.3-cuda nccl
module load gcc/8.3.0 cuda/10.1.168 cmake hpcx-mpi/2.5.0-cuda nccl
export UCX_MEMTYPE_CACHE=n #  Workaround for bug in hpcx-mpi/2.5.0

load_default_case() {
  # Pinned or RDMA
  sed -i 's/#define MPI_USE_PINNED ([0-9]*)/#define MPI_USE_PINNED (0)/' src/core/device.cc

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

# $1 test name
# $2 grid size
create_case() {
  DIR="benchmark_$1"
  mkdir -p $DIR
  cd $DIR
  /users/pekkila/cmake/build/bin/cmake .. && make -j
  cd ..
}

# Mesh size
load_default_case
create_case "meshsize_256"
sed -i 's/const size_t num_iters      = .*;/const size_t num_iters      = 100;/' samples/benchmark/main.cc
create_case "meshsize_512"
create_case "meshsize_1024"
create_case "meshsize_1792"


# Decomposition
load_default_case
sed -i 's/MPI_DECOMPOSITION_AXES (.)/MPI_DECOMPOSITION_AXES (1)/' src/core/device.cc
create_case "decomp_1D"
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "decomp_1D_comm"

load_default_case
sed -i 's/MPI_DECOMPOSITION_AXES (.)/MPI_DECOMPOSITION_AXES (2)/' src/core/device.cc
create_case "decomp_2D"
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "decomp_2D_comm"

load_default_case
sed -i 's/MPI_DECOMPOSITION_AXES (.)/MPI_DECOMPOSITION_AXES (3)/' src/core/device.cc
create_case "decomp_3D"
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "decomp_3D_comm"

# Stencil order
load_default_case
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (2)/' acc/stdlib/stdderiv.h
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (2)/' include/astaroth.h
create_case "stencilord_2"

sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (4)/' acc/stdlib/stdderiv.h
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (4)/' include/astaroth.h
create_case "stencilord_4"

sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (6)/' acc/stdlib/stdderiv.h
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (6)/' include/astaroth.h
create_case "stencilord_6"

sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (8)/' acc/stdlib/stdderiv.h
sed -i 's/#define STENCIL_ORDER ([0-9]*)/#define STENCIL_ORDER (8)/' include/astaroth.h
create_case "stencilord_8"

# Timings
load_default_case
create_case "timings_default"

sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (1)/' src/core/device.cc
create_case "timings_corners"

load_default_case
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "timings_control"

load_default_case
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "timings_comm"

load_default_case
sed -i 's/MPI_COMPUTE_ENABLED (.)/MPI_COMPUTE_ENABLED (1)/' src/core/device.cc
sed -i 's/MPI_COMM_ENABLED (.)/MPI_COMM_ENABLED (0)/' src/core/device.cc
sed -i 's/MPI_INCL_CORNERS (.)/MPI_INCL_CORNERS (0)/' src/core/device.cc
create_case "timings_comp"

# Weak scaling
load_default_case
sed -i 's/const TestType test = .*;/const TestType test = TEST_WEAK_SCALING;/' samples/benchmark/main.cc
create_case "weak_128"
create_case "weak_256"
sed -i 's/const size_t num_iters      = .*;/const size_t num_iters      = 100;/' samples/benchmark/main.cc
create_case "weak_448"

load_default_case
