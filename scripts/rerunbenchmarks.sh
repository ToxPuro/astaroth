#!/bin/bash

srun_all() {
    sbatch benchmark_1.sh
    sbatch benchmark_2.sh
    sbatch benchmark_4.sh
    sbatch benchmark_8.sh
    sbatch benchmark_16.sh
    sbatch benchmark_32.sh
    sbatch benchmark_64.sh
}

# $1 test name
# $2 grid size
create_case() {
  DIR="prefix_$1"
  cd $DIR
  srun_all $2
  cd ..
}

create_case "meshsize_256" 256
create_case "meshsize_512" 512
create_case "meshsize_1024" 1024
create_case "meshsize_1792" 1792
create_case "decomp_1D" 256
create_case "decomp_1D_comm" 256
create_case "decomp_2D" 256
create_case "decomp_2D_comm" 256
create_case "decomp_3D" 256
create_case "decomp_3D_comm" 256
create_case "stencilord_2" 256
create_case "stencilord_4" 256
create_case "stencilord_6" 256
create_case "stencilord_8" 256
create_case "timings_default" 256
create_case "timings_corners" 256
create_case "timings_control" 256
create_case "timings_comm" 256
create_case "timings_comp" 256
create_case "weak_128" 128
create_case "weak_256" 256
create_case "weak_448" 448

