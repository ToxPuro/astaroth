#!/bin/bash

SRUN="srun --account=project_2000403 --mem=32000 -t 00:14:59 -p gpu"
#SRUN="srun --account=project_2000403 --mem=32000 -t 00:14:59 -p gpu --exclusive"
# SLURM bug #1: Exclusive flag invalidates -ntasks-per-socket
# SLURM bug #2: ntasks-per-socket does not work w/ sbath, only interactive srun

SRUN_1="$SRUN --gres=gpu:v100:1 --ntasks-per-socket=1 -n 1 -N 1"
SRUN_2="$SRUN --gres=gpu:v100:2 --ntasks-per-socket=1 -n 2 -N 1"
SRUN_4="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 4 -N 1 --cpus-per-task=10"
SRUN_8="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 8 -N 2 --cpus-per-task=10"
SRUN_16="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 16 -N 4 --cpus-per-task=10"
SRUN_32="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 32 -N 8 --cpus-per-task=10"
SRUN_64="$SRUN --gres=gpu:v100:4 --ntasks-per-socket=2 -n 64 -N 16 --cpus-per-task=10"


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

