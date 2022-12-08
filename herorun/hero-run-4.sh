#!/bin/bash
#SBATCH --account=project_462000120
#SBATCH --gres=gpu:4
#SBATCH --partition=pilot
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --time=00:14:59
#SBATCH --exclusive
#SBATCH --reservation=team-korpi-lagg

module purge

module load CrayEnv
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm
module load buildtools
module load cray-python
        
srun ./ac_run_mpi --config /users/pekkila/astaroth/config/samples/subsonic_forced_nonhelical_turbulence/astaroth-small.conf
