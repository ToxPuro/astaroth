This is the hello world setup of Astaroth: simulating the advection of the concentration an admixture (C) by advection of prescribed velocity v.

# Building and Running 
* cd astaroth
* pip install -r requirements.txt
* . ./sourceme.sh
* cd samples/advection-example
* ./build.sh
* Run with `sbatch disbatch.sh` (You have to modify the sbatch file to fit your cluster) 
* In the end there should be a movie of a moving sine wave in output-postprocessed/movies/lines/C.png 

# Learning
* The code (DSL code in `DSL/solver.ac` and C++ code in `main.cc`) is to give a working explanation of how the computation happens with Astaroth. 
* For a more robust solver see `samples/standalone_mpi/main.cc` and for more extensive equation setup see `acc-runtime/samples/mhd_modular/mhdsolver.ac`, which solves the ideal MHD equations.
* For learning more about the DSL see `acc-runtime/README.md`
