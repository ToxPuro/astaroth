Gauss-Legendre quadrature requires GSL which can be installed on Linux:
sudo apt install libgsl-dev
And on Mac with:
brew install gsl

Build the code using `./build.sh`
Run it using either `./run.sh` or `mpirun -n 1 ./build/integrate-test NXPOINTS NYPOINTS`
or see `batch.sh` for SLURM usage.
For multiple processes points in the first non-unity direction needs to be divisible by the number of processes.

Integration bounds given in `integrate.conf`
If you want to integrate in log-space set AC_logspace = T in `integrate.conf`.
You can also choose integration method in `integrate.conf`
Integrand given in function `integrand()` in `DSL/solver.ac`
For integration over wavenumbers the wavenumber values can be given as a csv-file called `k.dat`
and the output will be in `res.dat`
