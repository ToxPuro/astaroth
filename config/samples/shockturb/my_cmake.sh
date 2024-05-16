#!/bin/bash

# This is a sample script. Please copy it to the directory you want to run the
# code in and customize occordingly.

# The following write the commit indentifier corresponding to the simulation
# run  into a file. This is to help keep track what version of the code was
# used to perform the simulation.

if [ -L "./COMMIT_CODE.log" ]; then
       	rm -f COMMIT_CODE.log
fi
cd $AC_HOME && pwd && git rev-parse HEAD > COMMIT_CODE.log && cd -
cp $AC_HOME/COMMIT_CODE.log .

#Prepare the dsl source files in a directory at the same level of the host
#directory as the run directory and replace the default simulation settings with
#those required for the shocktest
DIR=../DSL"$(basename "${PWD}")"
mkdir $DIR
rsync -avu $AC_HOME/acc-runtime/samples/mhd_modular/ $DIR
sed -i "s/LENTROPY (1)/LENTROPY (0)/" $DIR/mhdsolver.ac
sed -i "s/LFORCING (0)/LFORCING (1)/" $DIR/mhdsolver.ac
sed -i "s/LSHOCK (0)/LSHOCK (1)/" $DIR/mhdsolver.ac

# Run cmake to construct makefiles
# In the case you compile in astaroth/build/ directory. Otherwise change ".." to
# the correct path to astaroth/CMakeLists.txt

if  [[ "${HOSTNAME}" =~ *"tiara"* || "${HOSTNAME}" =~ "gp"*[8-11] ]];
then
     echo "building on Tiara"
     cmake -DOPTIMIZE_MEM_ACCESSES=ON -DDOUBLE_PRECISION=ON -DBUILD_SAMPLES=OFF  -DMPI_ENABLED=ON -DUSE_HIP=OFF -DUSE_CUDA_AWARE_MPI=OFF -DDSL_MODULE_DIR=$DIR -DCMAKE_CXX_COMPILER=/software/opt/gcc/9.1.0/bin/gcc -DCMAKE_C_COMPILER=/software/opt/gcc/9.1.0/bin/gcc $AC_HOME
elif  [[ "${HOSTNAME}" =~ "puhti"*[0-24]* ]];
then
     echo "building on Puhti"
     cmake -DOPTIMIZE_MEM_ACCESSES=ON -DDOUBLE_PRECISION=ON -DMPI_ENABLED=ON -DUSE_HIP=OFF -DUSE_CUDA_AWARE_MPI=ON -DUSE_POSIX_IO=ON -DDSL_MODULE_DIR=$DIR $AC_HOME
elif  [[ "${HOSTNAME}" =~ "mahti"*[0-11]* || "${HOSTNAME}" =~ "mahti"*[12-24]* ]];
then
     echo "building on Mahti exact"
     cmake -DOPTIMIZE_MEM_ACCESSES=ON -DDOUBLE_PRECISION=ON -DMPI_ENABLED=ON -DUSE_HIP=OFF -DUSE_CUDA_AWARE_MPI=ON -DUSE_POSIX_IO=ON -DDSL_MODULE_DIR=$DIR $AC_HOME
else
     echo "building on ${HOSTNAME}"
     cmake -DOPTIMIZE_MEM_ACCESSES=ON -DDOUBLE_PRECISION=ON -DMPI_ENABLED=ON -DUSE_CUDA_AWARE_MPI=ON -DDSL_MODULE_DIR=$DIR $AC_HOME
fi

# Standard compilation

make -j
