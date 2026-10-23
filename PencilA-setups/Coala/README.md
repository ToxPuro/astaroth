Steps to build and run:
Always remember to source Astaroth! Run source sourceme.sh from the Astaroth root.
cd $AC_HOME/PencilA-setups/Coala
./build.sh
Run from your preferred work directory
sbatch $AC_HOME/Pencil-A-setups/Coala/batch.sh (change the slurm settings to be compatible for your machine)
To verify your results against pencil sample 0d-tests/samples/coala run:
python3 $AC_HOME/analysis/test_tools/verify.py reference.ts timeseries.ts
