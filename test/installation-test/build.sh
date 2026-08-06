cmake -B ac_build -S $AC_HOME -DMPI_ENABLED=ON -DAC_STENCIL_ORDER=6 -DFFT_ENABLED=OFF -DOPTIMIZE_MEM_ACCESSES=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF -DBUILD_TESTS=OFF -DRUNTIME_COMPILATION=OFF -DDSL_MODULE_DIR=$AC_HOME/test/sor-test/DSL 
cmake --build ac_build --parallel
sudo cmake --install ac_build

cmake -B sor-build -S .
cmake --build sor-build



