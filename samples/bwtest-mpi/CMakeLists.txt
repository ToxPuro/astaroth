cmake_minimum_required(VERSION 3.17) # Required for moder CUDA::cudart linking

set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
find_package(MPI REQUIRED COMPONENTS C)
find_package(OpenMP)
find_package(CUDAToolkit)

add_executable(bwtest-mpi main.c)
target_link_libraries(bwtest-mpi MPI::MPI_C OpenMP::OpenMP_C CUDA::cudart_static CUDA::cuda_driver)
target_compile_options(bwtest-mpi PRIVATE -O3)
