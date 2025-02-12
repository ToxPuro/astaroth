## Packages
set(MPI_CXX_SKIP_MPICXX ON) # Skip MPI C++ bindings
find_package(MPI REQUIRED COMPONENTS C)

## Benchmarks
add_executable(acm-bm bm.cc)
target_link_libraries(acm-bm PRIVATE acm MPI::MPI_C)

