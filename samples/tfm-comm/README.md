# Astaroth Communication Module

## Development notes
- Use C++ exceptions to catch and throw unrecoverable errors in non-MPI code
- In MPI code, call MPI_Abort if an unrecoverable error was encountered or if an exception was caught
- Testing is automated with CMake CTest. Test programs should primarily throw an exception on failure, or return an error return value if exceptions cannot be used. Automated tests can be run by `cmake .. && make -j && ctest`.

## Design
- `StaticArray<type, maximum capacity>` is a custom structure used throughout the module to enable easy and efficient interoperability between C/C++/CUDA and MPI. It is a statically allocated, n-dimensional vector type similar to `Vec3` in graphics APIs. Operators for basic arithmetic is enabled for integer types, enabling concise coordinate calculations using syntax `StaticArray<uint64_t, N> a = {1,2,3}; StaticArray<uint64_t, N> b = {1,2,3}; auto c = a + b;`. The structure is specialized for common use-cases, e.g., `Shape` and `Index` are aliases of the `StaticArray` structure. Furthermore, `StaticArray` can also be used to encapsulate and pass arrays of objects as CUDA kernel parameters, like so: `StaticArray<double*, MAX_NBUFFERS> inputs = {buf0, buf1}; kernel<<<...>>>(inputs, ...);`. This approach eliminates the need to use global memory for arrays of kernel parameters and circumvents concurrency issues arising from sharing and updating device constants between kernel launches.
- Instead of duplicating the information stored in MPI communicators, the Astaroth communication module provides a thin wrapper for common MPI operations, such as creating communicators and waiting for packets. The wrapper functions perform extensive error checking and translation between MPI-C -style rows first data formats to Astaroth's C++/CUDA -style columns first format.
