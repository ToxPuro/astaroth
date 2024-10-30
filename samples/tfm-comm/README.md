# Astaroth Communication Module

## Development notes
- Use C++ exceptions to catch and throw unrecoverable errors in non-MPI code
- In MPI code, call MPI_Abort if an unrecoverable error was encountered or if an exception was caught
- Testing is automated with CMake CTest. Test programs should primarily throw an exception on failure, or return an error return value if exceptions cannot be used. Automated tests can be run by `cmake .. && make -j && ctest`.

## Design
- `StaticArray<type, maximum capacity>` is a custom structure used throughout the module to enable easy and efficient interoperability between C/C++/CUDA and MPI. It is a statically allocated, n-dimensional vector type similar to `Vec3` in graphics APIs. Operators for basic arithmetic is enabled for integer types, enabling concise coordinate calculations using syntax `StaticArray<uint64_t, N> a = {1,2,3}; StaticArray<uint64_t, N> b = {1,2,3}; auto c = a + b;`. The structure is specialized for common use-cases, e.g., `Shape` and `Index` are aliases of the `StaticArray` structure. Furthermore, `StaticArray` can also be used to encapsulate and pass arrays of objects as CUDA kernel parameters, like so: `StaticArray<double*, MAX_NBUFFERS> inputs = {buf0, buf1}; kernel<<<...>>>(inputs, ...);`. This approach eliminates the need to use global memory for arrays of kernel parameters and circumvents concurrency issues arising from sharing and updating device constants between kernel launches.
- Instead of duplicating the information stored in MPI communicators, the Astaroth communication module provides a thin wrapper for common MPI operations, such as creating communicators and waiting for packets. The wrapper functions perform extensive error checking and translation between MPI-C -style rows first data formats to Astaroth's C++/CUDA -style columns first format.

## C++ features and conventions

- Braced initialization `int a{1}` should be preferred over `int a(1)` or `int a = 1` in cases where it makes sense. Syntax `fn(1)` should be reserved for function calls. Syntax `int a = 1.234` is OK, but note that it allows implicit conversion and `int a{1.234}` does not compile. However, with proper flags the compiler does also warn about `int a = 1.234`.
- In contrast to pure C libraries which should return error codes, prefer returning values from C++ functions and throw exceptions in case of errors instead.
- Prefer standard library functions (i.e `std::vector`) instead of rolling your own, but limit their use to ensure CUDA/C interoperability. Many standard library features either do not work, or are inoptimally implemented for CUDA.
- Structs should be kept as simple as possible and close to C to avoid C++ pitfalls when developers with varying level of experience work with the code. For example, the drawbacks of using i.e. inheritance often outweigh the benefits and make the code harder to maintain, especially when composition is almost always a better option.
- The classic C-style/procedural approach should always be considered first, i.e., if functionality can be implemented using simple structs and composition, prefer that over complex templated classes that have private/protected/public members that inherit and overload many functions. Generic/abstract programming should only be used in cases where it makes the system easier to understand and reduces the amount of code.
- Nevertheless, strive to write idiomatic C++ (RAII, returning values, throwing exceptions, passing function parameters by reference&, avoiding raw pointers where possible, etc)
