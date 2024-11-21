# Astaroth Communication Module

## Development notes
- Use C++ exceptions to catch and throw unrecoverable errors in non-MPI code
- In MPI code, call MPI_Abort if an unrecoverable error was encountered or if an exception was caught
- Testing is automated with CMake CTest. Test programs should primarily throw an exception on failure, or return an error return value if exceptions cannot be used. Automated tests can be run by `cmake .. && make -j && ctest`.

## Design
- `ac::array<type, maximum capacity>` is a custom structure used throughout the module to enable easy and efficient interoperability between C/C++/CUDA and MPI. It is a statically allocated, n-dimensional array type similar to `Vec3` in graphics APIs. Operators for basic arithmetic is enabled for integer types, enabling concise coordinate calculations using syntax `ac::array<uint64_t, N> a{1,2,3}; ac::array<uint64_t, N> b{1,2,3}; auto c{a + b};`. The structure is specialized for common use-cases, e.g., `ac::shape` and `ac::index` are aliases of the `ac::array` structure. Furthermore, `ac::array` can also be used to encapsulate and pass arrays of objects as CUDA kernel parameters, like so: `ac::array<double*, MAX_NBUFFERS> inputs{buf0, buf1}; kernel<<<...>>>(inputs, ...);`. This approach eliminates the need to use global memory for arrays of kernel parameters and circumvents concurrency issues arising from sharing and updating device constants between kernel launches.
- Instead of duplicating the information stored in MPI communicators, the Astaroth communication module provides a thin wrapper for common MPI operations, such as creating communicators and waiting for packets. The wrapper functions perform extensive error checking and translation between MPI-C -style rows first data formats to Astaroth's C++/CUDA -style columns first format.

## C++ features and conventions

- Braced initialization `int a{1}` should be preferred over `int a(1)` or `int a = 1` in cases where it makes sense. Syntax `fn(1)` should be reserved for function calls. Syntax `int a = 1.234` is OK, but note that it allows implicit conversion and `int a{1.234}` does not compile. However, with proper flags the compiler does also warn about `int a = 1.234`.
- In contrast to pure C libraries which should return error codes, prefer returning values from C++ functions and throw exceptions in case of errors instead.
- Prefer standard library functions (i.e `std::vector`) instead of rolling your own, but limit their use to ensure CUDA/C interoperability. Many standard library features either do not work, or are inoptimally implemented for CUDA.
- Structs should be kept as simple as possible and close to C to avoid C++ pitfalls when developers with varying level of experience work with the code. For example, the drawbacks of using i.e. inheritance often outweigh the benefits and make the code harder to maintain, especially when composition is almost always a better option.
- The classic C-style/procedural approach should always be considered first, i.e., if functionality can be implemented using simple structs and composition, prefer that over complex templated classes that have private/protected/public members that inherit and overload many functions. Generic/abstract programming should only be used in cases where it makes the system easier to understand and reduces the amount of code.
- Nevertheless, strive to write idiomatic C++ (RAII, returning values, throwing exceptions, passing function parameters by reference&, avoiding raw pointers where possible, etc)


# Concurrency

- *MPI:* All calls should originate from a single thread: `std::future`, `pthreads`, `fork`, and other mechanisms should not be used because
    - MPI calls are executed sequentially even if called concurrently: therefore, if one thread is blocking on one MPI call, another thread cannot launch any further MPI calls before the first thread has completed its MPI call
    - "When multiple threads make MPI calls concurrently, the outcome will be as if the calls executed sequentially in some (any) order" W. Gropp, lecture slides: https://wgropp.cs.illinois.edu/courses/cs598-s15/lectures/lecture36.pdf
    - *Immediate result:* achieving asynchronous execution with synchronous MPI + threads *is not possible*. The only way to achieve truly asynchronous calls with MPI is to use MPI's asynchronous interface or a separate program with its own MPI context.
    - Furthermore, spawning new processes is strongly discouraged by major MPI implementations. See, e.g., https://docs.open-mpi.org/en/v5.0.0/tuning-apps/fork-system-popen.html
    - *Recommendation:* use the asynchronous MPI interface and `MPI_Request`s for handling concurrency with MPI

- *CUDA:* concurrency should be handled exclusively with `cudaStream_t`s. Using `std::future` and other tools for creating host threads requires special care because at least the current device and new streams are exclusive to a specific host thread, and these must be set/created in each host thread.

- Host threads: if asynchronous MPI IO is not supported on the system, implementing it with `std::future` or other host thread mechanisms requires that MPI is initialized with `MPI_init_threads`, which will likely cause a slight overhead to all MPI calls and may not be supported or work well on all system. The safest approach is to rely on single-threaded MPI, and use whichever asynchronous MPI IO functions are available. If asynchronous MPI IO is not available on the system, there is likely a reason for it and implementing a proper workaround ourselves will likely be non-trivial.

- *Bottom line:* **the whole communication module should be completely single-threaded, and rely on MPI and CUDA features for achieving asynchronous execution.**

- Further notes about Thrust. *It should be assumed that all Thrust functions are asynchronous.* In practice, functions that return values to host are generally synchronous, but thrust's asynchronous model is *not well defined*. Therefore one needs to be very careful with thrust functions. Prefer explicit functions provided by Astaroth and avoid Thrust's own functions for concurrent implementations. In non-concurrent cases and if using Thrust, use `cudaDeviceSynchronize` to ensure that the function has completed.

# Base types

```C++
namespace ac{
    #if defined (CUDA_ENABLED)
    template <typename T, size_t N> using base_array = cuda::std::array<T, N>;
    #else
    template <typename T, size_t N> using base_array = std::array<T, N>;
    #endif

    template<typename T, size_t N>
    class array {
        ac::base_array<T, N> resource;
    };

    template<size_t N> using shape = ac::array<uint64_t, N>;

    template<typename T, typename MemoryResource>
    class buffer {
        std::size_t capacity;
        std::unique_ptr<T> resource;
    };

    template <typename T, size_t N, typename MemoryResource>
    struct ndbuffer {
        ac::shape<N> shape;
        ac::buffer<T, MemoryResource> resource;
    }
}
```


# Naming conventions
- Internal structures: Lower-case in namespace `ac`, e.g., `ac::shape`.
- Outward-facing structures and template parameters: CamelCase, e.g., `template<typename T, typename MemoryResource> BufferExchangeTask`.
