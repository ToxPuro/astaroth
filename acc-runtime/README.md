# Building ACC runtime (incl. DSL files)

The DSL source files should have a postfix `*.ac` and there should be only one
`.ac` file per directory.

    * `mkdir build`

    * `cd build`

    * `cmake -DDSL_MODULE_DIR=<optional path to the dir containing DSL sources> ..`

    * `make -j`


## Debugging

As ACC is in active development, compiler bugs and cryptic error messages are
expected. In case of issues, please check the following files in
`acc-runtime/api` in the build directory.

1. `user_kernels.ac.preprocessed`. The DSL file after preprocessing.
1. `user_defines.h`. The project-wide defines generated with the DSL.
1. `user_declarations.h`. Forward declarations of user kernels.
1. `user_kernels.h`. The generated CUDA kernels.

To make inspecting the code easier, we recommend using an
autoformatting tool, for example, `clang-format` or GNU `indent`.


# The Astaroth Domain-Specific Language

The Astaroth Domain-Specific Language (DSL) is a high-level GPGPU language
designed for improved productivity and performance in stencil computations. The
Astaroth DSL compiler (acc) is a source-to-source compiler, which converts
DSL kernels into CUDA/HIP kernels. The generated kernels provide performance
that is on-par with hand-tuned low-level GPGPU code in stencil computations.
Special care has been taken to ensure efficient code generation in use cases
encountered in computational physics, where there are multiple coupled fields,
which makes manual caching notoriously difficult.

The Astaroth DSL is based on the stream processing model, where an array of
instructions is executed on streams of data in parallel. A kernel is a small
GPU program, which defines the operations performed on a number of data streams.
In our case, data streams correspond to a vertices in a grid, similar to how
vertex shaders operate in graphics shading languages.

# Syntax

Comments and preprocessor directives
```
// This is a comment
#define    ZERO (0)   // Visible only in device code
hostdefine ONE  (1) // Visible in both device and host code
```

Variables
```
real var    // Explicit type declaration
real dconst // The type of device constants must be explicitly specified

var0 = 1    // The type of local variables can be left out (implicit typing)
var1 = 1.0  // Implicit precision (determined based on compilation flags)
var2 = 1.   // Trailing zero can be left out
var3 = 1e3  // E notation
var4 = 1.f  // Explicit single-precision
var5 = 0.1d // Explicit double-precision
var6 = "Hello"
```

> Note: Shadowing is not allowed, all identifiers within a scope must be unique

Arrays
```
int arr0 = 1, 2, 3 // The type of arrays must be explicitly specified
real arr1 = 1.0, 2.0, 3.0
// len(arr1) // Length of an array **(disabled in the current build)**
```

Printing
```
// print is the same as `printf` in the C programming language
print("Hello from thread (%d, %d, %d)\n", vertexIdx.x, vertexIdx.y, vertexIdx.z)
```

Looping
```
int arr = 1, 2, 3
for var in arr {
    print("%d\n", var)
}

i = 0
while i < 3 {
    i += 1
}

for i in 0:10 { // Note: 10 is exclusive
  print("%d", i)
}
```

Functions
```
func(param) {
    print("%s", param)
}

func2(val) {
    return val
}

# Note `Kernel` type qualifier
Kernel func3() {
    func("Hello!")
}
```

> Note: Function parameters are **passed by reference**

> Note: Overloading is not allowed, all function identifiers must be unique

Stencils
```
// Format
Stencil <identifier> {
    [z][y][x] = coefficient,
    ...
}
// where [z][y][x] is the x/y/z offset from current position.

// For example,
Stencil example {
    [0][0][-1] = a,
    [0][0][0] = b
    [0][0][1] = c,
}
// is calculated equivalently to
example(field) {
    return  a * field[IDX(vertexIdx.x - 1, vertexIdx.y, vertexIdx.z)] +
            b * field[IDX(vertexIdx.x,     vertexIdx.y, vertexIdx.z)] +
            c * field[IDX(vertexIdx.x + 1, vertexIdx.y, vertexIdx.z)]
}

// Real-world example
Stencil derx {
    [0][0][-3] = -DER1_3,
    [0][0][-2] = -DER1_2,
    [0][0][-1] = -DER1_1,
    [0][0][1]  = DER1_1,
    [0][0][2]  = DER1_2,
    [0][0][3]  = DER1_3
}

Stencil dery {
    [0][-3][0] = -DER1_3,
    [0][-2][0] = -DER1_2,
    [0][-1][0] = ...
}

Stencil derz {
    [-3][0][0] = -DER1_3,
    [-2][0][0] = ...
}

gradient(field) {
    return real3(derx(field), dery(field), derz(field))
}
```

> Note: Stencil coefficients supplied in the DSL source must be compile-time constants. To set up coefficients at runtime, use the API function `acDeviceLoadStencils()`.


Built-in variables and functions
```
// Variables
dim3 threadIdx       // Current thread index within a thread block (see CUDA docs)
dim3 blockIdx        // Current thread block index (see CUDA docs)
dim3 vertexIdx       // The current vertex index within a single device
dim3 globalVertexIdx // The current vertex index across multiple devices

// Functions
write(Field, real) // Writes a real value to the output field at 'vertexIdx'

// Advanced functions (should avoid, dangerous)
real previous(Field) // Returns the value in the output buffer. Read after write() results in undefined behaviour.
```
