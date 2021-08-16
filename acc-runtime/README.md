# Building and running

    * `mkdir build`

    * `cd build`

    * `cmake ..`

    * `make -j`

And create an executable script file that looks like this (note the shebang #!):

```
#!<PATH TO ACC BINARY DIR>/acc
print "Hello!"
```

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

var = 1    // The type of local variables can be left out (implicit typing)
var = 1.0  // Implicit precision (determined based on compilation flags)
var = 1.   // Trailing zero can be left out
var = 1e3  // E notation
var = 1.f  // Explicit single-precision
var = 0.1d // Explicit double-precision
var = "Hello"
```

Arrays
```
// Note: Arrays have fixed length
arr = 1, 2, 3
arr = 1.0, 2.0, 3.0
arr = "a", "b", "c"
```

Printing
```
print("Hello from thread (%d, %d, %d)\n", vertexIdx.x, vertexIdx.y, vertexIdx.z)
```

Looping
```
arr = 1, 2, 3
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
real previous(Field) // Returns the value in the output buffer. Read after write results in undefined behaviour.
write(field, 0)
a = previous(field) // The value of 'a' is undefined
```

> Note: Parameters are passed by value

> Note: Shadowing is not allowed

# Debugging

As ACC is in active development, compiler bugs and cryptic error messages are
expected. In case of issues, please check the following files in the build directory.

1. `user_kernels.ac.preprocessed`. The DSL file after preprocessing.
1. `user_defines.h`. The project-wide defines generated with the DSL.
1. `user_declarations.h`. Forward declarations of user kernels.
1. `user_kernels.h`. The generated CUDA kernels.

To make inspecting the code easier, we recommend using an
autoformatting tool, for example, `clang-format` or GNU `indent`.