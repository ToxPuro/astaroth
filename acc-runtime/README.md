# Building ACC runtime (incl. DSL files)

The DSL source files should have a postfix `*.ac` and by default there should be only one
`*.ac` file per directory. Optionally you can give the `*.ac` file to be compiled.

    * `mkdir build`

    * `cd build`

    * `cmake -DDSL_MODULE_DIR=<path (relative to the build directory) to the dir containing DSL sources> \
             [-DSL_MODULE_FILE=<you can optionally give the file in DSL_MODULE_DIR to be compiled>] ..`

    * `make -j`


## Debugging

As ACC is in active development, compiler bugs and cryptic error messages are
expected. In case of issues, please check the following files in
`acc-runtime/api` in the build directory.

Intermediate files: 

1. `user_kernels.ac.pp_stage*`. The DSL file after a specific preprocessing stage.
2. `user_kernels.h.raw`. The generated CUDA kernels without formatting applied.
3. `user_kernels_backup.h.`. The generated CUDA/CPU kernels with formatting applied. 

Final files: 

1. `user_defines.h`. The project-wide defines generated with the DSL.
2. `user_kernels.h`. The generated CUDA kernels.

To make inspecting the code easier, we recommend using an
autoformatting tool, for example, `clang-format` or GNU `indent`.


## Known issues

  * The final function call of a kernel gets sometimes dropped due to an unknown reason. For instance:
  ```
  Kernel kernel() {
    ...
    device_function(...) // This call is not present in `user_kernels.h`
  }
  ```
  If you're able to reproduce this, please create a bitbucket issue with the incorrectly translated DSL code.

# The Astaroth Domain-Specific Language

The Astaroth Domain-Specific Language (DSL) is a high-level general-purpose(GP)GPU language
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
In our case, data streams correspond to vertices in a grid, similar to how
vertex shaders operate in graphics shading languages.

# Syntax


#### Comments and preprocessor directives
The Astaroth preprocessor works similar to the C preprocessor (gcc) with one exception: 

By default include files are searched from DSL_MODULE_DIR only if the include is not found are C include rules used. 

With two extensions: 
```
hostdefine ONE  (1) // Macro definition visible in both device and host code
#include ../stdlib/math //Includes all files and directories in ../stdlib/math
```

#### Variables
```
real var    // Explicit type declaration
real dconst // The type of device constants must be explicitly specified
run_const value_to_be_given //Value that will be constant during the run of the program. Used in runtime-compilation
input  real input_val  //Variable only used as input to e.g. `ComputeSteps`
output int output_val //Variable only used as output from e.g. reduce operation

var0 = 1    // The type of local variables can be left out (implicit typing)
var1 = 1.0  // Implicit precision (determined based on compilation flags)
var2 = 1.   // Trailing zero can be left out
var3 = 1e3  // E notation
var4 = 1.f  // Explicit single-precision
var5 = 0.1d // Explicit double-precision
var6 = "Hello"
```

> Note: Shadowing is not allowed, all identifiers within a scope must be unique

#### Arrays
```
int arr0 = [1, 2, 3] 
arr1 = [1.0, 2.0, 3.0] //inferred to be an array of reals
int arr2 = [[1,2,3], [3,4,5]] //Multidimensional arrays are supported
size(arr1) // Length of an array
gmem arr[AC_nx] //declaration for global array stored in GPU global memory
arr[3] (at global scope)         //equivalent to dconst arr[3]. Dimensions need to be known at compile time
gmem arr[AC_mx][AC_my]           //Multidimensional global array are supported. By default arrays are stored in column-major format but by setting ROW_MAJOR_ORDER=ON to cmake arrays are stored in row-major order
```

#### Casting
```
var7 = real(1)        // Cast
vec0 = real3(1, 2, 3) // Cast
```

#### Printing
```
// print is the same as `printf` in the C programming language
print("Hello from thread (%d, %d, %d)\n", vertexIdx.x, vertexIdx.y, vertexIdx.z)
```

#### Looping
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

#### Functions
```
func(param) {
    print("%s", param)
}

func2(val) {
    return val
}

//For overloading you have to specify the types of all input parameters

func3(real x) {
	return fabs(x)
}


func3(real3 v) {
	return 
		real3
		(
			fabs(v.x),
			fabs(v.y),
			fabs(v.z)
		)
}
elemental abs(real x)
{
	return fabs(x)
}

# Note `Kernel` type qualifier
Kernel func3() {
    func("Hello!")
}
```

> Note: Function parameters are **passed by constant reference**. Therefore input parameters **cannot be modified** and one may need to allocate temporary storage for intermediate values when performing more complex calculations.

The `elemental` type qualifier on a function means that it is a pure function that returns a value and it can be compossed on structures containing it
In the previous example functions we had some duplicate code since `func3` basically applies `func2` to all of its members. 
Since the `abs` function that takes in real value has been declared `elemental` it can be called also on `real3` and will produce the same effect if `abs` was called on all of the members of the parameter.

* A `elemental` function taking a `real` parameter can be called with:
	* `real`
	* `real3`
	* `Field`
	* `Field3`

* A `elemental` function taking a `real3` parameter can be called with:
	* `real3`
	* `Field3`

* A `elemental` function taking two `real3` parameters can be called with:
	* `real3`,`real3`
	* `real3`,`Field3`
	* `Field3`,`real3`
	* `Field3`,`Field3`

The semantics of passing a `Field` value to `elemental` functions is always the same as if `value` (explained in Fields section) was first called on the field

#### Stencils
```
// Format
<Optional reduction operation> Stencil <identifier> {
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

// Which is equivalent to
Sum Stencil example {
    ...
}

// and is calculated equivalently to
example(field) {
    return  a * field[IDX(vertexIdx.x - 1, vertexIdx.y, vertexIdx.z)] +
            b * field[IDX(vertexIdx.x,     vertexIdx.y, vertexIdx.z)] +
            c * field[IDX(vertexIdx.x + 1, vertexIdx.y, vertexIdx.z)]
}

By default, the binary operation for reducing `Stencil` elements is `Sum` (as above). Currently supported operations are `Sum` and `Max`. See the up-to-date list of the supported operations in `acc-runtime/acc/ac.y` rule `type_qualifier`.

// Real-world example
Max Stencil largest_neighbor {
    [1][0][0]  = 1,
    [-1][0][0] = 1,
    [0][1][0]  = 1,
    [0][-1][0] = 1,
    [0][0][1]  = 1,
    [0][0][-1] = 1,
}

real AC_dsx 
real AC_dsy 
real AC_dsz 

Stencil derx {
    [0][0][-3] = AC_dsx*-DER1_3,
    [0][0][-2] = AC_dsx*-DER1_2,
    [0][0][-1] = AC_dsx*-DER1_1,
    [0][0][1]  = AC_dsx*DER1_1,
    [0][0][2]  = AC_dsx*DER1_2,
    [0][0][3]  = AC_dsx*DER1_3
}

Stencil dery {
    [0][-3][0] = AC_dsy*-DER1_3,
    [0][-2][0] = AC_dsy*-DER1_2,
    [0][-1][0] = ...
}

Stencil derz {
    [-3][0][0] = AC_dsz*-DER1_3,
    [-2][0][0] = ...
}

gradient(field) {
    return real3(derx(field), dery(field), derz(field))
}
```

> Note: Stencil coefficients supplied in the DSL source must be either compile-time constants or device constants. For device constants the values are looked up from the loaded config. Stencil coefficients can also be set up coefficients at runtime with API calls, see [instructions below](#loading-and-storing-stencil-coefficients-at-runtime).

> Note: To reduce redundant communication or to enable larger stencils, the stencil order can be changed by declaring `#DEFINE STENCIL\_ORDER (YOUR VALUE)` in DSL. Modifying the stencil order with the DSL is currently not supported.


#### Fields

A `Field` is a scalar array that can be used in conjuction with `Stencil` operations. For convenience, a vector field can be constructed from three scalar fields by declaring them a `Field3` structure.
To get the value of a `Field` or `Field3` at the current vertex use the built-in function `value`
A `Field` can be used in unary expressions and binary arithmetic expressions (`+`, `-`, `/`,`*`)  which is equivalent to calling `value` on the `Field`
Calling a built-in math functions with `Field` is equivalent first calling `value` on the `Field`
```
Field ux, uy, uz // Three scalar fields `ux`, `uy`, and `uz`
#define uu Field3(ux, uy, uz) // A vector field `uu` consisting of components `ux`, `uy`, and `uz`

Kernel kernel() {
    write(ux, derx(ux))       // Writes the x derivative of the field `ux` to the output buffer
    field_val   = value(ux)   // Gets the value of ux from the input buffer at the current vertex
    field3_val  = value(uu)   // Gets the value of ux,uy,uz from the input buffer at the current vertex
    unary_expr  = -ux         // is equivalent to -value(ux)
    binary_expr = -ux*2.0*uz  // is equivalent to -value(ux)*2.0*value(uz)
    exp_val     = exp(uz)     // is equivalent to exp(value(uz))
}
```

#### Built-in variables and functions
```
// Variables
dim3 threadIdx       // Current thread index within a thread block (see CUDA docs)
dim3 blockIdx        // Current thread block index (see CUDA docs)
dim3 vertexIdx       // The current vertex index within a single device
dim3 globalVertexIdx // The current vertex index across multiple devices
dim3 globalGridN     // The total size of the computational domain (incl. all subdomains of all processes)

// Functions
void write(Field, real)  // Writes a real value to the output field at 'vertexIdx'
void print("int: %d", 0) // Printing. Uses the same syntax as printf() in C
real dot(real3, real3)   // Dot product
real3 cross(real3 a, real3 b) // Right-hand-side cross product a x b
size_t len(arr) // Returns the length of an array `arr`

// Trigonometric functions (Accessible from stdlib/math)
exp
sin
cos
sqrt
fabs

// Advanced functions (should avoid, dangerous)
real previous(Field) // Returns the value in the output buffer. Read after write() results in undefined behaviour.

// Constants
real AC_REAL_PI // Value of pi using the same precision as `real`
```

> See astaroth/acc-runtime/acc/codegen.c, function `symboltable_reset` for an up-to-date list of all built-in symbols.

# Advanced

The input arrays can also be accessed without declaring a `Stencil` as follows.

```
Field field0
Field field1

Kernel kernel() {
  // The example showcases two ways of accessing a field element without the Stencil structure
  int3 coord = ...
  //Writing to the input array is inherently unsafe so it really only be done inside boundary conditions
  field0[coord.x][coord.y][coord.y] = field1[coord.x][coord.y][coord.z]
}
```

> Note: Accessing field elements this way is suboptimal compared to accessing then using a `Stencil` or calling `value` since the reads are not cached. Only access field elements this way if otherwise not possible.

# Interaction with the Astaroth Core and Utils libraries

## Loading and storing stencil coefficients at runtime
The Astaroth Runtime API provides the functions `acLoadStencil` and `acStoreStencil` for loading/storing stencil coefficients at runtime. This is useful for, say, for setting the coefficients programmatically if too cumbersome by hand. We however highly recommend to use device constant variables in the stencils instead of separately loading stencil coefficients at runtime, since it eliminates error where the actual stencil coefficients are calculated separately from the stencil declarations.

See also the functions `acDeviceLoadStencil`, `acDeviceStoreStencil`, `acGridLoadStencil`, and `acGridStoreStencil` provided by the Astaroth Core library.


## Additional physics-specific API functions

To enable additional API functions in the Astaroth Core library for integration (`acIntegrate` function family) and MHD-specific tasks (automated testing, MHD samples), one must set `hostdefine AC_INTEGRATION_ENABLED (1)` in the DSL file. Note that if used in the DSL code, the hostdefine must not define anything that is not visible at compile-time. For example, `hostdefine R_PI (M_PI)`, where `M_PI` is defined is some host header, `M_PI` will not be visible in the DSL code and will result in a compilation error. Additionally, code such as `#if M_PI` will be always false in the DSL source if `M_PI` is not visible in the DSL file.

> Note: The extended API depends on several hardcoded fields and device constants. It is not recommended to enable it unless you work on the MHD sample case (`acc-runtime/samples/mhd`) or its derivatives.

## Stencil order

The stencil order can be set by the user by `hostdefine STENCIL_ORDER (x)`, where `x` is the total number of cells on both sides of the center point per axis. For example, a simple von Neumann stencil is of order 2.

> Note: The size of the halo surrounding the computational domain depends on `STENCIL_ORDER`.


## Reductions
This is still a experimental feature that only works if MPI is enabled and which still possibly changes in the future.

Reductions work only if the kernel is called at each vertex point of the domain.
```
output real max_derux
Field ux
Kernel reduce_kernel()
{
	reduce_max(true,derx(ux),max_derux)
}
```

* The reduce function parameters take three parameters:
	* Whether to reduce or not during this call
	* The value to at this vertex
	* The output value to which to store the reduced value

After executing the kernels the reduction has to be finalized with calling either `acGridFinalizeReduceLocal(graph)`, which reduces the values only on the local subdomain, or `acGridFinalize` which will reduce the value across processes.
The reduced values can be accessed with `acDeviceGetOutput`.

## ComputeSteps
This is still a experimental feature that only works if MPI is enabled and which still possibly changes in the future.

`ComputeSteps` are used to declare steps of kernel call invocations from which a `TaskGraph` is produced for the user.
```
input real ac_input_val
ComputeSteps(boundconds)
{
	kernel_call_1(ac_input_val,2.0)
	kernel_call_2()
	...
	...
	...
}
```
`ComputeSteps` take in a `BoundConds` which is used to calculate the values of `Field`s when the values at the boundaries are needed.

```
Field x
Field y
NGHOST_VAL = 3
int z_top
int z_bot
bc_sym_z(Field x, bool bottom)
{
	if(bottom)
	{
		for i in 0:NGHOST_VAL {
        		field[vertexIdx.x][vertexIdx.y][z_bot-i]=field[vertexIdx.x][vertexIdx.y][z_bot+i];
      		}

	}
	else
	{
		for i in 0:NGHOST_VAL {
        		field[vertexIdx.x][vertexIdx.y][z_bot-i]=field[vertexIdx.x][vertexIdx.y][z_top+i];
      		}
	}
}
set_y_z_bc(bool bottom)
{
	for i in 0:NGHOST_VAL {
        	y[vertexIdx.x][vertexIdx.y][z_bot-i]=y[vertexIdx.x][vertexIdx.y][z_bot+i];
      	}

	for i in 0:NGHOST_VAL {
        	y[vertexIdx.x][vertexIdx.y][z_bot-i]=y[vertexIdx.x][vertexIdx.y][z_top+i];
      	}
}
BoundConds(boundconds)
{
	periodic(BOUNDARY_XY)
	bc_sym_z(BOUNDARY_Z_TOP,x,false)
	bc_sym_z(BOUNDARY_Z_BOT,x,true)
	set_y_z_bc(BOUNDARY_Z)
}
```
The functions are called as normally in `BoundConds` except the first parameter represents which boundary the function is being used on.
The DSL compiler inserts calls to functions in `BoundConds` and communications between the processes as needed by the data dependencies of the `Stencil` operations inside the called kernels.


