
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
(TODO) move API generation funcs to dir and mention that

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
Astaroth DSL compiler (< build dir >/acc-runtime/acc) is a source-to-source compiler, which converts
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

Since the language is still an active development user recommendations/requests for new features are **highly desirable**.
To make a recommendation/request for a new feature make pull request where you outline the new feature and the use case it would be needed in.

## Comments and preprocessor directives
The Astaroth preprocessor works similar to the C preprocessor (gcc) with one exception: 

By default include files are searched relative to DSL_MODULE_DIR. 
(TODO) only if the include is not found are C include rules used. 

With two extensions: 
```
hostdefine ONE  (1) // Macro definition visible in both device and host code
#include ../stdlib/math //Includes all files and directories in ../stdlib/math
```
## Datatypes
The Astaroth DSL language is statically typed.
All global variables require an explicit datatype declaration.
Otherwise explicit typing is allowed but not required, with the exception of the parameters of overloaded functions.
Some advanced features also require an explicit type declaration.
By default we use C++ type inference rules (`auto`) with extensions where they would be ambiguous.

### Fields

A `Field` is a scalar array that can be used in conjuction with `Stencil` operations. For convenience, 
a vector field can be constructed from three scalar fields by declaring them a `Field3` structure.
To get the value of a `Field` or `Field3` at the current vertex use the built-in function `value`.
A `Field` can be used directly in unary expressions and binary arithmetic expressions (`+`, `-`, `/`,`*`)
which is equivalent to calling `value` on the `Field`.
Also, calling a built-in math functions with `Field` is equivalent to first calling `value` on the `Field`.

> Note: Internally, Field is an enum handle (VertexBufferHandle) referring to the input and output buffers (data cubes).
> Note: The numerical value of the handle is monotonically increasing in the order of the declarations of the `Fields`.

```
Field ux, uy, uz // Three scalar fields `ux`, `uy`, and `uz`
const Field3 uu  = Field3(ux, uy, uz) // A vector field `uu` consisting of components `ux`, `uy`, and `uz`
field_val   = value(ux)   // Gets the value of ux from the input buffer at the current vertex
field3_val  = value(uu)   // Gets the value of ux,uy,uz from the input buffers at the current vertex
```
Arrays of `Field` variables can be declared with constant dimensions. The individual `Field` elements can be accessed by indexing the declared array.
```
const int n_species = ...
Field chemistry[n_species]
for i 0:n_species
{
    chemistry[i] = 0.0
}
```

### Primitive types
The following primitive C++ types are usable: 
* `int`

* `bool`

* `long`

* `long long`

* `real` (by default double, float if DOUBLE_PRECISION=OFF)
    
> Note: Whenever possible one should prefer using bools compared to e.g. integers which only have the values 0 and 1, since using bools gives the DSL compiler more information, which it can use to perform optimizations.

### Additional built-in types
* `complex`

* `real2`

* `real3`

* `real4`

* `int3`

* `Matrix` (3x3 matrix of reals)
    
We support `Matrix*real3`,`real*Matrix` and `-Matrix`.

### User-defined types
#### Structures
Structures can be defined similar to the C syntax:
```
struct your_struct
{
    real x
    real y
}

struct your_struct_2
{
    your_struct x 
    your_struct y
}
```
They can be initialized as in C, except for type inference an explicit cast might be needed.

> Note: Currently, declaring multiple members from a single type specifier is not allowed.

##### Operators 
If all of the members of the structure are `real` we provide member-wise arithmetic operators (`+`, `-`, `/`, `*`) with `real` scalars.
Additionally `+` and `-` are supported between structures of the same type, all of the members of which are `real` or `int`.
For all operators the corresponding compound assignment operator is also supported. 

#### Enums
Enums are declared in the C syntax
```
enum Characters
{
    A,
    B,
    C
}
```
> Note: Whenever possible one should prefer using enums compared e.g. named integers, since using enums gives the DSL compile more information which it can use to perform optimizations.

### Type qualifiers

* `const` Effectively the same as C++ const.

* `dconst` The implicit qualifier if no qualifiers are defined, for global variables, stored in the device constant memory of the GPU, which makes accesses fast.
Their values are loaded through the Astaroth config.

* `gmem` Used for arrays to be allocated on the global memory of GPU

>Note: For performance reasons you should use gmem arrays when different indexes are used at different vertices at the same time. 
Additionally too large arrays on the device constant memory can degrade performance by limiting the amount of available cache too much.

* `run_const`
Variables that are constant during the execution context of Astaroth (e.g. during a timeloop in a simulation).
By default the same as dconst, but with RUNTIME_COMPILATION=ON they will be effectively replaced by their value (C++ `constexpr`).


### Advanced
* `communicated`
The implicit qualifier for `Fields` if no qualifiers are defined, their halos are updated. 
* `auxiliary`
`Fields` where the input and output buffers are the same (e.g. no stencil operations and writes in the same kernel).

> Note: one can combine these to have communicated auxiliary `Fields`, without `communicated`, `auxiliary` `Fields` are not communicated.

The DSL compiler can also infer these qualifiers if OPTIMIZE_FIELDS=ON from `write`, `value` and `Stencil` calls.
**Important** requires that all conditionals are known at compile-time.

* `input`
Designed for variables that are inputs to Kernels, but should not be allocated/loaded to the GPU.
> Note: At the moment, mostly useful for `ComputeSteps`
* `output`
At the moment, restricted to `real` scalar quantities resulting from reductions across the whole subdomain.
> Note: implicitly allocates memory on the GPU to perform reductions.



## Variables
Variable declaration and naming conventions follow C, except
shadowing is not allowed: all identifiers within a scope must be unique.

### Arrays 
Instead of `{` `}` initializer of C++ we use `[` `]` 
```
int arr0 = [1, 2, 3] 
arr1 = [1.0, 2.0, 3.0] //type real is inferred
int arr2 = [[1,2,3], [3,4,5]] //Multidimensional arrays can be initialized (dimensions inferred)
size(arr1) // Size of the array
real arr[3] (at global scope)         //equivalent to dconst arr[3]. Dimensions need to be known at compile time
gmem real arr[AC_nx] //declaration for global array stored on the GPU global memory. Dimensions need to be known at compile time or be dconst int variables [not expressions involving dconsts].
gmem arr[AC_mx][AC_my]           //Multidimensional global array.
```
> Note: By default arrays are stored in column-major format,
but by setting ROW_MAJOR_ORDER=ON arrays are stored in row-major order
If OPTIMIZE_ARRAYS=ON the DSL compiler will identify unused `gmem` arrays and will not allocate them on the GPU.

```
Kernel kernel() {
    write(ux, derx(ux))       // Writes the x derivative of the field `ux` to the output buffer
    
    unary_expr  = -ux         // is equivalent to -value(ux)
    binary_expr = -ux*2.0*uz  // is equivalent to -value(ux)*2.0*value(uz)
    exp_val     = exp(uz)     // is equivalent to exp(value(uz))
}
```

### Casting
One can use C++ casting.
```
var7 = real(1)        // Cast
vec0 = real3(1, 2, 3) // Cast
```

## Looping
Loops follow the Python style
```
int arr = [1, 2, 3]
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

## Functions
Functions follow the C declaration syntax.
We support overloading for functions for which all input parameter types are declared.
```
func(param) {
    print("%s", param)
}

func2(val) {
    return val
}

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

```
> Note: Function parameters are **passed by constant reference**. Therefore input parameters **cannot be modified** and one may need to allocate temporary storage for intermediate values when performing more complex calculations.

The `elemental` type qualifier on a function means that it is a pure function that returns a value,
for which the function's semantics is composable on structures and arrays containing the type of the return value.

In the previous example functions we had some duplicate code since `func3` basically applies `func2` to all of its members. 
Since the `abs` function that takes in a real value has been declared `elemental` it can also be called with a `real3` and will produce the same effect as if `abs` was called on all of the members of the parameter.

The semantics of passing a `Field` parameter to `elemental` functions is always the same as if `value` was first called on the `Field`.

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

### Printing
```
// print is the same as `printf` in the C programming language
print("Hello from thread (%d, %d, %d)\n", vertexIdx.x, vertexIdx.y, vertexIdx.z)
```

### Kernels
Kernels are functions visible outside of the DSL code, called by the host.
Calling kernels is the only way to execute DSL code.

> Note: Kernels can not be called from other kernels.
> Note: Array input parameters are not supported
> Note: The types of the input parameters have to be declared.
> Note: There are different ways to pass input parameters to kernels through API functions.

```
Kernel func3(input param) {

}
```

### Stencils
Stencils are functions that take in a `Field` input parameter and have an unique syntax and semantics. 
**Importantly**, they are the only way to access other vertexes than the local of a `Field`.
A stencil operation multiplies all points of the stencil by their corresponding coefficients and performs a binary reduction operation (`Sum` or `Max`) over these values.
Because Stencils act uniformly on the input parameter it is not declared.
```
// Format
<Optional reduction operation> Stencil <identifier> {
    [z][y][x] = coefficient,
    ...
}
```
where [z][y][x] is the x/y/z offset from the current vertex position.

For example
```
Stencil example {
    [0][0][-1] = a,
    [0][0][0] = b,
    [0][0][1] = c
}
```
Which is equivalent to
```
Sum Stencil example {
    ...
}
```

And is calculated equivalently to
```
example(field) {
    return  a * field[IDX(vertexIdx.x - 1, vertexIdx.y, vertexIdx.z)] +
            b * field[IDX(vertexIdx.x,     vertexIdx.y, vertexIdx.z)] +
            c * field[IDX(vertexIdx.x + 1, vertexIdx.y, vertexIdx.z)]
}
```

By default, the binary operation for reducing `Stencil` elements is `Sum` (as above). Currently supported operations are `Sum` and `Max`.

```
// Real-world example
Max Stencil largest_neighbor {
    [1][0][0]  = 1,
    [-1][0][0] = 1,
    [0][1][0]  = 1,
    [0][-1][0] = 1,
    [0][0][1]  = 1,
    [0][0][-1] = 1
}

real AC_dsx 
real AC_dsy 
real AC_dsz 

//dconst variables can be used in stencil coefficients (their values are looked up from the config during loading).

Stencil derx {
    [0][0][-3] = -AC_dsx*DER1_3,
    [0][0][-2] = -AC_dsx*DER1_2,
    [0][0][-1] = -AC_dsx*DER1_1,
    [0][0][1]  = AC_dsx*DER1_1,
    [0][0][2]  = AC_dsx*DER1_2,
    [0][0][3]  = AC_dsx*DER1_3
}

Stencil dery {
    [0][-3][0] = -AC_dsy*DER1_3,
    [0][-2][0] = ...
}

Stencil derz {
    [-3][0][0] = -AC_dsz*DER1_3,
    [-2][0][0] = ...
}

gradient(field) {
    return real3(derx(field), dery(field), derz(field))
}
```

> Note: Stencil coefficients supplied in the DSL source must either be compile-time constants or dconst. The coefficients can also be loaded at runtime with API calls, see [instructions below](#loading-and-storing-stencil-coefficients-at-runtime).

> Note: To reduce redundant communication or to enable larger stencils, the stencil order can be changed by declaring `#DEFINE STENCIL\_ORDER (YOUR VALUE)` in DSL. Modifying the stencil order with the DSL is currently not supported.



## Built-in variables, functions and constants
```
// Variables
int3 vertexIdx       // The vertex index that is iterated over, within the subdomain of the current device
int3 globalVertexIdx // The vertex index that is iterated over, within the global domain
int3 globalGridN     // The total size of the computational domain (incl. all subdomains of all processes)
int3 threadIdx       // Current thread index within a thread block (see CUDA docs)
int3 blockIdx        // Current thread block index (see CUDA docs)

// Built-in Functions
void write(Field, real)  // Writes a real value to the output field at 'vertexIdx'
elemental dot(real3, real3)   // Dot product
real3 cross(real3 a, real3 b) // Right-hand-side cross product a x b
size_t size(arr) // Returns the length of an array `arr`
```
Accessible from stdlib/math
```
real sin(real) 
real cos(real) 
real tan(real) 
real atan2(real)
real sinh(real)
real cosh(real)
real tanh(real)
real/complex exp(real/complex)
real log(real)
real sqrt(real)
real pow(real)
real fabs(real)
real min(real,real)
real max(real,real)
int  ceil(real)
real random_uniform()
```
Advanced functions (should avoid, dangerous)
```
real previous(Field) // Returns the value in the output buffer. Call after write() results in undefined behaviour.


```
### Built-in constants
```
real AC_REAL_PI // Value of pi using the same precision as `real`
real AC_REAL_MAX // Either DBL_MAX or FLT_MAX base on precision of `real`
real AC_REAL_MIN // Either DBL_MIN or FLT_MIN base on precision of `real`
real AC_REAL_EPSILON // Either DBL_EPSILON or FLT_EPSILON base on precision of `real`
```
### Built-in dconsts

uniform spacings of the grid:
real AC_dsx
real AC_dsy
real AC_dsz
and their inverses:
real AC_inv_dsx
real AC_inv_dsy
real AC_inv_dsz
Subdomain size (not incl. halos)
int AC_nx
int AC_ny
int AC_nz
Subdomain size (incl. halos)
int AC_mx
int AC_my
int AC_mz
Domain size (not incl. halos)
int AC_nxgrid
int AC_nygrid
int AC_nzgrid
Derivatives of subdomain sizes
int AC_mxy //AC_mx*AC_my
int AC_nxy //AC_nx*AC_ny
int AC_nxyz //AC_nx*AC_ny*AC_nz
Physical domain extents
real AC_xlen
real AC_ylen
real AC_zlen
Library config parameters (explained on the library documentation)
Not meaningful for DSL
int AC_proc_mapping_strategy
int AC_decompose_strategy
int AC_MPI_comm_strategy
Coordinate vectors of a Lagrangian grid (need LAGRANGIAN_GRID=ON)
Field COORDS_X
Field COORDS_Y
Field COORDS_Z

## Advanced features
If OPTIMIZE_FIELDS=ON, the DSL compiler will identify unused `Fields` and will not allocate them on the GPU.
To still enable a loop across all allocated `Fields` non-allocated `Fields` are not counted in `NUM_FIELDS`
and their numerical index is higher than `NUM_FIELDS`. 

**Important** If you use runtime-compilation this can mean that the index value
of the main application that calls Astaroth and the index value inside it might not match. 
To get the correct indexes, use API functions acGet< Field name >.


The input arrays can also be accessed without declaring a `Stencil` as follows.
**Important!!** Do not use this if you do not know what you are doing.
``` 
Field field0
Field field1

Kernel kernel() {
  int3 coord = ...
  //Writing to the input array is inherently unsafe so it really only be done inside boundary conditions
  field0[coord.x][coord.y][coord.y] = field1[coord.x][coord.y][coord.z]
}
//Since the input parameter f has explicitly been given the type of Field one can access with it the input buffer,
which would not be possible without the explicit type declaration.
func(Field f)
{
    return field[vertexIdx.x][vertexIdx.y][vertexIdx.z]
}
```

> Note: Accessing field elements this way is suboptimal compared to accessing then using a `Stencil` or calling `value` since the reads are not cached. Only access field elements this way if otherwise not possible.
### Reductions
This is still an experimental feature that only works if MPI is enabled and which still possibly changes in the future.

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
	* The value to reduce at this vertex
	* The output value to which to store the reduced value

After executing the kernels the reduction has to be finalized with calling either `acGridFinalizeReduceLocal(graph)`, which reduces the values only on the local subdomain, or `acGridFinalize` which will reduce the value across processes.
The reduced values can be accessed with `acDeviceGetOutput`.

### ComputeSteps
**Requires that all conditionals are known at compile-time (Note compilation can happen at runtime)**
This is still a experimental feature that only works if MPI is enabled and which still possibly changes in the future.

The `BoundConds` construct is used to declare how to calculate the values of `Field`s when the boundary conditions are to be imposed.
```
BoundConds boundconds
{
	periodic(BOUNDARY_XY)
	bc_sym_z(BOUNDARY_Z_TOP,FIELD_X,false)
	bc_sym_z(BOUNDARY_Z_BOT,FIELD_Y,true)
	set_y_z_bc(BOUNDARY_Z) //Calculated fields inferred from write calls inside set_y_z_bc
}
```
It contains function calls that are used to calculate the values of the outermost halo regions.
`periodic` is a unique construct to tell that the domain is periodic in the specified directions.
Otherwise normal functions are called with an additional mandatory first parameter specifying the boundary.

The `ComputeSteps` construct is used to declare a sequence of (possibly dependent) kernel invocations in the DSL,
from which a `TaskGraph` is produced. 
The kernels are analysed to infer when halo exchanges and evaluations of boundary conditions are needed.

Dependencies between invocations are based on the needed `Fields` for the kernel and the `Fields` updated inside it.
E.g. if kernel_call_1 would update `A` and kernel_call_2 would read `A`, kernel_call_2 is only called after kernel_call_1 has updated `A`.
```
input real ac_input_val
ComputeSteps rhs(boundconds)
{
	kernel_call_1(ac_input_val,2.0)
	kernel_call_2()
	...
	...
	...
}
```

`ComputeSteps` take in a parameter of `BoundConds`, from which it knows which boundary conditions to impose.
In case required boundary conditions are not declared, the compilation will fail.
You can access the generated `TaskGraph` with `acGetDSLTaskGraph`.

```
bc_sym_z(Field field, bool bottom)
{
	if(bottom)
	{
		for i in 0:NGHOST {
        		field[vertexIdx.x][vertexIdx.y][NGHOST-i]=field[vertexIdx.x][vertexIdx.y][NGHOST+i];
      		}

	}
	else
	{
		for i in 0:NGHOST {
        		field[vertexIdx.x][vertexIdx.y][AC_nz+i]=field[vertexIdx.x][vertexIdx.y][AC_nz-i];
      		}
	}
}
```


# Interaction with the Astaroth Core and Utils libraries
## Loading and storing stencil coefficients at runtime
The Astaroth Runtime API provides the functions `acLoadStencil` and `acStoreStencil` for loading/storing stencil coefficients at runtime. 
This is useful for, say, for setting the coefficients programmatically if too cumbersome by hand. 
We however highly recommend to use device constant variables in the stencil coefficients instead of separately loading stencil coefficients at runtime
, since it eliminates coding errors where the actual stencil coefficients are calculated separately from the stencil declarations.

See also the functions `acDeviceLoadStencil`, `acDeviceStoreStencil`, `acGridLoadStencil`, and `acGridStoreStencil` provided by the Astaroth Core library.


## Additional physics-specific API functions

To enable additional API functions in the Astaroth Core library for integration (`acIntegrate` function family) and MHD-specific tasks (automated testing, MHD samples), one must set `hostdefine AC_INTEGRATION_ENABLED (1)` in the DSL file. Note that if used in the DSL code, the hostdefine must not define anything that is not visible at compile-time. For example, `hostdefine R_PI (M_PI)`, where `M_PI` is defined is some host header, `M_PI` will not be visible in the DSL code and will result in a compilation error. Additionally, code such as `#if M_PI` will be always false in the DSL source if `M_PI` is not visible in the DSL file.

> Note: The extended API depends on several hardcoded fields and device constants. It is not recommended to enable it unless you work on the MHD sample case (`acc-runtime/samples/mhd`) or its derivatives.

## Stencil order

The stencil order can be set by the user by `hostdefine STENCIL_ORDER (x)`, where `x` is the total number of cells on both sides of the center point per axis. For example, a simple von Neumann stencil is of order 2.

> Note: The size of the halo surrounding the computational domain depends on `STENCIL_ORDER`.
