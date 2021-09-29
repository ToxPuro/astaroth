set(CMAKE_Fortran_COMPILER "/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/bin/gfortran")
set(CMAKE_Fortran_COMPILER_ARG1 "")
set(CMAKE_Fortran_COMPILER_ID "GNU")
set(CMAKE_Fortran_COMPILER_VERSION "9.1.0")
set(CMAKE_Fortran_COMPILER_WRAPPER "")
set(CMAKE_Fortran_PLATFORM_ID "")
set(CMAKE_Fortran_SIMULATE_ID "")
set(CMAKE_Fortran_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_Fortran_COMPILER_AR "/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/bin/gcc-ar")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_Fortran_COMPILER_RANLIB "/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/bin/gcc-ranlib")
set(CMAKE_COMPILER_IS_GNUG77 1)
set(CMAKE_Fortran_COMPILER_LOADED 1)
set(CMAKE_Fortran_COMPILER_WORKS TRUE)
set(CMAKE_Fortran_ABI_COMPILED TRUE)
set(CMAKE_COMPILER_IS_MINGW )
set(CMAKE_COMPILER_IS_CYGWIN )
if(CMAKE_COMPILER_IS_CYGWIN)
  set(CYGWIN 1)
  set(UNIX 1)
endif()

set(CMAKE_Fortran_COMPILER_ENV_VAR "FC")

set(CMAKE_Fortran_COMPILER_SUPPORTS_F90 1)

if(CMAKE_COMPILER_IS_MINGW)
  set(MINGW 1)
endif()
set(CMAKE_Fortran_COMPILER_ID_RUN 1)
set(CMAKE_Fortran_SOURCE_FILE_EXTENSIONS f;F;fpp;FPP;f77;F77;f90;F90;for;For;FOR;f95;F95)
set(CMAKE_Fortran_IGNORE_EXTENSIONS h;H;o;O;obj;OBJ;def;DEF;rc;RC)
set(CMAKE_Fortran_LINKER_PREFERENCE 20)
if(UNIX)
  set(CMAKE_Fortran_OUTPUT_EXTENSION .o)
else()
  set(CMAKE_Fortran_OUTPUT_EXTENSION .obj)
endif()

# Save compiler ABI information.
set(CMAKE_Fortran_SIZEOF_DATA_PTR "8")
set(CMAKE_Fortran_COMPILER_ABI "")
set(CMAKE_Fortran_LIBRARY_ARCHITECTURE "")

if(CMAKE_Fortran_SIZEOF_DATA_PTR AND NOT CMAKE_SIZEOF_VOID_P)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_Fortran_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_Fortran_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_Fortran_COMPILER_ABI}")
endif()

if(CMAKE_Fortran_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()





set(CMAKE_Fortran_IMPLICIT_INCLUDE_DIRECTORIES "/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib/gcc/x86_64-pc-linux-gnu/9.1.0/finclude;/appl/spack/install-tree/gcc-4.8.5/libxml2-2.9.8-ffzbok/include/libxml2;/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/include;/appl/spack/install-tree/gcc-9.1.0/hpcx-mpi-2.4.0-dnpuei/include;/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/mkl/include;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/include;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib/gcc/x86_64-pc-linux-gnu/9.1.0/include;/usr/local/include;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib/gcc/x86_64-pc-linux-gnu/9.1.0/include-fixed;/usr/include")
set(CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES "gfortran;m;gcc_s;gcc;quadmath;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES "/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib64;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib/gcc/x86_64-pc-linux-gnu/9.1.0;/lib64;/usr/lib64;/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/lib64;/appl/spack/install-tree/gcc-9.1.0/hpcx-mpi-2.4.0-dnpuei/lib;/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/tbb/lib/intel64_lin/gcc4.7;/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin;/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib;/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/lib64/stubs")
set(CMAKE_Fortran_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
