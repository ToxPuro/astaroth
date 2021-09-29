set(CMAKE_CUDA_COMPILER "/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "11.1.74")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "14")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "9.1")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/targets/x86_64-linux/lib/stubs;/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/appl/spack/install-tree/gcc-4.8.5/libxml2-2.9.8-ffzbok/include/libxml2;/appl/spack/install-tree/gcc-9.1.0/hpcx-mpi-2.4.0-dnpuei/include;/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/mkl/include;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/include;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/include/c++/9.1.0;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/include/c++/9.1.0/x86_64-pc-linux-gnu;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/include/c++/9.1.0/backward;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib/gcc/x86_64-pc-linux-gnu/9.1.0/include;/usr/local/include;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib/gcc/x86_64-pc-linux-gnu/9.1.0/include-fixed;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/targets/x86_64-linux/lib/stubs;/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/targets/x86_64-linux/lib;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib64;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib/gcc/x86_64-pc-linux-gnu/9.1.0;/lib64;/usr/lib64;/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/lib64;/appl/spack/install-tree/gcc-9.1.0/hpcx-mpi-2.4.0-dnpuei/lib;/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/tbb/lib/intel64_lin/gcc4.7;/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/compiler/lib/intel64_lin;/appl/opt/cluster_studio_xe2019/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64_lin;/appl/spack/install-tree/gcc-4.8.5/gcc-9.1.0-vpjht2/lib;/appl/spack/install-tree/gcc-9.1.0/cuda-11.1.0-vvfuk2/lib64/stubs")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
