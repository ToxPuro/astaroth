from distutils.core import setup, Extension
import os 

mpi_compile_args = os.popen("mpicc.openmpi --showme:compile").read().strip().split(' ')
mpi_link_args    = os.popen("mpicc.openmpi --showme:link").read().strip().split(' ')

print(mpi_compile_args)
print(mpi_link_args)

module1 = Extension('mpitest',
                    #include_dirs = ['/usr/include/'],
                    libraries = ['mpi'],
                    #library_dirs = ['/usr/lib/x86_64-linux-gnu/'],
                    #library_dirs = ["-pthread -L/usr//lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi"]
                    library_dirs = ["/usr//lib", "/usr/lib/x86_64-linux-gnu/openmpi/lib"], 
                    extra_compile_args = mpi_compile_args,
                    extra_link_args    = mpi_link_args,
                    sources = ['mpitestmodule.c'])

setup (name = 'mpitest',
       version = '1.0',
       description = 'This is a MPI test',
       ext_modules = [module1])


