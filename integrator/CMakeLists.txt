## mpitest

#add_executable(integrator main.cc)
#target_link_libraries(integrator astaroth_core astaroth_utils)
#if(TARGET GSL::gsl)
#  target_link_libraries(integrator astaroth_core astaroth_utils GSL::gsl)
#endif()

find_package(pybind11)
find_package(GSL REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)
pybind11_add_module(ac_integrator main.cc)
target_link_libraries(ac_integrator
    PRIVATE
        astaroth_core
        astaroth_utils
)
target_compile_definitions(ac_integrator PRIVATE
    PYTHON_BINDINGS
)

target_link_libraries(ac_integrator PRIVATE astaroth_core astaroth_utils GSL::gsl)
install(TARGETS ac_integrator
    LIBRARY DESTINATION .
)
