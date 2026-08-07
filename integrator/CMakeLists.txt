## mpitest


find_package(pybind11 QUIET)
find_package(GSL REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(pybind11_FOUND AND Python3_FOUND)
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
endif()

add_executable(integrator_standalone main.cc)
target_link_libraries(integrator_standalone astaroth_core astaroth_utils)
target_link_libraries(integrator_standalone astaroth_core astaroth_utils GSL::gsl)
