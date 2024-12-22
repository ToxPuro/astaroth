add_library(acc_runtime_compiler compiler.cpp ../src/core/update_params.cc ../src/core/decomposition.cc)
target_link_libraries(acc_runtime_compiler PRIVATE acc-runtime-headers)
target_include_directories(acc_runtime_compiler PUBLIC ${AC_BASE_PATH}/include)
