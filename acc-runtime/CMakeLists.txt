## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(acc-runtime C CXX CUDA)

## Subdirectories
add_subdirectory(acc)
add_subdirectory(api)