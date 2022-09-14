## CMake settings
cmake_minimum_required(VERSION 3.19)

## Project settings
project(acc-runtime)

## Project-wide compilation flags
set(COMMON_FLAGS "-Wall -Wextra -Wdouble-promotion -Wfloat-conversion -Wshadow")

## 
include_directories(.)

## Subdirectories
add_subdirectory(acc)
add_subdirectory(api)
