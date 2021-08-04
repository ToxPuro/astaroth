## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(acc-runtime)

## Project-wide compilation flags
set(COMMON_FLAGS "-Wall -Wextra -Werror -Wdouble-promotion -Wfloat-conversion -Wshadow")

## Subdirectories
add_subdirectory(acc)
add_subdirectory(api)