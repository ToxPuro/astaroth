## CMake settings
cmake_minimum_required(VERSION 3.5)

## Project settings
project(inih C)

## Library
add_library(inih STATIC ini.c)
target_include_directories(inih PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
