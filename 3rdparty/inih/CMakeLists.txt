## CMake settings
cmake_minimum_required(VERSION 3.0)

## Project settings
project(inih C)

## Library
add_library(inih SHARED ini.c)