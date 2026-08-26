#!/bin/bash
cmake -B build -S $AC_HOME -DBUILD_SAMPLE=blur && cmake --build build -j
