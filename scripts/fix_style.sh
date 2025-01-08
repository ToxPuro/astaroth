#!/usr/bin/env bash
if [[ $1 == "DO" && $2 == "IT!" ]]; then
    find ./include ./src ./acc-runtime -name "*.h" -o -name "*.cc" -o -name "*.c" -o -name "*.cu" -o -name "*.cuh" | xargs clang-format -i -style=file
    echo "It is done."
else
    find ./include ./src ./acc-runtime -name "*.h" -o -name "*.cc" -o -name "*.c" -o -name "*.cu" -o -name "*.cuh"
    echo "I'm going to try to fix the style of these files."
    echo "If you're absolutely sure, give \"DO IT!\" as a parameter."
fi
