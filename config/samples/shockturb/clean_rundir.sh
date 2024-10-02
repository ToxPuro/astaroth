#!/bin/bash

shopt -s extglob
rm -vr !("astaroth.conf"|"new_rundir.sh"|"clean_rundir.sh"|"my_cmake.sh"|"README.md"|"a2_timeseries.ts"|"run_on_"*[a-z]".sh"|"moduleinfo_"*[a-z])
