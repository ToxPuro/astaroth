#!/bin/bash

shopt -s extglob
rm -vr !("astaroth.conf"|"dslsource"|"clean_rundir.sh"|"my_cmake.sh"|"README.md"|"a2_timeseries.ts"|"run_on_tiara.sh"|"add_to_pythonpath.sh"|"check_run_properties.py")

