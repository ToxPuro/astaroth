#!/bin/bash

shopt -s extglob
rm -vr !("astaroth.conf"|"twcc_batch_16.sh"|"twcc_batch_32.sh"|"twcc_batch_8.sh"|"purgedir.sh"|"dslsource"|"clean_rundir.sh"|"my_cmake.sh"|"README.md"|"a2_timeseries.ts"|"run_on_tiara.sh"|"add_to_pythonpath.sh"|"check_run_properties.py")

