#!/bin/bash
./test/run_syntaxtest.sh
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
cd $AC_HOME/test-builds && ac_build_tests --parallel
cd $AC_HOME/test-builds && ac_run_tests
cd $AC_HOME/test && ac_build_tests 
cd $AC_HOME/test && ac_build_tests --cpu
cd $AC_HOME/test && ac_run_tests

