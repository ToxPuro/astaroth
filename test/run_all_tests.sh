#!/bin/bash
cpu_only=OFF
parallel=--parallel
if [[ -v AC_HOME ]]; then
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            cpu_only=ON
            shift
            ;;
        --serial)
            parallel=
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

./test/run_syntaxtest.sh
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
if [[ "$cpu_only" == "OFF" ]]; then
    cd $AC_HOME/test-builds && ac_build_tests $parallel
    cd $AC_HOME/test-builds && ac_run_tests
    cd $AC_HOME/test && ac_build_tests 
else
    cd "$AC_HOME/test-builds" && ac_build_tests --cpu $parallel
    cd "$AC_HOME/test-builds" && ac_run_tests
fi
cd $AC_HOME/test && ac_build_tests --cpu
cd $AC_HOME/test && ac_run_tests

else
    echo "source Astaroth to run the auto-test!"
    echo "You can source Astaroth by running source sourceme.sh from the root of the repo."
    exit 1
fi

