#!/bin/bash
mpiexec -n 1 build/mpitest 8 8 8 100 30
mpitest_failed=$?
./build/devicetest 8 8 8 100 10
devicetest_failed=$?
exit $(( devicetest_failed > mpitest_failed ? devicetest_failed : mpitest_failed ))

