#!/bin/bash

export AC_HOME=$PWD
export PATH=${PATH}:$AC_HOME/scripts/
export PATH=${PATH}:$AC_HOME/bin
echo $AC_HOME
echo $PATH
#TP: we don't want to return a non-zero error code in case pulling submodules fails since it is not fatal
git submodule update --init --remote > /dev/null 2>&1 || true
