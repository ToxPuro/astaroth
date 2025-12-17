#!/bin/bash

export AC_HOME=$PWD
export PATH=${PATH}:$AC_HOME/scripts/
export PATH=${PATH}:$AC_HOME/bin
echo $AC_HOME
echo $PATH
fetched_submodule=$(git submodule update --init --remote > /dev/null 2>&1)
