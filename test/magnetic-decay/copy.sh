#!/bin/bash
rm -rf DSL
rm -f PC_moduleflags.h
rm -f ../*.h
cp ~/pencil-code/pencil-private/projects/PC-A/1-gpu/src/*.h ../
cp ~/pencil-code/pencil-private/projects/PC-A/1-gpu/src/astaroth/PC_moduleflags.h .
cp -r ~/pencil-code/pencil-private/projects/PC-A/1-gpu/src/astaroth/DSL DSL
cp ~/pencil-code/pencil-private/projects/PC-A/1-gpu/PC-AC.conf ../..
