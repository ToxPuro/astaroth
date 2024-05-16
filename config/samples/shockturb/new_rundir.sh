#!/bin/bash
#
# Adapted from pc_newrun in the PENCIL CODE
# https://github.com/pencil-code
#
# Usage:
#   ./new_rundir.sh run1 run2
# Alternativ eusage if one is already in run1
#   ./new_rundir.sh run2
#

case "$1" in
    -h |--help)
    echo "Usage: ./new_rundir.sh [-s]"
    exit;;

    -s|--same_source)
    echo "Using the same directory"
    same_source=1
    shift;;
esac

if [ $# -eq 2 ]; then
  #
  #  check whether absolute or relative path
  #  by checking whether target exists
  #
  if [ -d `pwd`/$1 ]; then
    olddir=`pwd`/$1
  else
    olddir=$1
  fi
  #
  #  same for newdir
  #
  if [ -d `pwd`/$2 ]; then
    newdir=`pwd`/$2
  else
    newdir=$2
  fi
else
  olddir=`pwd`
  cd ..
  #
  #  same for newdir
  #
  if [ -d `pwd`/$1 ]; then
    newdir=`pwd`/$1
  else
    newdir=$1
  fi
fi
#
#  save current working directory and make new run directory and go there.
#
parentdir=`pwd`
echo "Parent directory: " $parentdir
if [ -d $newdir ]; then
  echo "The directory $newdir already exists!"
  echo "You had better stop and check..."
  exit
else
  mkdir $newdir
  echo "Created directory $newdir"
fi
cd $newdir
targetdir=`pwd`
#
#  go back into source directory and write the name
#  of the new target directory into file new_to.dir
#
cd $olddir
#
#  copy setup files required into new directory
#
rsync -avu *.sh $targetdir
rsync -avu moduleinfo* $targetdir
cp -vr a2_timeseries.ts $targetdir
cp -vr astaroth.conf $targetdir
cp -vr README.md $targetdir
