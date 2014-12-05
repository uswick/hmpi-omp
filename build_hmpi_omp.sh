#!/bin/sh 

echo "building hmpi - omp"

cd /home1/02636/uswickra/Sync/HMPI/hmpi-omp/hmpi-svn
make -f Makefile.mic

echo "Done !!"
