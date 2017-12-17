#!/bin/sh
rm core
ulimit -c unlimited
make -C ../
cp ../gjk_proc ./
./gjk_proc 4 0