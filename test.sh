#!/bin/bash


echo "=========== TEST DATA =========="
echo "PWD: `pwd`"
ls

source activate rapids
conda-env list

echo "=========== TEST ==============="
echo "Test args: $@"
GRAPHISTRY_API_KEY="" python -B -O -m pytest -v  graphistry/tests $@
#python -B setup.py test $@
