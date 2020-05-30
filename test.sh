#!/bin/bash


echo "=========== TEST DATA =========="
echo "PWD: `pwd`"
ls

source activate rapids
conda-env list

echo "install mock"
conda install --no-deps mock -y
#conda install --no-deps -c conda-forge python-igraph textable -y

echo "=========== TEST ==============="
echo "Test args: $@"
GRAPHISTRY_API_KEY="" python -B -O -m pytest -v  graphistry/tests $@
#python -B setup.py test $@
