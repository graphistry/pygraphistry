#!/bin/bash
set -ex

EXCLUDES="../setup.py ../graphistry/util.py"
echo "SKIPPING " $EXCLUDES
sphinx-apidoc -o source .. $EXCLUDES

make clean
make html SPHINXOPTS="-W --keep-going -n"
make latexpdf SPHINXOPTS="-W --keep-going -n"
make epub SPHINXOPTS="-W --keep-going -n"

PLATFORM=`uname`
if [[ "$PLATFORM" == "Darwin" ]]
then
    echo "Opening in Darwin.."
    open build/html/index.html
fi
