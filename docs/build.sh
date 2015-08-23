#!/bin/sh

EXCLUDES="../setup.py ../graphistry/util.py"
echo "SKIPPING " $EXCLUDES
sphinx-apidoc -o source .. $EXCLUDES
make html

PLATFORM=`uname`
if [ "$PLATFORM" == "Darwin" ]; then
    open build/html/graphistry.html
fi
