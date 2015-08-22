EXCLUDES="../setup.py ../graphistry/util.py"
echo "SKIPPING " $EXCLUDES
sphinx-apidoc -o source .. $EXCLUDES
make html
open build/html/graphistry.html
