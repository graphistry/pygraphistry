#!/bin/bash

TMP_PATH=${TMP_PATH:-/tmp/graphistry-stubgen}

rm -rf $TMP_PATH
mkdir -p $TMP_PATH

stubgen -o $TMP_PATH -q --export-less --no-import graphistry
#find $TMP_PATH

(
# generate combined .pyi stub file for each file in graphistry
for f in `find graphistry | grep -v graphistry/tests | grep -E '*.py$' | grep -E 'chain|hop'`; do
    #stubgen -o $TMP_PATH $f

    # f2 as f except without the prefix graphistry
    f2=`echo $f | sed 's/^graphistry\///'`
    
    echo "### $f2"
    cat $TMP_PATH/${f}i \
        | grep -v -E '^import|^from|^#|^$' \
        | grep -E '(^class)|(^    )|(^def)|(^@)'
    echo
done 
) #| wc -c