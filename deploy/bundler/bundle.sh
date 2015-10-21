#!/bin/bash

#### Creates ~air-gapped dev.tar.gz (/central, /viz-server, README.md, install.sh)
#### user must still install pigz, opencl


OUT="dist"
ARCHIVE="bundle.tar.gz"

###
echo "====== CLEAN ======"
rm -rf $OUT 
rm -f $ARCHIVE

### 
echo "====== DOWNLOAD SOURCES ======"
mkdir $OUT 
git clone https://github.com/graphistry/central.git $OUT/central
git clone https://github.com/graphistry/viz-server.git $OUT/viz-server

###
echo "====== INSTALL CENTRAL ======"
cd $OUT/central
npm install
npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz
echo "------ (BUILD STREAMGL) ------"
npm install StreamGL
cd ../..

###
echo "====== INSTALL VIZ-SERVER ======"
cd $OUT/viz-server
npm install
npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz
echo "------ (COPY STREAMGL FROM CENTRAL) ------"
cp -r ../central/node_modules/StreamGL node_modules/StreamGL
cd ../..

###
echo "====== COPY STATIC SOURCES ======"
cp static/* $OUT 

###
echo "====== BUNDLE ======"
tar -cvzf $ARCHIVE $OUT

echo "Created $ARCHIVE"


