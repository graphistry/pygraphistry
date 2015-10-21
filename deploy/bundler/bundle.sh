#!/bin/bash


echo "Need node 10.30-40 and npm@3"
node -v
npm -v

#### Creates ~air-gapped dev.tar.gz (/central, /viz-server, README.md, install.sh)
#### user must still install pigz, opencl


OUT="dist"
ARCHIVE="bundle.tar.gz"
LOGS="build"

###
echo "====== CLEAN ======"
rm -rf $OUT 
rm -f $ARCHIVE
rm -rf $LOGS

###
echo "===== INIT LOGS ======"
mkdir -p $LOGS
touch $LOGS/git.central.error
touch $LOGS/git.central.log
touch $LOGS/git.viz.error
touch $LOGS/git.viz.log
touch $LOGS/git.streamgl.error
touch $LOGS/git.streamgl.log
touch $LOGS/central.error
touch $LOGS/central.log
touch $LOGS/central.helpers.error
touch $LOGS/central.helpers.log
touch $LOGS/streamgl.error
touch $LOGS/streamgl.log
touch $LOGS/central.streamgl.error
touch $LOGS/central.streamgl.log
touch $LOGS/viz.helpers.error
touch $LOGS/viz.helpers.log
touch $LOGS/viz.error
touch $LOGS/viz.log

### 
echo "====== DOWNLOAD SOURCES ======"
mkdir $OUT 
git clone https://github.com/graphistry/central.git $OUT/central 2> $LOGS/git.central.error 1> $LOGS/git.central.log
git clone https://github.com/graphistry/viz-server.git $OUT/viz-server 2> $LOGS/git.viz.error 1> $LOGS/git.viz.log
git clone https://github.com/graphistry/StreamGl.git $OUT/StreamGL 2> $LOGS/git.streamgl.error 1> $LOGS/git.streamgl.log

###
echo "===== INSTALL STREAMGL"
cd $OUT/StreamGL
npm install 2> ../../$LOGS/streamgl.error 1> ../../$LOGS/streamgl.log
cd ../..

###
echo "====== INSTALL CENTRAL ======"
cd $OUT/central
npm install 2> ../../$LOGS/central.error 1> ../../$LOGS/central.log
npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz 2> ../../$LOGS/central.helpers.error 1> ../../$LOGS/central.helpers.log
echo "------ COPY STREAMGL ------"
cp -r ../StreamGL/ node_modules/StreamGL
cd ../..

###
echo "====== INSTALL VIZ-SERVER ======"
cd $OUT/viz-server
npm install 2> ../../$LOGS/viz.error 1> ../../$LOGS/viz.log
npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz 2> ../../$LOGS/viz.helpers.error 1> ../../$LOGS/viz.helpers.log
echo "------ COPY STREAMGL ------"
cp -r ../StreamGL/ node_modules/StreamGL
cd ../..

###
echo "====== COPY STATIC SOURCES ======"
cp static/* $OUT 

###
echo "====== BUNDLE ======"
tar -cvzf $ARCHIVE $OUT

echo "Created $ARCHIVE"




