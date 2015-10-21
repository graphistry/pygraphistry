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
echo "====== BIN ======"
mkdir -p $OUT/deploy
cp ../duck-tape-prod.sh $OUT/deploy/duck-tape-prod.sh



###
echo "===== INIT LOGS ======"
mkdir -p $LOGS
touch $LOGS/git.central.error
touch $LOGS/git.central.log
touch $LOGS/git.viz.error
touch $LOGS/git.viz.log
touch $LOGS/git.streamgl.error
touch $LOGS/git.streamgl.log
touch $LOGS/git.config.error
touch $LOGS/git.config.log
touch $LOGS/git.common.error
touch $LOGS/git.common.log
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
touch $LOGS/common.error
touch $LOGS/common.log
touch $LOGS/common.helpers.error
touch $LOGS/common.helpers.log
touch $LOGS/config.error
touch $LOGS/config.log
touch $LOGS/graph.error
touch $LOGS/graph.log

### 
echo "====== DOWNLOAD SOURCES ======"
mkdir -p $OUT 
git clone https://github.com/graphistry/central.git $OUT/central 2> $LOGS/git.central.error 1> $LOGS/git.central.log
git clone https://github.com/graphistry/viz-server.git $OUT/viz-server 2> $LOGS/git.viz.error 1> $LOGS/git.viz.log
git clone https://github.com/graphistry/StreamGl.git $OUT/StreamGL 2> $LOGS/git.streamgl.error 1> $LOGS/git.streamgl.log
git clone https://github.com/graphistry/config-public.git $OUT/config-public 2> $LOGS/git.config.error 1> $LOGS/git.config.log
git clone https://github.com/graphistry/common.git $OUT/common 2> $LOGS/git.common.error 1> $LOGS/git.common.log

###
echo "====== INSTALL CONFIG-PUBLIC ======"
cd $OUT/config-public
npm install 2> ../../$LOGS/config.error 1> ../../$LOGS/config.log
cd ../..

###
echo "====== COMMON ======"
cd $OUT/common
npm install 2> ../../$LOGS/common.error 1> ../../$LOGS/common.log
npm install bunyan 2> ../../$LOGS/common.helpers.error 1> ../../$LOGS/common.helpers.log
echo "------ COPY CONFIG ------"
rm -rf node_modules/config
cp -r ../config-public node_modules/config
cd ../..


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
echo "------ COPY STREAMGL/CONFIG ------"
cp -r ../StreamGL/ node_modules/StreamGL
rm -rf node_modules/config
cp -r ../config-public node_modules/config
echo "------ GENERATING CSS ------"
cd node_modules/graph-viz
npm run less
cd ../../../..


###
echo "====== INSTALL VIZ-SERVER ======"
cd $OUT/viz-server
npm install 2> ../../$LOGS/viz.error 1> ../../$LOGS/viz.log
npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz 2> ../../$LOGS/viz.helpers.error 1> ../../$LOGS/viz.helpers.log
echo "------ COPY STREAMGL/CONFIG ------"
cp -r ../StreamGL/ node_modules/StreamGL
rm -rf node_modules/config
cp -r ../config-public node_modules/config
cd ../..

###
echo "====== CLEANUP ======"
rm -rf $OUT/StreamGL
rm -rf $OUT/config-public

###
echo "====== COPY STATIC SOURCES ======"
cp static/* $OUT 

###
echo "====== BUNDLE ======"
tar -cvzf $ARCHIVE $OUT

echo "Created $ARCHIVE"




