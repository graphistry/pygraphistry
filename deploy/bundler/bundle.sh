#!/bin/bash


echo "Need node 10.30-40 and npm@3"
node -v
npm -v

#### Creates ~air-gapped dev.tar.gz (/central, /viz-server, README.md, install.sh)
####   User must still install pigz, opencl
####   Keeps on side for install linker: StreamGL, config-public, natives (node-opencl, node-pigz, segfault-handler)
####   Deletes heavy files (uber TSVs, ...)

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
echo "====== INIT LOGS ======"
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
touch $LOGS/git.pigz.error
touch $LOGS/git.pigz.error
touch $LOGS/git.cl.log
touch $LOGS/git.cl.log
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
touch $LOGS/natives.seg.error
touch $LOGS/natives.seg.log
touch $LOGS/natives.cl.error
touch $LOGS/natives.cl.log
touch $LOGS/natives.pigz.error
touch $LOGS/natives.pigz.log

### 
echo "====== DOWNLOAD SOURCES ======"
mkdir -p $OUT 

#primary
git clone https://github.com/graphistry/central.git $OUT/central 2> $LOGS/git.central.error 1> $LOGS/git.central.log
git clone https://github.com/graphistry/viz-server.git $OUT/viz-server 2> $LOGS/git.viz.error 1> $LOGS/git.viz.log

#weird
git clone https://github.com/graphistry/StreamGl.git $OUT/StreamGL 2> $LOGS/git.streamgl.error 1> $LOGS/git.streamgl.log
git clone https://github.com/graphistry/config-public.git $OUT/config-public 2> $LOGS/git.config.error 1> $LOGS/git.config.log
git clone https://github.com/graphistry/common.git $OUT/common 2> $LOGS/git.common.error 1> $LOGS/git.common.log
git clone https://github.com/graphistry/node-pigz.git $OUT/node-pigz 2> $LOGS/git.pigz.error 1> $LOGS/git.pigz.log

#external
git clone https://github.com/mikeseven/node-opencl.git $OUT/node-opencl 2> $LOGS/git.cl.error 1> $LOGS/git.cl.log
git clone https://github.com/ddopson/node-segfault-handler.git $OUT/node-segfault-handler 2> $LOGS/git.seg.error 1> $LOGS/git.seg.log

###
echo "====== ISOLATE AND INSTALL NATIVE DEPS ======"

cd $OUT/node-segfault-handler
npm install -only=prod 2> ../../$LOGS/natives.seg.error 1> ../../$LOGS/natives.seg.log
cd ../..

cd $OUT/node-opencl
npm install -only=prod 2> ../../$LOGS/natives.cl.error 1> ../../$LOGS/natives.cl.log 
cd ../..

cd $OUT/node-pigz
npm install -only=prod 2> ../../$LOGS/natives.pigz.error 1> ../../$LOGS/natives.pigz.log
cd ../..

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
cd ../..

###
echo "====== INSTALL CENTRAL CSS ======
cd $OUT/central/node_modules/graph-viz
npm run less
cd ../..


###
echo "====== INSTALL VIZ-SERVER ======"
cd $OUT/viz-server
npm install 2> ../../$LOGS/viz.error 1> ../../$LOGS/viz.log
cd ../..


###
echo "====== DELETE PRIVATE ======"
rm -rf $OUT/viz-server/node_modules/config
rm -rf $OUT/central/node_modules/config

###
echo "====== CLEANUP ======"


###
echo "===== TRIM ========="
rm -rf $OUT/central/node_modules/graph-viz/assets/libs/ace/src
rm -rf $OUT/central/node_modules/graph-viz/assets/libs/ace/src-min
rm -rf $OUT/central/node_modules/graph-viz/assets/libs/ace/src-min-noconflict
rm -rf $OUT/central/node_modules/node-opencl/build
rm -rf $OUT/central/node_modules/uber-viz/screenshots
rm -rf $OUT/central/node_modules/uber-viz/uber_gps_tsv
rm -rf $OUT/central/node_modules/StreamGL/node_modules
rm -rf $OUT/central/.git
rm -rf $OUT/viz-server/node_modules/graph-viz/assets/libs/ace/src
rm -rf $OUT/viz-server/node_modules/graph-viz/assets/libs/ace/src-min
rm -rf $OUT/viz-server/node_modules/graph-viz/assets/libs/ace/src-min-noconflict
rm -rf $OUT/viz-server/node_modules/node-opencl/build
rm -rf $OUT/viz-server/node_modules/uber-viz/screenshots
rm -rf $OUT/viz-server/node_modules/uber-viz/uber_gps_tsv
rm -rf $OUT/viz-server/node_modules/StreamGL/node_modules
rm -rf $OUT/viz-server/.git
rm -rf $OUT/common/.git
rm -rf $OUT/StreamGL/.git
rm -rf $OUT/config-public/.git

###
echo "====== COPY STATIC SOURCES ======"
cp static/* $OUT 

###
echo "====== BUNDLE ======"
tar -cvzf $ARCHIVE $OUT

echo "Created $ARCHIVE"




