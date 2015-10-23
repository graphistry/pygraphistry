#### RUN FROM DISTRIBUTION ROOT FOLDER

LOGS=build

echo "====== CLEAN ======"
rm -rf $LOGS 
mkdir $LOGS 
rm -rf bin

touch $LOGS/central.error
touch $LOGS/central.log
touch $LOGS/worker.error
touch $LOGS/worker.log

echo "====== PREBUILD NATIVES ======"
cd natives

npm install --prefix . -only=prod node_modules/segfault-handler
cd node_modules/segfault-handler
npm link
cd ../..

npm install --prefix . -only=prod node_modules/node-opencl
cd node_modules/node-opencl
npm link
cd ..

npm install --prefix . -only=prod node_modules/node-pigz
cd node_modules/node-pigz
npm link 
cd ..

cd ..


echo "====== CENTRAL ======"
cd central
npm install
npm link node-pigz
npm link segfault-handler
npm link node-opencl
#npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz 2> ../$LOGS/central.error 1> ../$LOGS/central.log 
cd ..

echo "====== WORKER ======"
cd viz-server
npm install
npm link node-pigz
npm link segfault-handler
npm link node-opencl
#npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz 2> ../$LOGS/worker.error 1> ../$LOGS/worker.log 
cd ..

echo "===== BIN ======"
mkdir bin
cp deploy/duck-tape-prod.sh bin/server.sh 

echo "===== DONE ======"
echo "Run bin/server.sh to start"
