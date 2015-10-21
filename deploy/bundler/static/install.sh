#### RUN FROM DISTRIBUTION ROOT FOLDER

LOGS=build

echo "====== CLEAN ======"
rm -rf $LOGS 
mkdir $LOGS 
rm -rf bin

touch $LOGS/central.error
touch $LOGS/central.log


echo "====== CENTRAL ======"
cd central
npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz 2> ../$LOGS/central.error 1> ../$LOGS/central.log 
cd ..

echo "====== WORKER ======"
cd viz-server
npm install node_modules/segfault-handler node_modules/node-opencl node_modules/node-pigz 2> ../$LOGS/central.error 1> ../$LOGS/central.log 
cd ..

echo "===== BIN ======"
mkdir bin
cp deploy/duck-tape-prod.sh bin/server.sh 

echo "===== DONE ======"
echo "Run bin/server.sh to start"
