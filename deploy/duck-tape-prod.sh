#!/bin/bash

# Start Central and MAX workers using the local dev setup

MAX=10
WORKERS=`seq -separator=, 10000 $((10000 + $MAX))`

CENTRAL_CONFIG="{\"HTTP_LISTEN_ADDRESS\": \"0.0.0.0\", \"VIZ_LISTEN_PORTS\": [$WORKERS]}"
declare -a WORKER_CONFIG
for ID in `seq 1 $MAX`; do
 	WORKER_CONFIG[$ID]="{\"VIZ_LISTEN_ADDRESS\": \"0.0.0.0\", \"VIZ_LISTEN_PORT\": $((10000+ID))}"
done

echo $CENTRAL_CONFIG
for i in "${WORKER_CONFIG[@]}"; do
	echo $i
done

killall node

cd ..
pushd . > /dev/null
cd central
npm start "$CENTRAL_CONFIG" | node ../common/oneLineLogs.js &> central.log &
disown
popd > /dev/null

pushd . > /dev/null
cd viz-server
for ID in `seq 1 $MAX`; do
	npm start "${WORKER_CONFIG[$ID]}" | node ../common/oneLineLogs.js &> worker${ID}.log &
	disown
done
popd > /dev/null
