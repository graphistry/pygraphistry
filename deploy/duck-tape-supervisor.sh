#!/bin/bash

cd ../viz-server

while :; do
    node --max-old-space-size=4096 server.js "${2}" | node ../common/oneLineLogs.js &> worker${1}.log
    if [[ ${PIPESTATUS[0]} == 42 ]] ; then 
        sleep 1
    else 
        break
    fi
done
