#!/bin/bash -ex

## 2 ETLs, to $1.

AGENT="$(basename $0)"
WHOAMI=smoketest001@graphistry.com
SEEKRIT=Validated
APIKEY=$(curl -L --silent $1'/api/encrypt?text='${WHOAMI}${SEEKRIT} | awk -F '"' '{print $(NF-1)}')

echo " >> DEBUG: APIKEY == ${APIKEY}"

DATASETNAME=$(head -c 16 < /dev/urandom | od -x -A n | tr -d ' ')
DATASET='{"name":"'${DATASETNAME}'","graph":[{"s":"a","d":"b"},{"s":"b","d":"c"}],"bindings":{"sourceField":"s","destinationField":"d"}}'

for i in {1..2} ; do
    echo "${1}/graph/graph.html?dataset=${DATASETNAME}&viztoken=$(
       curl -L --verbose -X POST -H "Content-Type: application/json" --data $DATASET "${1}/etl?apiversion=1&agent=${AGENT}&key=${APIKEY}" | awk -F '"' '{print $(NF-1)}'
    )" ; done

echo Check out the above link to see the result of a successful ETL.

