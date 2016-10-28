#!/bin/bash -e

## 2 ETLs, to $1.

WHOAMI=smoketest001@graphistry.com
SEEKRIT=Validated
APIKEY=$(curl -L --silent $1'/api/encrypt?text='${WHOAMI}${SEEKRIT} | awk -F '"' '{print $(NF-1)}')
DATASETNAME=`LC_CTYPE=C tr -dc A-Za-z0-9 < /dev/urandom | fold -w 32 | head -n 1`

echo " >> DEBUG: APIKEY == ${APIKEY}"

for i in {1..2} ; do echo -n ${1}'/graph/graph.html?dataset='${DATASETNAME}'&viztoken=' ; curl -L --silent -X POST -H "Content-Type: application/json" --data '{"name":"'${DATASETNAME}'","graph":[{"s":"a","d":"b"},{"s":"b","d":"c"}],"bindings":{"sourceField":"s","destinationField":"d"}}' ${1}'/etl?key='${APIKEY} | awk -F '"' '{print $(NF-1)}' ; done

echo Check out the above link to see the result of a successful ETL.

