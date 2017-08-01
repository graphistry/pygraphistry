#!/bin/bash

echo Making api key for "<<$1>>" on local Graphistry app...
docker exec ${GRAPHISTRY_NETWORK:-monolith-network}-nginx wget -q -O - 'http://vizapp:3000/api/internal/provision?text='"$1" ; echo
