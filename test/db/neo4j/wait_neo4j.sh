#!/bin/bash

export CONTAINER=${CONTAINER:-neo4j4-test}
export NEO4J_USERNAME=${NEO4J_USERNAME:-neo4j}
export NEO4J_PASSWORD=${NEO4J_PASSWORD:-test}
export NEO4J_PORT=${NEO4J_PORT:-7687}

if [[ -z ${CONTAINER} ]]
  then
     echo "Usage:"
     echo "  wait4bolt_outside docker_container"
     echo "  e.g. wait4bolt_outside neo_ag"
     exit 1
fi

#PORT=$(docker exec -t ${CONTAINER} bash -c 'echo $NEO4J_dbms_connector_bolt_advertised__address'
# |cut -d : -f 2)
#echo "neo4j bolt is on port ${NEO4J_PORT}"
#if [[ -z ${NEO4J_PORT} ]]
#  then
#     echo "Is ${CONTAINER} running neo4j?  I don't see it..."
#     echo 
#     echo "Usage:"
#     echo "  wait4bolt_outside docker_container"
#     echo "  e.g. wait4bolt_outside neo_ag"
#     exit 1
#fi

echo "wait for neo4j bolt to respond at port ${NEO4J_PORT}"

# this returns before server is ready
#    curl -i http://127.0.0.1:${PORT} 2>&1 | grep -c -e '200 OK' || break 

# try an actual query as test?
docker exec -e NEO4J_USERNAME -e NEO4J_PASSWORD -t ${CONTAINER} \
  bash -c "until echo 'match (n) return count(n);' | bin/cypher-shell -a bolt://localhost:${NEO4J_PORT}; do echo $? ; sleep 1; done"

echo 'neo4j online!'

exit 0