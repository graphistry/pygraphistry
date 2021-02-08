#!/bin/bash
set -ex

WITH_SUDO=${WITH_SUDO:-sudo}

${WITH_SUDO} docker-compose -f neo4j4.yml down -v || exit 1
${WITH_SUDO} docker-compose -f neo4j4.yml up -d || exit 1
${WITH_SUDO} ./wait_neo4j.sh
NEO4J_PORT=10006 ./add_data.sh || exit 1