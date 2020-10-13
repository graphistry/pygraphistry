#!/bin/bash
set -ex

sudo docker-compose -f neo4j4.yml down -v || exit 1
sudo docker-compose -f neo4j4.yml up -d || exit 1
sudo ./wait_neo4j.sh
sudo NEO4J_PORT=10006 ./add_data.sh || exit 1