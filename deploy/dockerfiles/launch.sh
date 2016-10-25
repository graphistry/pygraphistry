#!/bin/bash -lex

### 0. Ensure that we can get an OpenCL context.

nvidia-docker run --rm --name graphistry_httpd_test graphistry/central-and-vizservers:$1 clinfo

### 1. Ensure we have a network for our application to run in.

GRAPHISTRY_NETWORK=${GRAPHISTRY_NETWORK:-monolith-network}
docker network inspect $GRAPHISTRY_NETWORK || docker network create $GRAPHISTRY_NETWORK

### 2. Stop app, make log directories, start app.

docker rm -f -v graphistry_httpd || true

mkdir -p central-app worker graphistry-json clients reaper

stat ../httpd-config.json || (echo '{}' > ../httpd-config.json)
echo "${GRAPHISTRY_APP_CONFIG:-{\}}" > ./httpd-config.json

CENTRAL_MERGED_CONFIG=$(docker   run --rm -v `pwd`/../httpd-config.json:/tmp/box-config.json -v `pwd`/httpd-config.json:/tmp/local-config.json graphistry/central-and-vizservers:$1 bash -c 'mergeThreeFiles.js $graphistry_install_path/central-cloud-options.json    /tmp/box-config.json /tmp/local-config.json')
VIZWORKER_MERGED_CONFIG=$(docker run --rm -v `pwd`/../httpd-config.json:/tmp/box-config.json -v `pwd`/httpd-config.json:/tmp/local-config.json graphistry/central-and-vizservers:$1 bash -c 'mergeThreeFiles.js $graphistry_install_path/viz-worker-cloud-options.json /tmp/box-config.json /tmp/local-config.json')

nvidia-docker run --net host --restart=unless-stopped --name graphistry_httpd -e "GRAPHISTRY_CENTRAL_CONFIG=${CENTRAL_MERGED_CONFIG}" -e "GRAPHISTRY_VIZWORKER_CONFIG=${VIZWORKER_MERGED_CONFIG}" -d -v `pwd`/central-app:/var/log/central-app -v `pwd`/worker:/var/log/worker -v `pwd`/graphistry-json:/var/log/graphistry-json -v `pwd`/clients:/var/log/clients -v `pwd`/reaper:/var/log/reaper -v ${GRAPHISTRY_DATA_CACHE:-`pwd`/data_cache}:/tmp/graphistry/data_cache -v ${GRAPHISTRY_WORKBOOK_CACHE:-`pwd`/workbook_cache}:/tmp/graphistry/workbook_cache -v `pwd`/supervisor:/var/log/supervisor graphistry/central-and-vizservers:$1

### 3. Nginx, maybe with ssl.

docker rm -f -v graphistry_nginx || true

if [ -n "$SSLPATH" ] ; then
    docker run --net host --restart=unless-stopped --name graphistry_nginx -d -v ${SSLPATH}:/etc/graphistry/ssl:ro graphistry/nginx-central-vizservers:1.2.0.32
else
    docker run --net host --restart=unless-stopped --name graphistry_nginx -d graphistry/nginx-central-vizservers:1.2.0.32.httponly
fi

### 4. Cluster membership.

docker rm -f -v graphistry_mongo || true
docker run --net host --restart=unless-stopped --name graphistry_mongo -d mongo:2

for i in {1..10} ; do echo $i; docker exec graphistry_mongo mongo --eval "2+2" || sleep $i ; done
MONGO_NAME=cluster
MONGO_USERNAME=graphistry
MONGO_PASSWORD=graphtheplanet
docker exec graphistry_mongo bash -c "mongo --eval '2+2' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME || (mongo --eval \"db.createUser({user: '$MONGO_USERNAME', pwd: '$MONGO_PASSWORD', roles: ['readWrite']})\" localhost/$MONGO_NAME && mongo --eval 'db.gpu_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})'  -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME && mongo --eval 'db.node_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME )"

### 5. Splunk.

docker rm -f -v graphistry_splunk || true

if [ -n "$SPLUNK_PASSWORD" ] ; then
    docker run --name graphistry_splunk --restart=unless-stopped -d -v /etc/graphistry/splunk/:/opt/splunkforwarder/etc/system/local -v `pwd`/central-app:/var/log/central-app -v `pwd`/worker:/var/log/worker -v `pwd`/graphistry-json:/var/log/graphistry-json -v `pwd`/clients:/var/log/clients -v `pwd`/reaper:/var/log/reaper -v `pwd`/supervisor:/var/log/supervisor graphistry/splunkforwarder:6.4.1 bash -c "/opt/splunkforwarder/bin/splunk edit user admin -password $SPLUNK_PASSWORD -auth admin:$SPLUNK_ADMIN --accept-license --answer-yes ; /opt/splunkforwarder/bin/splunk start --nodaemon --accept-license --answer-yes"
fi

### 6. Postgres.

DB_BACKUP_DIRECTORY=${DB_BACKUP_DIRECTORY:-../.pgbackup}
PG_USER=${PG_USER:-graphistry}
PG_PASS=${PG_PASS:-graphtheplanet}
if (docker exec pg psql -c "select 'database is up' as healthcheck" postgresql://${PG_USER}:${PG_PASS}@pg:5432) ; then
  echo Keeping db.
else
  echo Bringing up db.
  docker run -d --restart=unless-stopped --net none --name pg -e POSTGRES_USER=${PG_USER} -e POSTGRESS_PASSWORD=${PG_PASS} postgres:9.5
  if [ -z $DB_RESTORE ] ; then
    echo Nothing to restore.
  else
    echo Restoring db with the contents of ${DB_RESTORE}.
    for i in {1..100} ; do docker exec pg psql -U${PG_USER} -c "select 'database is up' as healthcheck" || sleep 5 ; done
    docker exec -i pg psql -U${PG_USER} < ${DB_RESTORE}
  fi
  docker network disconnect none pg
  docker network connect $GRAPHISTRY_NETWORK pg
fi

### Done.

docker restart graphistry_httpd # to be removed once DEV-898 is fixed.


echo SUCCESS.
echo Graphistry has been launched, and should be up and running.
echo SUCCESS.
