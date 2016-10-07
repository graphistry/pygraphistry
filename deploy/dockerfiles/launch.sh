#!/bin/bash -xe

### 1. Ensure that we can get an OpenCL context.

nvidia-docker run --rm --name graphistry_httpd_test graphistry/central-and-vizservers:$1 clinfo

### 2. Stop app, make log directories, start app.

docker rm -f -v graphistry_httpd || true

mkdir -p central-app worker graphistry-json clients reaper

nvidia-docker run --net host --restart=unless-stopped --name graphistry_httpd -e "GRAPHISTRY_LOG_LEVEL=${GRAPHISTRY_LOG_LEVEL:-INFO}" -d -v `pwd`/central-app:/var/log/central-app -v `pwd`/worker:/var/log/worker -v `pwd`/graphistry-json:/var/log/graphistry-json -v `pwd`/clients:/var/log/clients -v `pwd`/reaper:/var/log/reaper -v ${GRAPHISTRY_DATA_CACHE:-`pwd`/data_cache}:/tmp/graphistry/data_cache -v `pwd`/supervisor:/var/log/supervisor graphistry/central-and-vizservers:$1

### 3. Nginx, maybe with ssl.

docker rm -f -v graphistry_nginx || true

if [ -n "$SSLPATH" ] ; then
    docker run --net host --restart=unless-stopped --name graphistry_nginx -d -v ${SSLPATH}:/etc/graphistry/ssl:ro graphistry/nginx-central-vizservers:1.1.0.32
else
    docker run --net host --restart=unless-stopped --name graphistry_nginx -d graphistry/nginx-central-vizservers:1.1.0.32.httponly
fi

### 4. Cluster membership.

docker rm -f -v graphistry_mongo || true
docker run --net host --restart=unless-stopped --name graphistry_mongo -d mongo:2

for i in {1..5} ; do echo $i; docker exec graphistry_mongo mongo --eval "2+2" || sleep 1 ; done
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
if (docker inspect pg) ; then
  DB_BU=$(tempfile)
  DB_LD=$(tempfile)
  docker network disconnect host pg
  docker network connect none pg
  docker exec pg pg_dumpall -Upostgres > $DB_BU
  echo 'psql -Upostgres < /tmp/backup.sql' > $DB_LD
  docker rm -f -v pg
  docker run -d --restart=unless-stopped --net none --name pg -e POSTGRES_USER=${PG_USER} -e POSTGRES_PASSWORD=${PG_PASS} -v ${DB_BU}:/tmp/backup.sql:ro -v ${DB_LD}:/docker-entrypoint-initdb.d/backup.sh:ro postgres:9.5
else
  mkdir -p DB_BACKUP_DIRECTORY
  docker run -d --restart=unless-stopped --net none --name pg -e POSTGRES_USER=${PG_USER} -e POSTGRES_PASSWORD=${PG_PASS} postgres:9.5
fi
# spinlock until pg starts up?
docker network disconnect none pg
docker network connect host pg

### Done.

echo SUCCESS.
echo Graphistry has been launched, and should be up and running.
echo SUCCESS.
