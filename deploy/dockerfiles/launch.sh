#!/bin/sh -xe

docker rm -f graphistry_httpd_test || true
nvidia-docker run --name graphistry_httpd_test graphistry/central-and-vizservers:$1 clinfo
docker rm graphistry_httpd_test
docker rm -f graphistry_httpd || true

mkdir -p central-app worker graphistry-json clients reaper

nvidia-docker run --net host --restart=unless-stopped --name graphistry_httpd -e "GRAPHISTRY_LOG_LEVEL=${GRAPHISTRY_LOG_LEVEL:-INFO}" -d -v `pwd`/central-app:/var/log/central-app -v `pwd`/worker:/var/log/worker -v `pwd`/graphistry-json:/var/log/graphistry-json -v `pwd`/clients:/var/log/clients -v `pwd`/reaper:/var/log/reaper graphistry/central-and-vizservers:$1


docker rm -f graphistry_nginx || true

if [ -n "$SSLPATH" ] ; then
    docker run --net host --restart=unless-stopped --name graphistry_nginx -d -v ${SSLPATH}:/etc/graphistry/ssl:ro graphistry/nginx-central-vizservers:1.0.0.32
else
    docker run --net host --restart=unless-stopped --name graphistry_nginx -d graphistry/nginx-central-vizservers:1.0.0.32.httponly
fi


docker rm -f graphistry_mongo || true

docker run --net host --restart=unless-stopped --name graphistry_mongo -d graphistry/cluster-membership:1.0


docker rm -f graphistry_splunk || true
# docker run --net host --restart=unless-stopped graphistry_splunk -d -v `pwd`/central-app:/var/log/central-app -v `pwd`/worker:/var/log/worker -v `pwd`/graphistry-json:/var/log/graphistry-json -v `pwd`/clients:/var/log/clients -v `pwd`/reaper:/var/log/reaper graphistry/log-shipper:1.0.0

echo SUCCESS.
echo Graphistry has been launched, and should be up and running.
echo SUCCESS.
