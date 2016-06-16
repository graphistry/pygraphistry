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
# docker run -e "SPLUNK_START_ARGS=--accept-license" -e "SPLUNK_CMD_1='edit user admin -password $SPLUNK_PASSWORD -auth admin:changeme --answer-yes'" -v `pwd`/inputs.conf:/opt/splunk/etc/system/local/inputs.conf -v `pwd`/server.conf:/opt/splunk/etc/system/local/server.conf -v `pwd`/deploymentclient.conf:/opt/splunk/etc/system/local/deploymentclient.conf -e "SPLUNK_FORWARD_SERVER=splunk.graphistry.com:9997" -e "SPLUNK_FORWARD_SERVER_1=splunk.graphistry.com:9997" -e SPLUNK_DEPLOYMENT_SERVER="splunk.graphistry.com:8089" outcoldman/splunk:forwarder
# docker run --net host --restart=unless-stopped graphistry_splunk -d -v `pwd`/central-app:/var/log/central-app -v `pwd`/worker:/var/log/worker -v `pwd`/graphistry-json:/var/log/graphistry-json -v `pwd`/clients:/var/log/clients -v `pwd`/reaper:/var/log/reaper graphistry/log-shipper:1.0.0

echo SUCCESS.
echo Graphistry has been launched, and should be up and running.
echo SUCCESS.
