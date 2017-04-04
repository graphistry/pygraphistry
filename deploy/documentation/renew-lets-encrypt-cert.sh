#!/bin/bash -lex

# For a particular $HOSTNAME:
# stop the current $NGINX container,
# run certbot-standalone,
# move the certs into place,
# and restart the $NGINX container.
# This should run every three months.

OUR_NGINX=${NGINX:-monolith-network-nginx}
OUR_DOMAIN=${DOMAIN:-labs.graphistry.com}

docker stop $OUR_NGINX || true
docker run --rm -p 80:80 -p 443:443 -e DOMAIN=$OUR_DOMAIN -v $PWD/.le:/etc/letsencrypt alpine sh -c 'apk add --no-cache certbot && certbot certonly --agree-tos --email lsb@graphistry.com --standalone -t -n -d $DOMAIN && cd /etc/letsencrypt/live && (stat main || ln -s * main)'
sudo cp .le/live/main/fullchain.pem /etc/graphistry/ssl/ssl_certificate.pem
sudo cp .le/live/main/fullchain.pem /etc/graphistry/ssl/ssl_trusted_certificate.pem
sudo cp .le/live/main/privkey.pem   /etc/graphistry/ssl/ssl_certificate_key.pem
docker restart $OUR_NGINX || true
