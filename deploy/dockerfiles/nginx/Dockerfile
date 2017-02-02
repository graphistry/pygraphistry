FROM nginx:alpine
COPY nginx.conf /etc/nginx/
COPY server.conf upstreams-central.conf upstreams-workers.conf routes-workers.conf ssl.conf /etc/nginx/graphistry/
RUN apk add --no-cache openssl && openssl dhparam -out /etc/dhparam.pem 2048 && apk del openssl
# mount certs at /etc/nginx/graphistry/ssl/
