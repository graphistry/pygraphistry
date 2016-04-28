FROM nginx:alpine
COPY nginx.conf /etc/nginx/
COPY server.conf upstreams-central.conf upstreams-workers.conf routes-workers.conf ssl.conf /etc/nginx/graphistry/
# mount certs at /etc/nginx/graphistry/ssl/
