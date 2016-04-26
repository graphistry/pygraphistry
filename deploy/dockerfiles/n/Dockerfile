FROM nginx:alpine
COPY nginx.conf /etc/nginx/
COPY server.conf /etc/nginx/graphistry/
COPY upstreams-central.conf /etc/nginx/graphistry/
COPY upstreams-workers.conf /etc/nginx/graphistry/
RUN touch /etc/nginx/graphistry/ssl.conf
COPY routes-workers.conf /etc/nginx/graphistry/
