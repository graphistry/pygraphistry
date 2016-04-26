FROM graphistry/js-and-gpu:0.10.44
ENV graphistry_install_path "/var/graphistry"
ENV graphistry_temp_path "/tmp/graphistry"
ENV workers_count 2
ENV workers_port_start 10000
ENV upload_limit "128M"

# common & nodejs & app server.

RUN mkdir -p /var/log/graphistry-json $graphistry_temp_path $graphistry_install_path
RUN apt-get install -y supervisor libkrb5-dev gettext pigz
#FIXME COPY lol-graphistry-json.logrotate.conf /etc/logrotate.d/graphistry-json // app server

# logs.

RUN mkdir -p /var/log/central-app /var/log/clients /var/log/worker /var/log/graphistry-json

# secrets.

COPY gh-id-rsa     /root/.ssh/id_rsa
COPY gh-id-rsa-pub /root/.ssh/id_rsa.pub
RUN chmod 400 /root/.ssh/*
RUN ssh -o StrictHostKeyChecking=no -T git@github.com || true

# central.

WORKDIR /var/graphistry
RUN git clone git@github.com:graphistry/central.git
WORKDIR central
RUN bash -l -c "npm install --unsafe-perm" # as root, we do an unsafe perm; npm usually installs as nobody

# worker[s].

WORKDIR /var/graphistry
RUN git clone git@github.com:graphistry/viz-server.git
WORKDIR viz-server
COPY hot-patch-pings.js src/pings.js
RUN bash -l -c "npm install --unsafe-perm"

# reaper.

COPY reaper.py /var/graphistry/
RUN apt-get install -y python-pip && pip install pymongo

# supervisor.

COPY central.conf /etc/supervisor/conf.d/
COPY viz_workers.conf /etc/supervisor/conf.d/
COPY reaper.conf /etc/supervisor/conf.d/
COPY supervisord.conf /etc/supervisor
CMD bash -l -c "supervisord -n -c /etc/supervisor/supervisord.conf"
