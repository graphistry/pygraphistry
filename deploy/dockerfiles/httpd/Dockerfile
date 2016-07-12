FROM graphistry/js-and-gpu:0.10.45
ENV graphistry_install_path "/var/graphistry"
ENV graphistry_temp_path "/tmp/graphistry"
ENV workers_count 2
ENV workers_port_start 10000
ENV upload_limit "128M"
ENV GRAPHISTRY_LOG_LEVEL INFO

# common & nodejs & app server.

RUN mkdir -p /var/log/graphistry-json $graphistry_temp_path $graphistry_install_path
RUN apt-get install -y supervisor libkrb5-dev gettext pigz
COPY logrotate.conf /etc/logrotate.d/graphistry.conf

# logs.

RUN mkdir -p /var/log/central-app /var/log/clients /var/log/worker /var/log/graphistry-json /var/log/reaper /root/.ssh

# fetch secrets from our vault, do the npm installs, and delete the secrets.

RUN curl $(/sbin/ip route | awk '/default/ {print $3}'):17290/github/id_rsa > /root/.ssh/id_rsa && \
    curl $(/sbin/ip route | awk '/default/ {print $3}'):17290/github/id_rsa.pub > /root/.ssh/id_rsa.pub && \
    chmod 400 /root/.ssh/* && \
    md5sum /root/.ssh/* && \
    (ssh -o StrictHostKeyChecking=no -T git@github.com || true) && \
    cd $graphistry_install_path && \
    git clone git@github.com:graphistry/central.git && \
    git clone git@github.com:graphistry/viz-server.git && \
    git clone git@github.com:graphistry/datasets.git && \
    mv datasets/preloaded/* central/assets/ && \
    rm -rf datasets && \
    rm -rf central/.git viz-server/.git && \
    cd central && bash -l -c "npm install --unsafe-perm" && cd .. && \
    cd viz-server && bash -l -c "npm install --unsafe-perm" && cd .. && \
    rm -r /tmp/* && \
    shred -u /root/.ssh/*

COPY reaper.py $graphistry_install_path
RUN apt-get install -y python-pip && pip install pymongo

# supervisor.

COPY central.conf viz_workers.conf reaper.conf /etc/supervisor/conf.d/
COPY supervisord.conf /etc/supervisor
CMD bash -l -c "supervisord -n -c /etc/supervisor/supervisord.conf"
