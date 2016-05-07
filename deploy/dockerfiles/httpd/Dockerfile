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

RUN mkdir -p /var/log/central-app /var/log/clients /var/log/worker /var/log/graphistry-json /root/.ssh

# fetch secrets from our vault, do the npm installs, and delete the secrets.

RUN curl $(/sbin/ip route | awk '/default/ {print $3}'):17290/github/id_rsa > /root/.ssh/id_rsa && \
    curl $(/sbin/ip route | awk '/default/ {print $3}'):17290/github/id_rsa.pub > /root/.ssh/id_rsa.pub && \
    chmod 400 /root/.ssh/* && \
    (ssh -o StrictHostKeyChecking=no -T git@github.com || true) && \
    cd $graphistry_install_path && \
    git clone git@github.com:graphistry/central.git && \
    git clone git@github.com:graphistry/viz-server.git && \
    cd central && bash -l -c "npm install --unsafe-perm" && cd .. && \
    cd viz-server && bash -l -c "npm install --unsafe-perm" && cd .. && \
    shred -u /root/.ssh/*

COPY reaper.py $graphistry_install_path
RUN apt-get install -y python-pip && pip install pymongo

# supervisor.

COPY central.conf viz_workers.conf reaper.conf /etc/supervisor/conf.d/
COPY supervisord.conf /etc/supervisor
CMD bash -l -c "supervisord -n -c /etc/supervisor/supervisord.conf"
