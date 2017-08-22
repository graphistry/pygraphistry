FROM graphistry/baseapp
ENV graphistry_install_path "/var/graphistry"
ENV graphistry_temp_path "/tmp/graphistry"
ENV workers_port_start 10000
ENV upload_limit "128M"
ENV GRAPHISTRY_LOG_LEVEL INFO
ENV NODE_ENV production

RUN mkdir -p /var/log/graphistry-json $graphistry_temp_path $graphistry_install_path \
             /var/log/central-app /var/log/clients /var/log/worker /var/log/graphistry-json /var/log/reaper /root/.ssh

# logrotate.

COPY logrotate.conf /etc/logrotate.d/graphistry.conf

# reaper.

COPY reaper.py $graphistry_install_path

# supervisor.

COPY central.conf.envsubst viz_workers.conf.envsubst reaper.conf /etc/supervisor/conf.d/
COPY supervisord.conf /etc/supervisor
CMD ["bash", "-l", "-c", "envsubst < /etc/supervisor/conf.d/central.conf.envsubst > /etc/supervisor/conf.d/central.conf && envsubst < /etc/supervisor/conf.d/viz_workers.conf.envsubst > /etc/supervisor/conf.d/viz_workers.conf && exec supervisord -n -c /etc/supervisor/supervisord.conf"]

# config.

COPY central-cloud-options.json ${graphistry_install_path}/
COPY viz-worker-cloud-options.json ${graphistry_install_path}/viz-worker-cloud-options.json.envsubst
COPY mergeThreeFiles.js /usr/local/bin

# fetch secrets from our vault, do the npm installs, and delete the secrets.

RUN curl $(/sbin/ip route | awk '/default/ {print $3}'):17290/github/id_rsa > /root/.ssh/id_rsa && \
    curl $(/sbin/ip route | awk '/default/ {print $3}'):17290/github/id_rsa.pub > /root/.ssh/id_rsa.pub && \
    curl $(/sbin/ip route | awk '/default/ {print $3}'):17290/npm/rc > /root/.npmrc && \
    chmod 400 /root/.ssh/* && \
    md5sum /root/.ssh/* && \
    (ssh -o StrictHostKeyChecking=no -T git@github.com || true) && \
    cd $graphistry_install_path && \
    npm install @graphistry/central && \
    bash -l -c "npm install @graphistry/viz-app" && \
    git clone git@github.com:graphistry/datasets.git && \
    git clone git@github.com:graphistry/docs.git && \
    mkdir node_modules/@graphistry/central/assets/docs && \
    mv docs/* node_modules/@graphistry/central/assets/docs && \
    mv datasets/preloaded/* node_modules/@graphistry/central/assets/ && \
    rm -rf docs && \
    rm -rf datasets && \
    rm -r /tmp/* && \
    shred -u /root/.ssh/* /root/.npmrc
