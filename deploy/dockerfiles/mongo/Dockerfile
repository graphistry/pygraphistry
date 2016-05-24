FROM mongo:2
ENV MONGO_USERNAME graphistry
ENV MONGO_PASSWORD graphtheplanet
ENV MONGO_NAME cluster
CMD mongod & \
    ( sleep 5 && \
      mongo --eval "2+2" -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME || ( \
        mongo --eval "db.createUser({user: '$MONGO_USERNAME', pwd: '$MONGO_PASSWORD', roles: ['readWrite']})" localhost/$MONGO_NAME && \
        mongo --eval 'db.gpu_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})'  -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME && \
        mongo --eval 'db.node_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME )) && \
    fg
