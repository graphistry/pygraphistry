FROM mongo:2
ENV MONGO_USERNAME graphistry
ENV MONGO_PASSWORD graphtheplanet # sooper seekrit
ENV MONGO_NAME cluster
RUN mongo --eval "db.createUser({user: \"$MONGO_USERNAME\", pwd: \"$MONGO_PASSWORD\", roles: [\"readWrite\"]})" localhost/$MONGO_NAME
RUN mongo --eval '2+2' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME
RUN mongo --eval 'db.gpu_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})'  -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME
RUN mongo --eval 'db.node_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME
CMD mongo --eval '2+2' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME


