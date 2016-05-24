FROM mongo:2
ENV MONGO_USERNAME graphistry
ENV MONGO_PASSWORD graphtheplanet
ENV MONGO_NAME cluster
CMD ( for i in 1 2 3 4 5 6 7 8 9 10; do sleep $i ; (mongo --eval "2+2" -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME || (mongo --eval "db.createUser({user: '$MONGO_USERNAME', pwd: '$MONGO_PASSWORD', roles: ['readWrite']})" localhost/$MONGO_NAME && mongo --eval 'db.gpu_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})'  -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME && mongo --eval 'db.node_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME )); done) & mongod --smallfiles
