MONGO_USERNAME=graphistry
MONGO_PASSWORD=graphtheplanet # sooper seekrit
MONGO_NAME=cluster
docker run --net host mongo:2 mongo --eval "db.createUser({user: \"$MONGO_USERNAME\", pwd: \"$MONGO_PASSWORD\", roles: [\"readWrite\"]})" localhost/$MONGO_NAME
docker run --net host mongo:2 mongo --eval '2+2' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME
docker run --net host mongo:2 mongo --eval 'db.gpu_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})'  -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME
docker run --net host mongo:2 mongo --eval 'db.node_monitor.createIndex({updated: 1}, {expireAfterSeconds: 30})' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME
docker run --net host mongo:2 mongo --eval '2+2' -u $MONGO_USERNAME -p $MONGO_PASSWORD localhost/$MONGO_NAME


