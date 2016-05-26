#!/bin/sh -xe
for i in graphistry/nginx-central-vizservers:1.0.0.32 graphistry/nginx-central-vizservers:1.0.0.32.httponly graphistry/cluster-membership:1.0 graphistry/central-and-vizservers:$1 ; do docker pull $i ; done
docker save graphistry/nginx-central-vizservers:1.0.0.32 graphistry/nginx-central-vizservers:1.0.0.32.httponly graphistry/cluster-membership:1.0 graphistry/central-and-vizservers:$1 | gzip -c9 > containers.lxc.gz
sed -i -e 's_$1_'$1'_' launch.sh
tar -cvzf iqt-package.tar.gz instructions.txt ../documentation/certs.txt containers.lxc.gz load.sh launch.sh
s3cmd -c /home/ubuntu/.s3cfg put iqt-package.tar.gz s3://graphistry.releases/