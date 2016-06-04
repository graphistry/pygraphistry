#!/bin/sh -xe
for i in graphistry/nginx-central-vizservers:1.0.0.32 graphistry/nginx-central-vizservers:1.0.0.32.httponly graphistry/cluster-membership:1.0 graphistry/central-and-vizservers:$1 ; do docker pull $i ; done
docker save graphistry/nginx-central-vizservers:1.0.0.32 graphistry/nginx-central-vizservers:1.0.0.32.httponly graphistry/cluster-membership:1.0 graphistry/central-and-vizservers:$1 | gzip -c6 > containers.lxc.gz
sed -i -e 's_$1_'$1'_' launch.sh
cp ../documentation/certs.txt .
tar -cvzf tmp.tar.gz instructions.txt certs.txt containers.lxc.gz load.sh launch.sh
SUFFIX=`sha1sum tmp.tar.gz | cut -d ' ' -f 1`
TARBALL=iqt-package-${2}-${SUFFIX}.tar.gz
mv tmp.tar.gz ${TARBALL}
s3cmd -c /home/ubuntu/.s3cfg put ${TARBALL} s3://graphistry.releases/
docker rmi graphistry/central-and-vizservers:$1 || true
