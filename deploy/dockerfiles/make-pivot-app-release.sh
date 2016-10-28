#!/bin/sh -xe
C1=graphistry/pivot-app:$1
BUCKET=s3://graphistry.releases/
for i in    $C1 ; do (docker rmi $i || true) ; docker pull $i ; done
docker save $C1 | pigz -b500 > containers.lxc.gz
for i in    $C1 ; do docker rmi $i ; done
TARBALL=graphistry-pivot-app-${1}.tar.gz
tar -cvzf $TARBALL pivot-instructions.txt containers.lxc.gz load.sh
s3cmd -c /home/ubuntu/.s3cfg put ${TARBALL} ${BUCKET}

echo SUCCESS. If you want to share this with your friends, give them the following link, which expires in a month and a half.

s3cmd -c /home/ubuntu/.s3cfg signurl ${BUCKET}${TARBALL} +4000000