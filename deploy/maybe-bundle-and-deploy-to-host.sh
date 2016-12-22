#!/bin/bash -ex

if [ -z $RELEASE ]
then
  cd dockerfiles/
  ./make-release.sh
  mv graphistry-app-* ../release.tar.gz
  cd ..
else
  if (echo $RELEASE | grep "s3://" > /dev/null)
  then
    s3cmd -c /home/ubuntu/.s3cfg --force get ${RELEASE} release.tar.gz
  else
    s3cmd -c /home/ubuntu/.s3cfg --force get s3://graphistry-releases/graphistry-app-${RELEASE}-* release.tar.gz
  fi
fi

export KEY_PATH="$(pwd)/wholly-innocuous/files/aws/ansible_id_rsa.pem"
chmod 400 "${KEY_PATH}"
ssh -T -i "${KEY_PATH}" ${BOXUSER}@${HOST} whoami

JOB_PATH=artifactdeploy/deploy-s3-artifact

mv release.tar.gz ${JOB_PATH}/files/

cd $JOB_PATH

echo "---
deploy_directory: \"$(date -u +%s).${BUILD_NUMBER}\"
ansible_ssh_user: ${BOXUSER}
ansible_ssh_private_key_file: \"${KEY_PATH}\"" > group_vars/all

envsubst < setup.yml.envsubst > setup.yml

echo $HOST > inventory

ansible-playbook -v setup.yml -i inventory
