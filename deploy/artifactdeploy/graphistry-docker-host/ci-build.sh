#!/bin/bash -lex
SECRETS=wholly-innocuous/files
KEY_PATH=${SECRETS}/aws/ansible_id_rsa.pem
TLS_PATH=${SECRETS}/tls
SPLUNK_UP=${SECRETS}/internalsplunk/userpassword
SPLUNK_AP=${SECRETS}/internalsplunk/adminpassword
SPLUNK_PP=${SECRETS}/internalsplunk/pivotpassword
echo "---
splunk_user_password: $(cat $SPLUNK_UP)
splunk_admin_password: $(cat $SPLUNK_AP)
splunk_deploy_server_uri: splunk.graphistry.com:8089
pivot_app_splunk_user: admin
pivot_app_splunk_key: ${SPLUNK_PP}
pivot_app_splunk_host: splunk.graphistry.com
inventory_friendlyname: ${HOSTNAMEOVERRIDE:-HOST}
vizapp_s3_access: dev1087
vizapp_s3_secret: dev1087
ansible_ssh_user: ${BOXUSER}
ansible_ssh_private_key_file: \"$(pwd)/${KEY_PATH}\"" > deploy/artifactdeploy/graphistry-docker-host/group_vars/all

cp -r ${TLS_PATH} deploy/artifactdeploy/graphistry-docker-host/files/certs

chmod 400 ${KEY_PATH}
ssh -T -i ${KEY_PATH} ${BOXUSER}@${HOST} whoami
cd deploy/artifactdeploy/graphistry-docker-host
echo "[$(echo $HOSTNAMEOVERRIDE | cut -d - -f 1)]\n$HOST" > inventory
ansible-playbook setup.yml -i inventory

