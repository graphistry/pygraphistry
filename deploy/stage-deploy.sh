#!/bin/bash

chmod 600 ansible_id_rsa.pem
echo "Checking that all local repos are up-to-date..."
OUTPUT=`./check.sh`
PULLCOUNT=`echo $OUTPUT | grep "Need to pull" | wc -l`
PUSHCOUNT=`echo $OUTPUT | grep "Need to push" | wc -l`
DIVCOUNT=`echo $OUTPUT | grep "Diverged" | wc -l`
COMMITCOUNT=`echo $OUTPUT | grep "local changes" | wc -l`
COUNT=$(($PULLCOUNT + $PUSHCOUNT + $DIVCOUNT + $COMMITCOUNT))

if [ $COUNT = "0" ] ; then
    echo "All repos up-to-date, deploying staging..."
    ansible-playbook system.yml -vv --tags fast --skip-tags provision,prod-slack -i hosts -l staging
else
    echo "$OUTPUT"
fi
