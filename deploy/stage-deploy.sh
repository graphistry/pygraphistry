#!/bin/bash

chmod 600 ansible_id_rsa.pem
echo "Checking that all local repos are up-to-date and passing tests..."
OUTPUT=`./check.sh`
PULLCOUNT=`echo $OUTPUT | grep "Need to pull" | wc -l`
PUSHCOUNT=`echo $OUTPUT | grep "Need to push" | wc -l`
DIVCOUNT=`echo $OUTPUT | grep "Diverged" | wc -l`
COMMITCOUNT=`echo $OUTPUT | grep "local changes" | wc -l`
FAILURECOUNT=`echo $OUTPUT | grep "Failed" | wc -l`
COUNT=$(($PULLCOUNT + $PUSHCOUNT + $DIVCOUNT + $COMMITCOUNT + $FAILURECOUNT))

if [ $COUNT = "0" ] ; then
    echo "All repos up-to-date, all tests passed. Deploying staging..."
    ansible-playbook site.yml -i staging -vv --tags deploy
else
    echo "$OUTPUT"
fi
