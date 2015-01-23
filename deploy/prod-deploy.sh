#!/bin/bash

read -p "Deploying to production. Are you sure [y\\n]? " -n 1 -r
echo    # (optional) move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

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
    echo "All repos up-to-date, all tests passed. Deploying production..."
    ansible-playbook system.yml -vv --tags fast --skip-tags provision,staging-slack -i hosts -l prod
else
    echo "$OUTPUT"
fi
