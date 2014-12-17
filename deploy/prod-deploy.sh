#!/bin/bash

chmod 600 ansible_id_rsa.pem
echo "Checking that all local repos are up-to-date..."
OUTPUT=`./check.sh`
PULLCOUNT=`echo $OUTPUT | grep "Need to pull" | wc -l`
PUSHCOUNT=`echo $OUTPUT | grep "Need to push" | wc -l`
DIVCOUNT=`echo $OUTPUT | grep "Diverged" | wc -l`
COUNT=$(($PULLCOUNT + $PUSHCOUNT + $DIVCOUNT))

if [ $COUNT = "0" ] ; then
    echo "All repos up-to-date, deploying production..."
    ansible-playbook system.yml -vv --tags fast --skip-tags provision,staging-slack -i hosts -l prod
else
    echo $OUTPUT
fi
