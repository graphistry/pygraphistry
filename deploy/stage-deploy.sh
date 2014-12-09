chmod 600 ansible_id_rsa.pem
echo "checking that all local repos are up to date..."
OUTPUT=`sh check.sh`
COUNT=`echo $OUTPUT | grep "Need to pull" | wc -l`
if [ $COUNT = "0" ]
then
    echo "all repos up to date, deploying staging..."
    ansible-playbook system.yml -vv --tags fast --skip-tags provision,prod-slack -i hosts -l staging
else
    sh check.sh | grep "Need to pull"
fi
