For the most up-to-date deploy flow, please check out the wiki:

https://github.com/graphistry/deploy/wiki/Deployment-at-Graphistry

## Login ##

`ssh -A leo@54.183.193.90 -p 61630`

## To set up a fresh machine:##

1. Create new machine in EC2, provision with ansible_id_rsa key.

2. Log onto the box by hand:

`ssh -A -i ansible_id_rsa.pem ubuntu@XXXXXXX`

`sudo apt-get update && sudo apt-get install linux-headers-generic`

3. Exit box, go to deploy repo

4. Comment out ansible port in hosts.yml (box is started with port 22 open instead of 61630)

5. `ansible-playbook -i hosts system.yml -vvvv --tags ssh`

6. Uncomment ansible port.

7. `ansible-playbook -i hosts system.yml -vvvv --tags YOUR_BOX'S_TAG`

8. It'll reboot. Then deploy:

`ansible-playbook -i hosts system.yml -vvvv --tags YOUR_BOX'S_TAG --skip-tags provision`

9. You're done.

To start the workers/servers:
```
sudo supervisorctl start all
sudo supervisorctl stop all
sudo supervisorctl restart all
```

Logs:

`tail -f /var/log/worker/worker-10000.log`

## To Deploy:

### Staging:

`./stage-deploy.sh`

### Prod:

Fast version:

`./prod-deploy.sh`

Full version:

`ansible-playbook system.yml -vv --skip-tags provision,staging-slack -i hosts -l prod`

##Localdev (defunct for now):

```
vagrant up dev
vagrant ssh
sudo apt-get install linux-headers-generic
sudo ansible-playbook -i hosts system.yml -vvvv --tags localdev --skip-tags splunk,ssh
reboot
sudo ansible-playbook -i hosts system.yml -vvvv --tags localdev --skip-tags splunk,ssh,worker-reboot
vagrant ssh
```
